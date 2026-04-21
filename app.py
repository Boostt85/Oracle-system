"""
Oracle Real-Time Pattern Matching Server v2
-------------------------------------------
Refactored for memory efficiency: NumPy vectorized matching,
no async threads, Discord webhook alerts.

DigitalOcean: 67.205.152.231
Deploy: replace /root/oracle/app.py, restart oracle.service
"""

from flask import Flask, request, jsonify
from google.oauth2.service_account import Credentials
import gspread
import numpy as np
import re
import time
import threading
import requests as http_requests
from datetime import datetime, timedelta

app = Flask(__name__)

# ─── CONFIG ────────────────────────────────────────────────────────
CREDS_FILE = '/root/oracle/credentials.json'
SHEET_ID = '113vo-lj9SIU_RVwqbKOqCLB2wbNJWWSGQo_w4cFLC7s'

TFS = ['15s', '2m', '3m', '5m', '15m', '30m', '1h', '4h', '8h', '12h', 'D', 'W']
INSTS = ['NQ', 'ES', 'RT']

# Build ordered key list for consistent NumPy column mapping
# 72 columns: 3 instruments × 12 timeframes × 2 values (S1, S2)
RAPTOR_KEYS = []
for inst in INSTS:
    for tf in TFS:
        for sv in ['S1', 'S2']:
            RAPTOR_KEYS.append(f"{inst}_{tf}_{sv}")
NUM_FEATURES = len(RAPTOR_KEYS)  # 72

MOVE_MIN = 50
MATCH_THRESH = 80
COOLDOWN_MINUTES = 30
RELOAD_INTERVAL = 300

# ─── DISCORD CONFIG ────────────────────────────────────────────────
# Paste your Discord webhook URL here
DISCORD_WEBHOOK_URL = ''  # <-- SET THIS ON THE SERVER

# ─── CACHE (NumPy-based) ──────────────────────────────────────────
# raptor_matrix: shape (N, 72) float64 — the STC values
# meta: list of dicts with timestamp, prices, date_str, windows
# These are the ONLY data structures that persist in memory.
raptor_matrix = np.empty((0, NUM_FEATURES), dtype=np.float64)
meta = []
cache_lock = threading.Lock()
last_reload = 0

# Cooldown tracking: {('BULL','2026-04-21'): last_signal_timestamp, ...}
cooldowns = {}


# ─── GOOGLE SHEETS ─────────────────────────────────────────────────
def get_sheet():
    scopes = [
        'https://spreadsheets.google.com/feeds',
        'https://www.googleapis.com/auth/drive'
    ]
    creds = Credentials.from_service_account_file(CREDS_FILE, scopes=scopes)
    client = gspread.authorize(creds)
    return client.open_by_key(SHEET_ID)


# ─── PARSING ───────────────────────────────────────────────────────
def parse_snapshot(raw):
    """Parse raw TradingView alert text into structured snapshot.
    Returns (meta_dict, raptor_array_72) or (None, None)."""

    ts_match = re.search(r'T:(\d{4}-\d{2}-\d{2} \d{2}:\d{2})', raw)
    if not ts_match:
        return None, None

    price_match = re.search(
        r'NQ:([\d.]+)\s+ES:([\d.]+)\s+RTY:([\d.]+)\s+VOL:(\d+)', raw
    )
    if not price_match:
        return None, None

    timestamp = ts_match.group(1)
    nq_price = float(price_match.group(1))
    es_price = float(price_match.group(2))
    rty_price = float(price_match.group(3))

    # Parse 72 raptor values into a flat array
    raptor_vals = np.full(NUM_FEATURES, np.nan, dtype=np.float64)

    col = 0
    for tf in TFS:
        m = re.search(
            tf + r'\s+NQ:([\d.]+)\/([\d.]+)\s+ES:([\d.]+)\/([\d.]+)\s+RT:([\d.]+)\/([\d.]+)',
            raw
        )
        if m:
            # NQ S1, S2
            raptor_vals[col] = float(m.group(1))
            raptor_vals[col + 1] = float(m.group(2))
            # ES S1, S2
            raptor_vals[col + 2] = float(m.group(3))
            raptor_vals[col + 3] = float(m.group(4))
            # RT S1, S2
            raptor_vals[col + 4] = float(m.group(5))
            raptor_vals[col + 5] = float(m.group(6))
        col += 6  # Always advance 6 (3 instruments × 2 values)

    meta_dict = {
        'timestamp': timestamp,
        'date_str': timestamp[:10],
        'nqPrice': nq_price,
        'esPrice': es_price,
        'rtyPrice': rty_price,
        'dedup_key': f"{timestamp}_{nq_price}",
        'windows': {},
    }

    return meta_dict, raptor_vals


# ─── WINDOW COMPUTATION ───────────────────────────────────────────
def compute_windows_for_cache():
    """Compute 15m/30m/1h/4h forward windows for all snapshots in meta."""
    global meta
    n = len(meta)
    if n == 0:
        return

    # Pre-compute timestamps as floats for fast math
    ts_floats = []
    for m in meta:
        try:
            ts_floats.append(
                datetime.strptime(m['timestamp'], '%Y-%m-%d %H:%M').timestamp()
            )
        except:
            ts_floats.append(0)
    ts_arr = np.array(ts_floats, dtype=np.float64)
    nq_arr = np.array([m['nqPrice'] for m in meta], dtype=np.float64)

    for i in range(n):
        if ts_arr[i] == 0:
            meta[i]['windows'] = {}
            continue

        # Find all future snapshots within 240 minutes
        diffs = (ts_arr[i + 1:] - ts_arr[i]) / 60.0
        mask_240 = diffs <= 240
        if not np.any(mask_240):
            meta[i]['windows'] = {
                f'w{w}_max_up': None, f'w{w}_max_dn': None, f'w{w}_net': None
                for w in [15, 30, 60, 240]
            }
            continue

        future_diffs = diffs[mask_240]
        future_prices = nq_arr[i + 1:][mask_240] - nq_arr[i]

        windows = {}
        for w in [15, 30, 60, 240]:
            w_mask = future_diffs <= w
            if not np.any(w_mask):
                windows[f'w{w}_max_up'] = None
                windows[f'w{w}_max_dn'] = None
                windows[f'w{w}_net'] = None
            else:
                pts = future_prices[w_mask]
                windows[f'w{w}_max_up'] = round(float(np.max(pts)), 2)
                windows[f'w{w}_max_dn'] = round(float(np.min(pts)), 2)
                windows[f'w{w}_net'] = round(float(pts[-1]), 2)
        meta[i]['windows'] = windows


# ─── CACHE LOADING ─────────────────────────────────────────────────
def load_snapshots_from_sheet():
    """Load snapshots from Sheet1 into NumPy matrix + meta list."""
    global raptor_matrix, meta, last_reload

    try:
        sheet = get_sheet()
        sh1 = sheet.worksheet('Sheet1')
        rows = sh1.get_all_values()

        temp_meta = []
        temp_raptors = []
        seen = set()

        for row in rows[1:]:
            if len(row) < 2 or not row[1]:
                continue
            m, r = parse_snapshot(row[1])
            if m is None:
                continue
            if m['dedup_key'] in seen:
                continue
            seen.add(m['dedup_key'])
            temp_meta.append(m)
            temp_raptors.append(r)

        if not temp_meta:
            print("No snapshots found in sheet")
            return

        # Sort by timestamp
        indices = sorted(range(len(temp_meta)), key=lambda i: temp_meta[i]['timestamp'])
        temp_meta = [temp_meta[i] for i in indices]
        temp_raptors = [temp_raptors[i] for i in indices]

        # Build NumPy matrix
        new_matrix = np.array(temp_raptors, dtype=np.float64)

        # Temporarily assign for window computation
        old_meta = meta
        old_matrix = raptor_matrix
        meta = temp_meta
        compute_windows_for_cache()

        with cache_lock:
            raptor_matrix = new_matrix
            # meta is already assigned
            last_reload = time.time()

        mem_mb = raptor_matrix.nbytes / (1024 * 1024)
        print(f"Cache loaded: {len(meta)} snapshots, raptor matrix: {mem_mb:.1f} MB")

    except Exception as e:
        print(f"Cache load error: {e}")
        import traceback
        traceback.print_exc()


# ─── VECTORIZED PATTERN MATCHING ──────────────────────────────────
def find_matches(incoming_raptors, incoming_meta):
    """
    Compare incoming snapshot against entire cache using vectorized NumPy.
    Returns signal dict or None.
    """
    with cache_lock:
        if len(meta) == 0:
            return None
        # Snapshot the references (no copy of the big matrix needed)
        mat = raptor_matrix
        mta = meta

    n = mat.shape[0]
    incoming_date = incoming_meta['date_str']

    # ── Vectorized similarity: mean absolute difference across 72 values ──
    # Shape: (N, 72) - (1, 72) = (N, 72) → mean across axis 1 → (N,)
    incoming_vec = incoming_raptors.reshape(1, NUM_FEATURES)

    # Handle NaN: replace NaN with 50 (neutral) for both sides
    mat_clean = np.where(np.isnan(mat), 50.0, mat)
    inc_clean = np.where(np.isnan(incoming_vec), 50.0, incoming_vec)

    abs_diff = np.abs(mat_clean - inc_clean)
    mean_diff = np.mean(abs_diff, axis=1)
    scores = np.maximum(0, (1 - mean_diff / 100) * 100).astype(np.int32)

    # ── Filter: cross-day only, above threshold ──
    # Build date mask
    date_mask = np.array([m['date_str'] != incoming_date for m in mta], dtype=bool)
    thresh_mask = scores >= MATCH_THRESH
    valid_mask = date_mask & thresh_mask

    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) == 0:
        return None

    valid_scores = scores[valid_indices]

    # ── One best match per session day ──
    # Sort by score descending
    sort_order = np.argsort(-valid_scores)
    seen_dates = set()
    matches = []

    for idx in sort_order:
        cache_idx = valid_indices[idx]
        match_date = mta[cache_idx]['date_str']
        if match_date in seen_dates:
            continue
        seen_dates.add(match_date)
        matches.append({
            'cache_idx': int(cache_idx),
            'score': int(valid_scores[idx]),
            'date_str': match_date,
            'windows': mta[cache_idx].get('windows', {}),
        })

    if not matches:
        return None

    # ── Check qualifying move (50pt in 15m window) ──
    iw = incoming_meta.get('windows', {})
    mu = iw.get('w15_max_up')
    md = iw.get('w15_max_dn')
    if mu is None or md is None:
        return None
    if mu < MOVE_MIN and abs(md) < MOVE_MIN:
        return None

    net = iw.get('w15_net', 0) or 0
    direction = 'UP' if net >= 0 else 'DN'

    # ── Verdict calculation (signal bar counts in outcome) ──
    up_count = 1 if net > 1 else 0
    dn_count = 1 if net < -1 else 0
    for m in matches:
        w = m['windows']
        mnet = w.get('w15_net')
        if mnet is not None:
            if mnet > 1:
                up_count += 1
            elif mnet < -1:
                dn_count += 1

    total = len(matches) + 1  # +1 for signal bar itself
    up_pct = round(up_count / total * 100) if total else 0
    dn_pct = round(dn_count / total * 100) if total else 0

    verdict = 'MIXED'
    if up_pct >= 70:
        verdict = f'BULL {up_pct}%'
    elif dn_pct >= 70:
        verdict = f'BEAR {dn_pct}%'

    return {
        'timestamp': incoming_meta['timestamp'],
        'nqPrice': incoming_meta['nqPrice'],
        'esPrice': incoming_meta.get('esPrice'),
        'rtyPrice': incoming_meta.get('rtyPrice'),
        'direction': direction,
        'verdict': verdict,
        'match_count': total,
        'went_up': up_count,
        'went_down': dn_count,
        'best_score': matches[0]['score'],
        'best_date': matches[0]['date_str'],
        'w15_max_up': mu,
        'w15_max_dn': md,
        'w15_net': net,
        'w240_net': iw.get('w240_net'),
    }


# ─── COOLDOWN CHECK ───────────────────────────────────────────────
def check_cooldown(verdict, timestamp):
    """30-minute cooldown per direction, resets per calendar day."""
    if 'BULL' in verdict:
        direction = 'BULL'
    elif 'BEAR' in verdict:
        direction = 'BEAR'
    else:
        return False  # MIXED doesn't alert

    date_str = timestamp[:10]
    key = (direction, date_str)

    try:
        current_time = datetime.strptime(timestamp, '%Y-%m-%d %H:%M')
    except:
        return False

    last_time = cooldowns.get(key)
    if last_time:
        elapsed = (current_time - last_time).total_seconds() / 60
        if elapsed < COOLDOWN_MINUTES:
            return False  # Still in cooldown

    cooldowns[key] = current_time

    # Clean old dates from cooldown dict
    old_keys = [k for k in cooldowns if k[1] != date_str]
    for k in old_keys:
        del cooldowns[k]

    return True


# ─── DISCORD ALERT ─────────────────────────────────────────────────
def send_discord_alert(signal):
    """Send BULL/BEAR signal to Discord channel."""
    if not DISCORD_WEBHOOK_URL:
        print("Discord webhook URL not set — skipping alert")
        return

    verdict = signal['verdict']
    if 'BULL' in verdict:
        color = 0x00FF00  # Green
        emoji = '🟢'
        action = 'BUY SIGNAL'
    else:
        color = 0xFF0000  # Red
        emoji = '🔴'
        action = 'SELL SIGNAL'

    embed = {
        'title': f'{emoji} {action} — {verdict}',
        'color': color,
        'fields': [
            {'name': 'NQ', 'value': str(signal['nqPrice']), 'inline': True},
            {'name': 'ES', 'value': str(signal.get('esPrice', '')), 'inline': True},
            {'name': 'RTY', 'value': str(signal.get('rtyPrice', '')), 'inline': True},
            {'name': 'Direction', 'value': signal['direction'], 'inline': True},
            {'name': 'Best Match', 'value': f"{signal['best_score']}% ({signal['best_date']})", 'inline': True},
            {'name': 'Cross-Day Matches', 'value': str(signal['match_count']), 'inline': True},
            {'name': 'Went Up / Down', 'value': f"{signal['went_up']} / {signal['went_down']}", 'inline': True},
            {'name': '15m Net', 'value': str(signal.get('w15_net', '')), 'inline': True},
            {'name': '4h Net', 'value': str(signal.get('w240_net', '')), 'inline': True},
        ],
        'footer': {'text': f"Oracle System • {signal['timestamp']}"},
    }

    payload = {
        'username': 'Oracle System',
        'embeds': [embed],
    }

    try:
        resp = http_requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=5)
        if resp.status_code in (200, 204):
            print(f"Discord alert sent: {verdict}")
        else:
            print(f"Discord error: {resp.status_code} {resp.text}")
    except Exception as e:
        print(f"Discord send failed: {e}")


# ─── SHEET WRITE ───────────────────────────────────────────────────
def write_live_signal(signal):
    """Write qualified signal to LiveSignals tab."""
    try:
        sheet = get_sheet()
        try:
            live = sheet.worksheet('LiveSignals')
        except:
            live = sheet.add_worksheet('LiveSignals', 1000, 20)
            live.append_row([
                'TIMESTAMP', 'NQ', 'ES', 'RTY', 'DIR', 'VERDICT',
                'MATCHES', 'WENT_UP', 'WENT_DOWN', 'BEST_SCORE',
                'BEST_DATE', 'W15_UP', 'W15_DN', 'W15_NET', 'W240_NET'
            ])
        live.append_row([
            signal['timestamp'], signal['nqPrice'],
            signal.get('esPrice', ''), signal.get('rtyPrice', ''),
            signal['direction'], signal['verdict'],
            signal['match_count'], signal['went_up'], signal['went_down'],
            signal['best_score'], signal['best_date'],
            signal.get('w15_max_up', ''), signal.get('w15_max_dn', ''),
            signal.get('w15_net', ''), signal.get('w240_net', ''),
        ])
    except Exception as e:
        print(f"Write signal error: {e}")


# ─── WEBHOOK HANDLER (INLINE, NO THREADS) ─────────────────────────
@app.route('/webhook', methods=['POST'])
def webhook():
    try:
        raw = request.data.decode('utf-8')

        # Parse incoming snapshot
        snap_meta, snap_raptors = parse_snapshot(raw)
        if snap_meta is None:
            return jsonify({'status': 'parse_error'}), 200

        # Check dedup against cache
        with cache_lock:
            existing = any(
                m['dedup_key'] == snap_meta['dedup_key'] for m in meta
            )

        if not existing:
            # Add to cache inline
            with cache_lock:
                meta.append(snap_meta)
                global raptor_matrix
                raptor_matrix = np.vstack([raptor_matrix, snap_raptors.reshape(1, -1)])

            # Compute windows for recent snapshots only (last 240 min)
            # Full recompute happens on reload cycle
            _compute_window_for_latest()

        # Run pattern matching — vectorized, <50ms
        result = find_matches(snap_raptors, snap_meta)

        if result:
            verdict = result['verdict']
            direction = result['direction']

            is_bull_up = 'BULL' in verdict and direction == 'UP'
            is_bear_dn = 'BEAR' in verdict and direction == 'DN'

            if is_bull_up or is_bear_dn:
                # Check cooldown
                if check_cooldown(verdict, result['timestamp']):
                    # Write to Google Sheet
                    write_live_signal(result)
                    # Send Discord alert
                    send_discord_alert(result)
                    print(f"SIGNAL: {verdict} {direction} at {snap_meta['timestamp']}")
                else:
                    print(f"COOLDOWN: {verdict} {direction} at {snap_meta['timestamp']} (skipped)")

        return jsonify({'status': 'ok'}), 200

    except Exception as e:
        print(f"Webhook error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500


def _compute_window_for_latest():
    """Compute windows for the most recently added snapshot only."""
    with cache_lock:
        if len(meta) < 2:
            return
        idx = len(meta) - 1
        current = meta[idx]

    try:
        t0 = datetime.strptime(current['timestamp'], '%Y-%m-%d %H:%M').timestamp()
    except:
        return

    # This new snapshot won't have future data yet, so windows are empty.
    # But we can update PREVIOUS snapshots whose windows now extend through this bar.
    # For now, just mark current as empty — full recompute on 5-min reload.
    current['windows'] = {
        f'w{w}_max_up': None, f'w{w}_max_dn': None, f'w{w}_net': None
        for w in [15, 30, 60, 240]
    }


# ─── BACKGROUND RELOAD ────────────────────────────────────────────
def background_reload_loop():
    """Reload cache from Google Sheets every 5 minutes."""
    while True:
        time.sleep(RELOAD_INTERVAL)
        try:
            print("Background reload starting...")
            load_snapshots_from_sheet()
        except Exception as e:
            print(f"Background reload error: {e}")


# ─── HEALTH / RELOAD ENDPOINTS ────────────────────────────────────
@app.route('/health', methods=['GET'])
def health():
    with cache_lock:
        count = len(meta)
        mem_mb = raptor_matrix.nbytes / (1024 * 1024)
    return jsonify({
        'status': 'ok',
        'cached_snaps': count,
        'raptor_matrix_mb': round(mem_mb, 2),
        'last_reload': round(time.time() - last_reload),
        'discord_configured': bool(DISCORD_WEBHOOK_URL),
    }), 200


@app.route('/reload', methods=['POST'])
def reload_cache():
    threading.Thread(target=load_snapshots_from_sheet, daemon=True).start()
    return jsonify({'status': 'reloading'}), 200


# ─── STARTUP ───────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Oracle Server v2 — NumPy pattern matching + Discord alerts")
    print(f"Features: {NUM_FEATURES} values, threshold {MATCH_THRESH}%, move min {MOVE_MIN}pt")
    print(f"Discord: {'configured' if DISCORD_WEBHOOK_URL else 'NOT SET — edit DISCORD_WEBHOOK_URL'}")
    print("Loading initial snapshot cache...")
    load_snapshots_from_sheet()
    print(f"Starting background reload every {RELOAD_INTERVAL}s...")

    # Start background reload thread
    reload_thread = threading.Thread(target=background_reload_loop, daemon=True)
    reload_thread.start()

    app.run(host='0.0.0.0', port=80)
