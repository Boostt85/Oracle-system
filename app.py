from flask import Flask, request, jsonify
import gspread
from google.oauth2.service_account import Credentials
import re
from datetime import datetime
import threading
import time

app = Flask(__name__)

CREDS_FILE = '/root/oracle/credentials.json'
SHEET_ID = '113vo-lj9SIU_RVwqbKOqCLB2wbNJWWSGQo_w4cFLC7s'
TFS = ['15s','2m','3m','5m','15m','30m','1h','4h','8h','12h','D','W']
INSTS = ['NQ','ES','RT']
MOVE_MIN = 50
MATCH_THRESH = 80

snap_cache = []
cache_lock = threading.Lock()
last_reload = 0
RELOAD_INTERVAL = 300

def get_sheet():
    scopes = ['https://spreadsheets.google.com/feeds','https://www.googleapis.com/auth/drive']
    creds = Credentials.from_service_account_file(CREDS_FILE, scopes=scopes)
    client = gspread.authorize(creds)
    return client.open_by_key(SHEET_ID)

def parse_snapshot(raw):
    s = {'raptors': {}}
    ts = re.search(r'T:(\d{4}-\d{2}-\d{2} \d{2}:\d{2})', raw)
    if not ts:
        return None
    s['timestamp'] = ts.group(1)
    p3 = re.search(r'NQ:([\d.]+)\s+ES:([\d.]+)\s+RTY:([\d.]+)\s+VOL:(\d+)', raw)
    if p3:
        s['nqPrice'] = float(p3.group(1))
        s['esPrice'] = float(p3.group(2))
        s['rtyPrice'] = float(p3.group(3))
    else:
        return None
    for tf in TFS:
        m = re.search(tf + r'\s+NQ:([\d.]+)\/([\\d.]+)\s+ES:([\d.]+)\/([\d.]+)\s+RT:([\d.]+)\/([\d.]+)', raw)
        if m:
            s['raptors']['NQ_'+tf+'_S1'] = float(m.group(1))
            s['raptors']['NQ_'+tf+'_S2'] = float(m.group(2))
            s['raptors']['ES_'+tf+'_S1'] = float(m.group(3))
            s['raptors']['ES_'+tf+'_S2'] = float(m.group(4))
            s['raptors']['RT_'+tf+'_S1'] = float(m.group(5))
            s['raptors']['RT_'+tf+'_S2'] = float(m.group(6))
    return s

def similarity(a, b):
    tot = 0
    n = 0
    for inst in INSTS:
        for tf in TFS:
            for sv in ['S1','S2']:
                key = inst+'_'+tf+'_'+sv
                va = a['raptors'].get(key)
                vb = b['raptors'].get(key)
                if va is not None and vb is not None:
                    tot += abs(va - vb)
                    n += 1
    return 0 if n == 0 else round(max(0, (1 - tot/n/100) * 100))

def load_snapshots_from_sheet():
    global last_reload
    try:
        sheet = get_sheet()
        sh1 = sheet.worksheet('Sheet1')
        rows = sh1.get_all_values()
        snaps = []
        seen = {}
        for row in rows[1:]:
            if len(row) < 2 or not row[1]:
                continue
            s = parse_snapshot(row[1])
            if not s:
                continue
            key = s['timestamp'] + '_' + str(s['nqPrice'])
            if key in seen:
                continue
            seen[key] = True
            snaps.append(s)
        snaps.sort(key=lambda x: x['timestamp'])
        compute_windows(snaps)
        with cache_lock:
            snap_cache.clear()
            snap_cache.extend(snaps)
            last_reload = time.time()
        print(f"Cache loaded: {len(snaps)} snapshots")
    except Exception as e:
        print(f"Cache load error: {e}")

def get_cached_snaps():
    if time.time() - last_reload > RELOAD_INTERVAL or not snap_cache:
        threading.Thread(target=load_snapshots_from_sheet).start()
    with cache_lock:
        return list(snap_cache)

def compute_windows(snaps):
    for i, s in enumerate(snaps):
        try:
            t0 = datetime.strptime(s['timestamp'], '%Y-%m-%d %H:%M').timestamp()
        except:
            s['windows'] = {}
            continue
        nq0 = s['nqPrice']
        future = []
        for j in range(i+1, len(snaps)):
            nx = snaps[j]
            try:
                diff = (datetime.strptime(nx['timestamp'], '%Y-%m-%d %H:%M').timestamp() - t0) / 60
            except:
                continue
            if diff > 240:
                break
            future.append((diff, nx['nqPrice']))
        s['windows'] = {}
        for w in [15, 30, 60, 240]:
            pts = [p[1] - nq0 for p in future if p[0] <= w]
            if not pts:
                s['windows']['w'+str(w)+'_max_up'] = None
                s['windows']['w'+str(w)+'_max_dn'] = None
                s['windows']['w'+str(w)+'_net'] = None
            else:
                s['windows']['w'+str(w)+'_max_up'] = round(max(pts), 2)
                s['windows']['w'+str(w)+'_max_dn'] = round(min(pts), 2)
                s['windows']['w'+str(w)+'_net'] = round(pts[-1], 2)

def run_pattern_match(snap, snaps):
    wd = snap.get('windows', {})
    mu = wd.get('w15_max_up')
    md = wd.get('w15_max_dn')
    if mu is None or md is None:
        return None
    if mu < MOVE_MIN and abs(md) < MOVE_MIN:
        return None
    net = wd.get('w15_net', 0) or 0
    direction = 'UP' if net >= 0 else 'DN'
    tdate = snap['timestamp'][:10]
    raw_matches = []
    for c in snaps:
        if c['timestamp'][:10] == tdate:
            continue
        sc = similarity(snap, c)
        if sc >= MATCH_THRESH:
            raw_matches.append({'snap': c, 'score': sc, 'dateStr': c['timestamp'][:10]})
    raw_matches.sort(key=lambda x: -x['score'])
    seen_dates = {}
    matches = []
    for m in raw_matches:
        if m['dateStr'] not in seen_dates:
            seen_dates[m['dateStr']] = True
            matches.append(m)
    if not matches:
        return None
    up = [m for m in matches if (m['snap'].get('windows', {}).get('w15_net') or 0) > 1]
    dn = [m for m in matches if (m['snap'].get('windows', {}).get('w15_net') or 0) < -1]
    up_count = len(up) + (1 if net > 1 else 0)
    dn_count = len(dn) + (1 if net < -1 else 0)
    total = len(matches) + 1
    up_pct = round(up_count / total * 100) if total else 0
    dn_pct = round(dn_count / total * 100) if total else 0
    verdict = 'MIXED'
    if up_pct >= 70:
        verdict = 'BULL ' + str(up_pct) + '%'
    elif dn_pct >= 70:
        verdict = 'BEAR ' + str(dn_pct) + '%'
    return {
        'timestamp': snap['timestamp'],
        'nqPrice': snap['nqPrice'],
        'esPrice': snap.get('esPrice'),
        'rtyPrice': snap.get('rtyPrice'),
        'direction': direction,
        'verdict': verdict,
        'match_count': total,
        'went_up': up_count,
        'went_down': dn_count,
        'best_score': matches[0]['score'],
        'best_date': matches[0]['dateStr'],
        'w15_max_up': wd.get('w15_max_up'),
        'w15_max_dn': wd.get('w15_max_dn'),
        'w15_net': wd.get('w15_net'),
        'w240_net': wd.get('w240_net'),
    }

def write_live_signal(signal):
    try:
        sheet = get_sheet()
        try:
            live = sheet.worksheet('LiveSignals')
        except:
            live = sheet.add_worksheet('LiveSignals', 1000, 20)
            live.append_row(['TIMESTAMP','NQ','ES','RTY','DIR','VERDICT','MATCHES','WENT_UP','WENT_DOWN','BEST_SCORE','BEST_DATE','W15_UP','W15_DN','W15_NET','W240_NET'])
        live.append_row([
            signal['timestamp'], signal['nqPrice'],
            signal.get('esPrice',''), signal.get('rtyPrice',''),
            signal['direction'], signal['verdict'],
            signal['match_count'], signal['went_up'], signal['went_down'],
            signal['best_score'], signal['best_date'],
            signal.get('w15_max_up',''), signal.get('w15_max_dn',''),
            signal.get('w15_net',''), signal.get('w240_net',''),
        ])
    except Exception as e:
        print(f"Write signal error: {e}")

@app.route('/webhook', methods=['POST'])
def webhook():
    try:
        raw = request.data.decode('utf-8')
        snap = parse_snapshot(raw)
        if not snap:
            return jsonify({'status': 'parse_failed'}), 200
        snaps = get_cached_snaps()
        existing = any(s['timestamp'] == snap['timestamp'] and s['nqPrice'] == snap['nqPrice'] for s in snaps)
        if not existing:
            snaps.append(snap)
            snaps.sort(key=lambda x: x['timestamp'])
            compute_windows(snaps)
            with cache_lock:
                snap_cache.clear()
                snap_cache.extend(snaps)
        current = next((s for s in snaps if s['timestamp'] == snap['timestamp'] and s['nqPrice'] == snap['nqPrice']), None)
        if not current:
            return jsonify({'status': 'snap_not_found'}), 200
        result = run_pattern_match(current, snaps)
        if result:
            verdict = result['verdict']
            direction = result['direction']
            if ('BULL' in verdict and direction == 'UP') or ('BEAR' in verdict and direction == 'DN'):
                threading.Thread(target=write_live_signal, args=(result,)).start()
                return jsonify({'status': 'signal', 'verdict': verdict, 'direction': direction}), 200
        return jsonify({'status': 'no_signal'}), 200
    except Exception as e:
        print(f"Webhook error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'cached_snaps': len(snap_cache)}), 200

@app.route('/reload', methods=['POST'])
def reload_cache():
    threading.Thread(target=load_snapshots_from_sheet).start()
    return jsonify({'status': 'reloading'}), 200

if __name__ == '__main__':
    print("Loading initial snapshot cache...")
    load_snapshots_from_sheet()
    app.run(host='0.0.0.0', port=80)
