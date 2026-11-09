"""Renderer for vn-cinematic-edit test. Python + OpenCV + PIL.
Usage: python render.py OUT.mp4 [start_s end_s]  (optional range for previews)"""
import sys, json, math, subprocess
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
import plan as P

FPS = 30; W, H = 1080, 1920; SW, SH = 2160, 3840
ROOT = '/home/user/project'
SRC = f'{ROOT}/src/1109.mp4'
FD = f'{ROOT}/assets/fonts/BeVietnamPro-'
FONTS = {k: FD + v + '.ttf' for k, v in dict(xlight='ExtraLight', light='Light', regular='Regular', bold='Bold', xbold='ExtraBold', black='Black').items()}
BRAND_RED = (32, 25, 215)  # BGR of #D71920

CAP = json.load(open(f'{ROOT}/edit/caption_plan.json'))
FACE = json.load(open(f'{ROOT}/work/face_raw.json'))
M1 = np.load(f'{ROOT}/work/mattes.npz'); M2 = np.load(f'{ROOT}/work/mattes2.npz')
MKEYS = {int(k): M1 for k in M1.files}; MKEYS.update({int(k): M2 for k in M2.files})

def chunk(si, text, nth=0):
    c = [x for x in CAP[si]['chunks'] if x['text'] == text]
    return c[nth]

# ---------------- easing ----------------
def clamp01(x): return max(0.0, min(1.0, x))
def ease(x): x = clamp01(x); return x * x * (3 - 2 * x)
def ease_out(x): x = clamp01(x); return 1 - (1 - x) ** 3
def ease_in(x): x = clamp01(x); return x ** 3
def back_out(x, s=1.7):
    x = clamp01(x) - 1; return x * x * ((s + 1) * x + s) + 1
def lerp(a, b, u): return a + (b - a) * u

# ---------------- face track (smoothed, source px) ----------------
def build_face():
    n = FACE['n']; xs = np.full(n, np.nan); ys = np.full(n, np.nan)
    for k, (x, y, h) in FACE['pts'].items():
        if h > 0.05: xs[int(k)] = x; ys[int(k)] = y
    idx = np.arange(n); ok = ~np.isnan(xs)
    xs = np.interp(idx, idx[ok], xs[ok]); ys = np.interp(idx, idx[ok], ys[ok])
    k = np.ones(31) / 31
    xs = np.convolve(np.pad(xs, 15, mode='edge'), k, 'valid'); ys = np.convolve(np.pad(ys, 15, mode='edge'), k, 'valid')
    return xs * SW, ys * SH
FX, FY = build_face()

# ---------------- key times (out) ----------------
T = dict(
    h1=chunk(0, 'thông số')['s'], h2=chunk(0, 'thương hiệu')['s'], q=chunk(0, 'đâu mới là')['s'],
    s1_brand=chunk(1, 'thương hiệu')['s'], s1_yeuto=chunk(1, 'một yếu tố')['s'], s1_quantam=chunk(1, 'quan tâm.')['s'],
    chuadu=chunk(4, 'chưa đủ')['s'], s4_brand2=chunk(4, 'thương hiệu', 1)['s'],
    thongso_kt=chunk(7, 'thông số kỹ thuật')['s'], totnhat=chunk(7, 'tốt nhất là')['s'], cahai=chunk(7, 'cả hai.')['s'],
)
LIST = [(c['text'], c['s']) for c in CAP[5]['chunks'] if c['role'] == 'list']
ITEM = {it['id']: it for it in P.ITEMS}
CUT_A2B = ITEM['v5']['t0'] + (ITEM['v5']['a_until_src'] - ITEM['v5']['src'])   # 10.82
T_END_LIST = ITEM['v8']['t1']
T_PILL = ITEM['v10']['t0'] + 0.08
T_JUMP = ITEM['v10']['t0'] + (38.0 - 37.80)
T_OUTRO = ITEM['v11']['t0']

# ---------------- zoom plan ----------------
def zoom_A(t):
    if t < 4.20:
        if t < T['h1']: return lerp(2.0, 2.15, ease(t / T['h1']))
        if t < T['h2']: return 2.6
        if t < T['q']: return 2.05
        return lerp(2.05, 2.5, ease_in((t - T['q']) / (4.20 - T['q'])))
    if t < CUT_A2B:
        if t < 7.2: return lerp(2.9, 1.9, ease_out((t - 6.60) / 0.6))
        if t < T['s1_yeuto']: return 1.9
        if t < T['s1_quantam']: return lerp(1.9, 2.05, ease((t - T['s1_yeuto']) / (T['s1_quantam'] - T['s1_yeuto'])))
        return 2.45
    it = ITEM['v7']
    if it['t0'] <= t < it['t1']:
        if t < T['chuadu']: return lerp(2.0, 2.05, (t - it['t0']) / (T['chuadu'] - it['t0']))
        if t < T['s4_brand2']: return 2.55
        return lerp(2.0, 2.15, (t - T['s4_brand2']) / (it['t1'] - T['s4_brand2']))
    it = ITEM['v9']
    if it['t0'] <= t < it['t1']:
        if t < T['thongso_kt']: return lerp(2.0, 2.15, ease((t - it['t0']) / (T['thongso_kt'] - it['t0'])))
        if t < T['totnhat']: return 2.45
        if t < T['cahai']: return lerp(2.0, 2.05, (t - T['totnhat']) / (T['cahai'] - T['totnhat']))
        return lerp(2.05, 2.45, ease_out((t - T['cahai']) / (5 / FPS)))
    it = ITEM['v10']
    return lerp(1.9, 2.05, (t - it['t0']) / (it['t1'] - it['t0']))

SHAKES = [(4.20, 14), (4.80, 8), (5.40, 14), (T['chuadu'], 12), (T['cahai'] + 5 / FPS, 14), (T_OUTRO, 8)]
def shake(t):
    dx = dy = 0.0
    for t0, amp in SHAKES:
        u = (t - t0) / 0.27
        if 0 <= u < 1:
            a = amp * (1 - u) ** 2
            dx += a * math.sin(t * 97.0 + t0); dy += a * math.cos(t * 83.0 + 2 * t0)
    return dx, dy

# whip windows: (center, left_s, right_s, kmax)  vertical motion blur
WHIPS = [(4.20, 0.13, 0.13, 80), (4.80, 0.07, 0.10, 50), (5.40, 0.07, 0.10, 50), (6.60, 0.10, 0.10, 55),
         (CUT_A2B, 0.13, 0.13, 70), (18.17, 0.22, 0.16, 80), (ITEM['v8']['t0'], 0.0, 0.13, 50),
         (ITEM['v9']['t0'], 0.13, 0.13, 70), (T_OUTRO, 0.13, 0.13, 60)]
def whip(t):
    k = 0.0
    for c, l, r, km in WHIPS:
        if c - l <= t < c and l > 0: k = max(k, km * ease_in((t - (c - l)) / l))
        elif c <= t < c + r: k = max(k, km * (1 - ease_out((t - c) / r)))
    return int(k)

ZBLUR = [(4.20 - 0.17, 4.20, 'in'), (6.60, 6.60 + 0.17, 'out'), (T['cahai'], T['cahai'] + 0.13, 'out')]
def zblur(t):
    for a, b, mode in ZBLUR:
        if a <= t < b:
            u = (t - a) / (b - a); return (u if mode == 'in' else 1 - u) * 0.10
    return 0.0

FLASHES = [(5.40, 2, 0.8), (T_JUMP, 2, 0.6), (T_OUTRO, 2, 0.4)]
def flash(t):
    for t0, nf, a in FLASHES:
        if t0 <= t < t0 + nf / FPS: return a
    return 0.0

# ---------------- grade ----------------
def make_lut():
    x = np.arange(256) / 255.0
    c = 0.5 + (x - 0.5) * 1.08
    c = np.clip(c + 0.04 * np.sin((x - 0.5) * np.pi) * 0, 0, 1)
    b = np.clip(c * 0.965 + 0.01, 0, 1); g = c; r = np.clip(c * 1.035, 0, 1)
    return np.dstack([(b * 255), (g * 255), (r * 255)]).astype(np.uint8).reshape(1, 256, 3)
LUT = make_lut()
def grade(img):
    img = cv2.LUT(img, LUT)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[..., 1] = cv2.multiply(hsv[..., 1], 1.15)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# ---------------- video readers ----------------
class Reader:
    def __init__(self, a, b, sp):
        vf = f'setpts=PTS/{sp},fps={FPS}'
        self.p = subprocess.Popen(['ffmpeg', '-loglevel', 'error', '-ss', f'{a}', '-t', f'{b - a}', '-i', SRC,
                                   '-vf', vf, '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-'], stdout=subprocess.PIPE, bufsize=10 ** 8)
        self.last = None
    def read(self):
        buf = self.p.stdout.read(SW * SH * 3)
        if len(buf) == SW * SH * 3: self.last = np.frombuffer(buf, np.uint8).reshape(SH, SW, 3)
        return self.last
    def close(self):
        try: self.p.kill()
        except Exception: pass

def warp(img, s, ox, oy, out=(W, H), src_size=(SW, SH)):
    """screen = s*(P - (ox,oy)); crops ROI first and pre-blurs for antialias when downscaling."""
    sw, sh = src_size
    x0 = max(0, int(ox) - 4); y0 = max(0, int(oy) - 4)
    x1 = min(sw, int(ox + out[0] / s) + 6); y1 = min(sh, int(oy + out[1] / s) + 6)
    roi = img[y0:y1, x0:x1]
    if s < 0.9: roi = cv2.GaussianBlur(roi, (0, 0), 0.42 / s)
    M = np.float32([[s, 0, -(ox - x0) * s], [0, s, -(oy - y0) * s]])
    return cv2.warpAffine(roi, M, out, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def frame_transform(z, anchor, t, target=(W * 0.5, H * 0.42)):
    s = 0.5 * z
    ox = anchor[0] - target[0] / s; oy = anchor[1] - target[1] / s
    ox = min(max(ox, 0), SW - W / s); oy = min(max(oy, 0), SH - H / s)
    dx, dy = shake(t)
    ox = min(max(ox + dx / s, 0), SW - W / s); oy = min(max(oy + dy / s, 0), SH - H / s)
    return s, ox, oy

# ---------------- text ----------------
_tc = {}
def text_img(txt, font, size, fill=(255, 255, 255, 255), shadow=True, track=0):
    key = (txt, font, size, fill, shadow)
    if key in _tc: return _tc[key]
    f = ImageFont.truetype(FONTS[font], size)
    bb = f.getbbox(txt)
    pad = int(size * 0.35)
    w = bb[2] - bb[0] + 2 * pad; h = int(size * 1.6) + 2 * pad
    im = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    y = pad + int(size * 0.3) - 0
    if shadow:
        sh = Image.new('RGBA', (w, h), (0, 0, 0, 0)); ds = ImageDraw.Draw(sh)
        ds.text((pad - bb[0], y + 3), txt, font=f, fill=(0, 0, 0, 170))
        a = np.array(sh); a[..., 3] = np.clip(cv2.GaussianBlur(a[..., 3], (0, 0), max(3, size / 11)).astype(np.float32) * 1.6, 0, 255).astype(np.uint8); sh = Image.fromarray(a)
        im = Image.alpha_composite(im, sh); d = ImageDraw.Draw(im)
    d.text((pad - bb[0], y), txt, font=f, fill=fill)
    arr = np.array(im)[..., [2, 1, 0, 3]].copy()
    _tc[key] = arr
    return arr

def paste(dst, rgba, cx, cy, alpha=1.0):
    h, w = rgba.shape[:2]
    x0 = int(round(cx - w / 2)); y0 = int(round(cy - h / 2))
    xa, ya = max(0, x0), max(0, y0); xb, yb = min(W, x0 + w), min(H, y0 + h)
    if xa >= xb or ya >= yb or alpha <= 0: return
    src = rgba[ya - y0:yb - y0, xa - x0:xb - x0].astype(np.float32)
    a = src[..., 3:4] / 255.0 * alpha
    reg = dst[ya:yb, xa:xb].astype(np.float32)
    dst[ya:yb, xa:xb] = (reg * (1 - a) + src[..., :3] * a).astype(np.uint8)

def anim_word(dst, img, cx, cy, t, t_on, t_off, mode='soft'):
    if t < t_on - 2 / FPS or t >= t_off: return
    ts = t - (t_on - 2 / FPS)
    alpha = 1.0; sc = 1.0; blur = 0.0; dy = 0.0
    if mode == 'soft':
        e = ease_out(ts / (6 / FPS)); alpha = e; blur = 9 * (1 - e); dy = 22 * (1 - e) - 8 * ts; sc = 1.06 - 0.06 * e
    elif mode == 'phrase':
        e = ease_out(ts / (6 / FPS)); alpha = e; blur = 7 * (1 - e); dy = 14 * (1 - e); sc = 1.04 - 0.04 * e
    elif mode == 'slam':
        alpha = clamp01(ts / (2 / FPS)); sc = 1.7 - 0.7 * back_out(ts / (6 / FPS), 2.2); dy = 0
    rem = t_off - t
    if rem < 3 / FPS: alpha *= rem / (3 / FPS)
    im = img
    if abs(sc - 1) > 0.005:
        im = cv2.resize(im, (max(1, int(im.shape[1] * sc)), max(1, int(im.shape[0] * sc))), interpolation=cv2.INTER_LINEAR)
    if blur > 0.4: im = cv2.GaussianBlur(im, (0, 0), blur)
    paste(dst, im, cx, cy + dy, alpha)

KICKER_OF = {}
STY = {1: ('black', 150), 2: ('xbold', 64), 3: ('light', 52)}
CAP_Y = 1160; HERO_Y = 540

def disp(c):
    """tier-2 keywords are ALL CAPS; others as written"""
    return c['text'].upper() if c['tier'] == 2 else c['text']

# Sentence-mode ("cả câu"): chunk index ranges shown as a building phrase block (normal + keyword on one frame).
# Chosen rule: informational sentences with >= 2 tier-2 words, outside the hook (t > 10 s), no hero inside the range.
PHRASE = {2: (0, 10), 3: (0, 9), 6: (0, 7), 7: (0, 6)}
LINE_W = 900; GAP = 16; LINE_H = 82

_pc = {}
def phrase_word(txt, tier):
    """word image on a common baseline canvas so mixed sizes/weights align on one line"""
    key = (txt, tier)
    if key in _pc: return _pc[key]
    font, size = STY[tier]
    f = ImageFont.truetype(FONTS[font], size)
    adv = int(f.getlength(txt)); pad = 18; H0 = 150; base = 100
    im = Image.new('RGBA', (adv + 2 * pad, H0), (0, 0, 0, 0))
    sh = Image.new('RGBA', im.size, (0, 0, 0, 0)); ImageDraw.Draw(sh).text((pad, base + 3), txt, font=f, fill=(0, 0, 0, 190), anchor='ls')
    a = np.array(sh); a[..., 3] = np.clip(cv2.GaussianBlur(a[..., 3], (0, 0), 5).astype(np.float32) * 1.7, 0, 255).astype(np.uint8)
    im = Image.alpha_composite(im, Image.fromarray(a))
    ImageDraw.Draw(im).text((pad, base), txt, font=f, fill=(255, 255, 255, 255), anchor='ls')
    arr = np.array(im)[..., [2, 1, 0, 3]].copy(); _pc[key] = (arr, adv, pad)
    return _pc[key]

def phrase_blocks(chunks, send):
    """greedy-wrap into blocks of <= 2 lines. Consecutive keywords (tier 2) are wrapped as one unit so
    'KIỂM SOÁT CHẤT LƯỢNG' never splits; a comma is a preferred block break."""
    units = []
    for c in chunks:
        if units and c['tier'] == 2 and units[-1][-1]['tier'] == 2: units[-1].append(c)
        else: units.append([c])
    def uw(u): return sum(phrase_word(disp(c), c['tier'])[1] for c in u) + GAP * (len(u) - 1)
    blocks, lines, lw = [], [[]], 0
    for u in units:
        need = uw(u) + (GAP if lines[-1] else 0)
        prev_comma = lines[-1] and lines[-1][-1]['text'].endswith(',') and len(lines) == 2
        if lw + need > LINE_W or prev_comma:
            if len(lines) == 2 or prev_comma:
                blocks.append(lines); lines = [[]]; lw = 0; need = uw(u)
            else:
                lines.append([]); lw = 0; need = uw(u)
        lines[-1].extend(u); lw += need
    blocks.append(lines)
    out = []
    for ls in blocks:
        ls = [l for l in ls if l]
        items = []; n = len(ls)
        for li, line in enumerate(ls):
            ws = [phrase_word(disp(c), c['tier']) for c in line]
            tw = sum(w[1] for w in ws) + GAP * (len(line) - 1)
            x = W / 2 - tw / 2
            cy = CAP_Y + (li - (n - 1) / 2) * LINE_H
            for c, (img, adv, pad) in zip(line, ws):
                items.append((img, x - pad + img.shape[1] / 2, cy, c['s'])); x += adv + GAP
        out.append([items, ls[0][0]['s'], None])
    for i in range(len(out)):
        out[i][2] = out[i + 1][1] - 0.02 if i + 1 < len(out) else send
    return out

def caption_events(t):
    """returns list of (layer, img, cx, cy, t_on, t_off, mode)"""
    ev = []
    for si, s in enumerate(CAP):
        ch = s['chunks']
        nxt = CAP[si + 1]['s'] if si + 1 < len(CAP) else 99
        send = min(s['e'] + 0.35, nxt - 0.02)
        if si == 0: send = min(send, 4.20)
        if t < s['s'] - 0.2 or t > send + 0.1: continue
        heroes = [c for c in ch if c['role'] == 'hero']
        pr = PHRASE.get(si)
        in_phrase = set(range(*pr)) if pr else set()
        caps = [(i, c) for i, c in enumerate(ch) if c['role'] == 'caption' and i not in in_phrase]
        if pr:
            pchunks = [c for i, c in enumerate(ch) if i in in_phrase and c['role'] == 'caption']
            p_end = min(send, caps[0][1]['s'] - 0.02) if caps else send
            for items, bon, boff in phrase_blocks(pchunks, p_end):
                for img, cx, cy, ton in items:
                    ev.append(('top', img, cx, cy, ton, boff, 'phrase'))
        for j, (i, c) in enumerate(caps):
            off = caps[j + 1][1]['s'] - 0.02 if j + 1 < len(caps) else send
            font, size = STY[c['tier']]
            ev.append(('top', text_img(disp(c), font, size), W / 2, CAP_Y, c['s'], off, 'soft'))
        if si == 0:   # versus stack
            k = [c for c in ch if c['role'] == 'kicker'][0]
            ev.append(('hero', text_img('thông số', 'black', 150), W / 2, 330, heroes[0]['s'], 4.20, 'soft'))
            ev.append(('top', text_img('hay là', 'light', 48), W / 2, 445, k['s'], 4.20, 'soft'))
            ev.append(('hero', text_img('thương hiệu', 'black', 150), W / 2, 560, heroes[1]['s'], 4.20, 'soft'))
        else:
            for hcv in heroes:
                img = text_img(hcv['text'].rstrip('.'), 'black', 160)
                ev.append(('hero', img, W / 2, HERO_Y, hcv['s'], send, 'soft'))
                ks = [c for c in ch if c['role'] == 'kicker']
                if ks:
                    kimg = text_img(ks[0]['text'], 'light', 48)
                    KICKER_OF[ks[0]['s']] = (hcv['s'], img.shape)
                    ev.append(('kicker', kimg, W / 2 - img.shape[1] / 2 + kimg.shape[1] / 2 + 30, HERO_Y - 95, ks[0]['s'], send, 'soft'))
    return ev

# ---------------- cards ----------------
def rounded_mask(w, h, r):
    m = Image.new('L', (w, h), 0); ImageDraw.Draw(m).rounded_rectangle([0, 0, w - 1, h - 1], r, fill=255)
    return np.array(m)

def build_label_card():
    cap = cv2.VideoCapture(SRC); cap.set(cv2.CAP_PROP_POS_MSEC, 22.85 * 1000); _, f = cap.read(); cap.release()
    crop = f[2270:2910, 930:1570]
    crop = cv2.resize(crop, (300, 300), interpolation=cv2.INTER_AREA)
    crop = grade(crop)
    bw = 6; w, h = 300 + 2 * bw, 300 + 2 * bw
    card = np.zeros((h, w, 4), np.uint8); card[..., :3] = 255
    card[bw:bw + 300, bw:bw + 300, :3] = crop
    card[..., 3] = rounded_mask(w, h, 26)
    tag = text_img('Thuận Phát ✓' if False else 'THUẬN PHÁT', 'xbold', 30, shadow=False)
    return card

def build_avatar():
    cap = cv2.VideoCapture(SRC); cap.set(cv2.CAP_PROP_POS_MSEC, 40.4 * 1000); _, f = cap.read(); cap.release()
    fi = int(40.4 * FPS); cx, cy = int(FX[fi]), int(FY[fi]); r = 190
    crop = cv2.resize(grade(f[cy - r:cy + r, cx - r:cx + r]), (76, 76), interpolation=cv2.INTER_AREA)
    m = Image.new('L', (76, 76), 0); ImageDraw.Draw(m).ellipse([0, 0, 75, 75], fill=255)
    out = np.zeros((76, 76, 4), np.uint8); out[..., :3] = crop; out[..., 3] = np.array(m)
    return out

def check_icon(d=56):
    im = Image.new('RGBA', (d, d), (0, 0, 0, 0)); dr = ImageDraw.Draw(im)
    dr.ellipse([0, 0, d - 1, d - 1], fill=(215, 25, 32, 255))
    dr.line([(d * 0.27, d * 0.52), (d * 0.44, d * 0.68), (d * 0.74, d * 0.34)], fill=(255, 255, 255, 255), width=max(3, d // 10), joint='curve')
    return np.array(im)[..., [2, 1, 0, 3]]

def build_list_item(text):
    f = ImageFont.truetype(FONTS['xbold'], 46); bb = f.getbbox(text)
    tw = bb[2] - bb[0]; h = 92; w = tw + 56 + 24 + 48
    im = Image.new('RGBA', (w + 30, h + 30), (0, 0, 0, 0))
    sh = Image.new('L', (w + 30, h + 30), 0); ImageDraw.Draw(sh).rounded_rectangle([15, 19, w + 15, h + 19], 46, fill=90)
    sh = Image.fromarray(cv2.GaussianBlur(np.array(sh), (0, 0), 8))
    im.putalpha(sh)
    shadow_layer = Image.new('RGBA', im.size, (0, 0, 0, 0)); shadow_layer.putalpha(sh)
    im = shadow_layer
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([15, 15, w + 15, h + 15], 46, fill=(255, 255, 255, 240))
    ic = Image.fromarray(check_icon(56)[..., [2, 1, 0, 3]])
    im.alpha_composite(ic, (15 + 18, 15 + 18))
    d.text((15 + 18 + 56 + 22 - bb[0], 15 + (h - (bb[3] - bb[1])) // 2 - bb[1]), text, font=f, fill=(30, 30, 30, 255))
    return np.array(im)[..., [2, 1, 0, 3]]

def pill_content(width):
    av = AVATAR
    f1 = ImageFont.truetype(FONTS['xbold'], 34); f2 = ImageFont.truetype(FONTS['light'], 24)
    im = Image.new('RGBA', (width, 104), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    im.alpha_composite(Image.fromarray(av[..., [2, 1, 0, 3]]), (14, 14))
    d.text((106, 16), 'Thuận Phát', font=f1, fill=(255, 255, 255, 255))
    d.text((106, 60), 'Chọn chất lượng – Chọn sự đẳng cấp', font=f2, fill=(255, 255, 255, 230))
    return np.array(im)[..., [2, 1, 0, 3]]

LABEL_CARD = build_label_card(); AVATAR = build_avatar()
LIST_IMGS = [build_list_item(tx) for tx, _ in LIST]
PILL_W = 640; PILL_CONTENT = pill_content(PILL_W)
HAND_SRC = (0.615 * SW, 0.455 * SH)
CARD_SCALE = 0.85   # sample board in her hand (src ~5.6s)

def draw_pill(dst, t):
    t0 = T_PILL
    if t < t0 or t >= ITEM['v10']['t1']: return
    u = (t - t0) / (8 / FPS)
    w = int(lerp(104, PILL_W, back_out(u, 1.2))) if u < 1 else PILL_W
    w = max(104, min(PILL_W + 20, w)); h = 104
    cx, cy = W // 2, 1330; x0, y0 = cx - w // 2, cy - h // 2
    reg = dst[y0:y0 + h, x0:x0 + w].astype(np.float32)
    fro = cv2.GaussianBlur(reg, (0, 0), 18) * 0.62 + 255 * 0.22
    m = rounded_mask(w, h, h // 2).astype(np.float32)[..., None] / 255
    dst[y0:y0 + h, x0:x0 + w] = (reg * (1 - m) + fro * m).astype(np.uint8)
    # border
    cv2.rectangle(dst, (0, 0), (0, 0), (0, 0, 0))
    ca = clamp01((t - t0 - 7 / FPS) / (5 / FPS))
    if ca > 0:
        cimg = PILL_CONTENT[:, :w] if w < PILL_W else PILL_CONTENT
        paste(dst, cimg, x0 + cimg.shape[1] / 2, cy, ca)

# ---------------- matte ----------------
def matte_for(src_frame, s, ox, oy):
    store = MKEYS.get(src_frame)
    if store is None: return None
    m = store[str(src_frame)]
    sm = s * 4
    M = np.float32([[sm, 0, -ox * s], [0, sm, -oy * s]])
    a = cv2.warpAffine(m, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    a = cv2.GaussianBlur(a, (0, 0), 1.5)
    return a.astype(np.float32)[..., None] / 255.0

# ---------------- hero occlusion rule ----------------
_hoff = {}
def hero_offset(img, cx, cy, key, matte, max_cov=0.12):
    """Per frame: find the smallest upward offset so the subject covers <= max_cov of the glyphs.
    Offset only ever moves up (never back down) and glides max 30px/frame."""
    k = (key, img.shape)
    cur = _hoff.get(k, 0.0)
    if matte is None: return cur
    a = img[..., 3].astype(np.float32) / 255; h, w = a.shape
    need = cur
    for off in (0, -40, -80, -120, -160, -200, -240, -280):
        if off > cur: continue
        y0 = int(cy + off - h / 2); x0 = int(cx - w / 2)
        if y0 < 120: break
        m = matte[y0:y0 + h, x0:x0 + w, 0]
        if m.shape != a.shape: continue
        need = off
        if float((a * m).sum() / max(1, a.sum())) <= max_cov: break
    new = max(need, cur - 30) if need < cur else cur
    _hoff[k] = new
    return new

# ---------------- main loop ----------------
def item_at(t):
    for it in P.ITEMS:
        if it['t0'] - 1e-6 <= t < it['t1'] - 1e-6: return it
    return P.ITEMS[-1]

def b_zoom(t, it):
    # B-roll: gentle push that restarts at each sub-shot cut
    starts = [it['t0']] + [it['t0'] + (c - it['src']) / it.get('sp', 1) for c in it.get('cuts_src', [])]
    st = max([x for x in starts if x <= t + 1e-6])
    return 1.0 + 0.035 * (t - st)

def render(out_path, t_start=0.0, t_end=None):
    t_end = t_end or P.DURATION
    enc = subprocess.Popen(['ffmpeg', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}',
                            '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-preset', 'medium', '-crf', '17', '-pix_fmt', 'yuv420p',
                            out_path], stdin=subprocess.PIPE)
    readers = {}; cur = None; last_v5 = None
    f0, f1 = int(round(t_start * FPS)), int(round(t_end * FPS))
    for fi in range(f0, f1):
        t = fi / FPS
        it = item_at(t)
        if it['id'] != cur:
            for r in readers.values(): r.close()
            readers = {}; cur = it['id']
            off = t - it['t0']
            if it['kind'] == 'split':
                readers['a'] = Reader(it['srcA'][0] + off, it['srcA'][1], 1)
                readers['b'] = Reader(it['srcB'][0] + off, it['srcB'][1], 1)
            elif it['kind'] != 'hold':
                sp = it.get('sp', 1)
                readers['v'] = Reader(it['src'] + off * sp, it['end'], sp)
        src_frame = None; matte = None
        if it['kind'] == 'split':
            fa, fb = readers['a'].read(), readers['b'].read()
            u = (t - it['t0']) / (it['t1'] - it['t0']); z = 1.0 + 0.06 * u
            frame = np.empty((H, W, 3), np.uint8)
            for half, img, cyf in ((0, fa, 0.42), (1, fb, 0.5)):
                s = 0.5 * z; vh = (H / 2) / s; ox = (SW - W / s) / 2; oy = min(max(cyf * SH - vh / 2, 0), SH - vh)
                frame[half * H // 2:(half + 1) * H // 2] = warp(img, s, ox, oy, out=(W, H // 2))
            frame = grade(frame)
        else:
            if it['kind'] == 'hold':
                img = last_v5; z = 1.08 + 0.3 * (t - it['t0'])
                s, ox, oy = 0.5 * z, (SW - W / (0.5 * z)) / 2, (SH - H / (0.5 * z)) / 2
            else:
                img = readers['v'].read()
                if it['id'] == 'v5': last_v5 = img
                sp = it.get('sp', 1); src_t = it['src'] + (t - it['t0']) * sp
                src_frame = int(round(src_t * FPS))
                is_a = it['kind'] == 'A' and not (it['id'] == 'v5' and src_t >= it['a_until_src'])
                if is_a:
                    z = zoom_A(t); anchor = (FX[min(src_frame, len(FX) - 1)], FY[min(src_frame, len(FY) - 1)])
                    s, ox, oy = frame_transform(z, anchor, t)
                else:
                    z = b_zoom(t, it); s = 0.5 * z
                    ox, oy = (SW - W / s) / 2, (SH - H / s) / 2
                    dx, dy = shake(t); ox += dx / s; oy += dy / s
                    ox = min(max(ox, 0), SW - W / s); oy = min(max(oy, 0), SH - H / s)
            frame = grade(warp(img, s, ox, oy))
            if src_frame is not None and it['kind'] == 'A':
                matte = matte_for(src_frame, s, ox, oy)
        # zoom blur
        zb = zblur(t)
        if zb > 0.005:
            acc = frame.astype(np.float32)
            for k in range(1, 5):
                sc = 1 + zb * k / 4; big = cv2.resize(frame, None, fx=sc, fy=sc)
                y0 = (big.shape[0] - H) // 2; x0 = (big.shape[1] - W) // 2
                acc += big[y0:y0 + H, x0:x0 + W]
            frame = (acc / 5).astype(np.uint8)
        # whip blur
        k = whip(t)
        if k > 2:
            ker = np.zeros((k, 1), np.float32); ker[:] = 1.0 / k
            frame = cv2.filter2D(frame, -1, ker)
        if 4.20 <= t < 5.40:
            frame = (frame.astype(np.float32) * 0.72).astype(np.uint8)
        video = frame.copy()
        # spotlight (question in hook)
        if T['q'] <= t < 4.20 and matte is not None:
            u = ease((t - T['q']) / (6 / FPS))
            bg = cv2.GaussianBlur(frame, (0, 0), 5).astype(np.float32) * (1 - 0.42 * u)
            frame = (frame.astype(np.float32) * matte + (bg * u + frame.astype(np.float32) * (1 - u)) * (1 - matte)).astype(np.uint8)
            video = frame.copy()
        # end card dim
        if it['id'] == 'v11':
            u = ease((t - T_OUTRO) / 0.3)
            frame = (frame.astype(np.float32) * (1 - 0.4 * u)).astype(np.uint8); video = frame.copy()
        # ---- text layers ----
        ev = caption_events(t)
        sdx, sdy = shake(t)
        for layer, img, cx, cy, ton, toff, mode in ev:
            if layer == 'hero':
                if ton > 4.20: cy = cy + hero_offset(img, cx, cy, ton, matte)   # versus stack keeps fixed layout
                anim_word(frame, img, cx + sdx * 0.5, cy + sdy * 0.5, t, ton, toff, mode)
        if matte is not None:
            frame = (frame.astype(np.float32) * (1 - matte) + video.astype(np.float32) * matte).astype(np.uint8)
        for layer, img, cx, cy, ton, toff, mode in ev:
            if layer == 'kicker':
                cy = cy + _hoff.get(KICKER_OF[ton], 0.0)
                anim_word(frame, img, cx, cy, t, ton, toff, mode); continue
            if layer == 'top':
                if ITEM['v8']['t0'] <= t < T_END_LIST and ton >= LIST[0][1] - 0.05: continue
                anim_word(frame, img, cx, cy, t, ton, toff, mode)
        # drop slams
        if 4.20 <= t < 4.80: anim_word(frame, text_img('thương hiệu', 'black', 150), W / 2 + sdx, 900 + sdy, t, 4.20 + 2 / FPS, 4.80, 'slam')
        if 4.80 <= t < 5.40: anim_word(frame, text_img('thông số', 'black', 160), W / 2 + sdx, 900 + sdy, t, 4.80 + 2 / FPS, 5.40, 'slam')
        if 5.40 <= t < 6.60:
            cv2.rectangle(frame, (0, H // 2 - 3), (W, H // 2 + 3), (255, 255, 255), -1)
            anim_word(frame, TAG_BRAND, W / 2, 300, t, 5.40 + 4 / FPS, 6.60, 'soft')
            anim_word(frame, TAG_SPEC, W / 2, 1400, t, 5.40 + 7 / FPS, 6.60, 'soft')
            if t >= 5.40:
                vs = VS_BADGE
                anim_word(frame, vs, W / 2 + sdx, H / 2 + sdy, t, 5.40 + 2 / FPS, 6.60, 'slam')
        # proof card on hand
        t_card0, t_card1 = T['s1_brand'], T['s1_brand'] + 1.7
        if t_card0 - 2 / FPS <= t < t_card1 and it['id'] == 'v5':
            hx = (HAND_SRC[0] - ox) * s; hy = (HAND_SRC[1] - oy) * s
            ts = t - (t_card0 - 2 / FPS)
            sc = 0.55 + 0.45 * back_out(ts / (6 / FPS), 2.0)
            al = clamp01(ts / (2 / FPS)) * clamp01((t_card1 - t) / (4 / FPS))
            img = LABEL_CARD
            if ts < 1.5 / FPS:
                img = img.copy(); img[..., :3] = 255
            img = cv2.resize(img, None, fx=sc * CARD_SCALE, fy=sc * CARD_SCALE)
            paste(frame, img, hx + 70, hy - img.shape[0] / 2 - 30 - 6 * math.sin(ts * 3), al)
        # spec checklist
        if ITEM['v8']['t0'] <= t < T_END_LIST + 0.01:
            for i, ((tx, ts0), img) in enumerate(zip(LIST, LIST_IMGS)):
                if t < ts0 - 2 / FPS: continue
                e = ease_out((t - ts0 + 2 / FPS) / (7 / FPS))
                al = e * clamp01((T_END_LIST - t) / (4 / FPS))
                paste(frame, img, 60 + img.shape[1] / 2 - 40 * (1 - e), 930 + i * 104, al)
        # contact pill
        draw_pill(frame, t)
        # end card
        if it['id'] == 'v11':
            cv2.rectangle(frame, (W // 2 - 70, 760), (W // 2 + 70, 768), BRAND_RED, -1) if t > T_OUTRO + 0.15 else None
            anim_word(frame, text_img('THUẬN PHÁT', 'black', 128), W / 2, 870, t, T_OUTRO + 0.12, 99, 'soft')
            anim_word(frame, text_img('Chọn chất lượng – Chọn sự đẳng cấp', 'light', 46), W / 2, 975, t, T_OUTRO + 0.45, 99, 'soft')
            u = clamp01((t - (P.DURATION - 0.45)) / 0.45)
            if u > 0: frame = (frame.astype(np.float32) * (1 - u)).astype(np.uint8)
        fa = flash(t)
        if fa > 0: frame = cv2.addWeighted(frame, 1 - fa, np.full_like(frame, 255), fa, 0)
        enc.stdin.write(frame.tobytes())
        if fi % 60 == 0: print(f'{t:6.2f}s  {it["id"]}', flush=True)
    for r in readers.values(): r.close()
    enc.stdin.close(); enc.wait()

def build_vs():
    d = 200; im = Image.new('RGBA', (d, d), (0, 0, 0, 0)); dr = ImageDraw.Draw(im)
    dr.ellipse([6, 6, d - 7, d - 7], fill=(215, 25, 32, 255), outline=(255, 255, 255, 255), width=6)
    f = ImageFont.truetype(FONTS['black'], 84); bb = f.getbbox('VS')
    dr.text(((d - (bb[2] - bb[0])) / 2 - bb[0], (d - (bb[3] - bb[1])) / 2 - bb[1]), 'VS', font=f, fill=(255, 255, 255, 255))
    return np.array(im)[..., [2, 1, 0, 3]]
VS_BADGE = build_vs()

def build_tag(txt):
    f = ImageFont.truetype(FONTS['xbold'], 64); bb = f.getbbox(txt)
    w, h = bb[2] - bb[0] + 80, 112
    im = Image.new('RGBA', (w, h), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    d.rounded_rectangle([0, 0, w - 1, h - 1], h // 2, fill=(215, 25, 32, 255))
    d.text((40 - bb[0], (h - (bb[3] - bb[1])) / 2 - bb[1]), txt, font=f, fill=(255, 255, 255, 255))
    return np.array(im)[..., [2, 1, 0, 3]]
TAG_BRAND = build_tag('THƯƠNG HIỆU'); TAG_SPEC = build_tag('THÔNG SỐ')

if __name__ == '__main__':
    out = sys.argv[1]
    a = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    b = float(sys.argv[3]) if len(sys.argv) > 3 else None
    render(out, a, b)
