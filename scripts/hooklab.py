"""Hook lab: 5 visual-hook variants on the first 10.1 s of 1109.mp4 (timeline == source time).
Variants differ in 0–4.98 s; 4.98–10.1 s is a shared body so hooks can be compared fairly.
Usage: python hooklab.py V1|V2|V3|V4|V5|all"""
import sys, math, json, subprocess
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
import render as R
import plan as P
sys.path.insert(0, '/home/user/vn-cinematic-edit/scripts')
import sfx_lib as L
_M3 = np.load('/home/user/project/work/mattes3.npz'); R.MKEYS.update({int(k): _M3 for k in _M3.files})

FPS, W, H, SW, SH = 30, 1080, 1920, 2160, 3840
DUR = 10.1; NF = int(DUR * FPS)
ROOT = '/home/user/project'
OUT = f'{ROOT}/out/hooks'
ease, ease_out, ease_in, back_out, lerp, c01 = R.ease, R.ease_out, R.ease_in, R.back_out, R.lerp, R.clamp01

# ---------------- words / chunks (src time) ----------------
WORDS = json.load(open(f'{ROOT}/work/words_raw.json'))
def chunks_src():
    k = 0; out = []
    for si, sent in enumerate(P.SENTENCES[:3]):
        cs = []
        for text, n, tier, role in sent:
            ws = WORDS[k:k + n]; k += n
            cs.append(dict(text=text, tier=tier, role=role, s=ws[0]['s'], e=ws[-1]['e']))
        out.append(cs)
    return out
CH = chunks_src()
def cs(si, text, nth=0): return [c for c in CH[si] if c['text'] == text][nth]['s']
TW = dict(mdf=cs(0, 'MDF'), h1=cs(0, 'thông số'), h2=cs(0, 'thương hiệu'), q=cs(0, 'đâu mới là'), qt=cs(0, 'quan trọng?'),
          s1=cs(1, 'Trước hết,'), brand=cs(1, 'thương hiệu'), yeuto=cs(1, 'một yếu tố'), quantam=cs(1, 'quan tâm.'))
GAP0, GAP1 = 4.10, TW['s1'] - 0.0     # silence between sentence 0 and 1
CUT_FK = 9.0667

def disp(c): return c['text'].upper() if c['tier'] in (1, 2) else c['text']

def caption_list(si, y, skip=()):
    """chunk-mode caption events for sentence si"""
    ev = []; ch = [c for c in CH[si] if c['role'] in ('caption', 'hero', 'kicker') and c['text'] not in skip]
    end = CH[si][-1]['e'] + 0.35
    for i, c in enumerate(ch):
        off = ch[i + 1]['s'] - 0.02 if i + 1 < len(ch) else end
        tier = 2 if c['tier'] == 1 else c['tier']
        font, size = R.STY[tier]
        ev.append((R.text_img(disp(c) if tier == 2 else c['text'], font, size), 540, y, c['s'], off))
    return ev

def draw_caps(frame, t, ev, cut_at=None):
    for img, x, y, on, off in ev:
        if cut_at is not None: off = min(off, cut_at)
        R.anim_word(frame, img, x, y, t, on, off, 'soft')

# ---------------- sources ----------------
FACE_X, FACE_Y = R.FX, R.FY
def preload(a, b, sp=1.0):
    r = R.Reader(a, b, sp); fr = []
    while True:
        f = r.read()
        if f is None or (fr and f is fr[-1][1]): break
        fr.append((None, f));
        if len(fr) >= int((b - a) / sp * FPS): break
    r.close()
    return [R.grade(cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA)) for _, f in fr]
print('preloading b-roll...', flush=True)
FK = preload(9.10, 11.6)        # forklift close
FK2 = preload(9.07, 10.2)       # forklift for body tail (same as source cut)
EDGE = preload(23.10, 24.55)    # green MDF edge macro
STACK = preload(24.65, 26.6)    # MDF stack macro
LABEL = preload(22.56, 23.05)   # product label shot
def bclip(clip, t, t0, sp=1.0):
    i = int((t - t0) * FPS * sp); return clip[max(0, min(len(clip) - 1, i))]

def zoom_img(img, z, cy=0.5, cx=0.5):
    if z <= 1.001: return img
    h, w = img.shape[:2]; nw, nh = int(w / z), int(h / z)
    x0 = int(min(max(cx * w - nw / 2, 0), w - nw)); y0 = int(min(max(cy * h - nh / 2, 0), h - nh))
    return cv2.resize(img[y0:y0 + nh, x0:x0 + nw], (w, h), interpolation=cv2.INTER_LINEAR)

def a_frame(img, sf, z, t, target=(540, 0.42 * H), shake=(0, 0), anchor=None):
    s = 0.5 * z
    ax, ay = anchor if anchor is not None else (FACE_X[sf], FACE_Y[sf])
    ox = ax - target[0] / s; oy = ay - target[1] / s
    ox = min(max(ox + shake[0] / s, 0), SW - W / s); oy = min(max(oy + shake[1] / s, 0), SH - H / s)
    fr = R.grade(R.warp(img, s, ox, oy))
    m = R.matte_for(sf, s, ox, oy)
    return fr, m, (s, ox, oy)

def shake_at(t, events):
    dx = dy = 0.0
    for t0, amp in events:
        u = (t - t0) / 0.27
        if 0 <= u < 1:
            a = amp * (1 - u) ** 2; dx += a * math.sin(t * 97 + t0); dy += a * math.cos(t * 83 + 2 * t0)
    return dx, dy

def vblur(fr, k):
    if k < 3: return fr
    ker = np.full((int(k), 1), 1.0 / int(k), np.float32); return cv2.filter2D(fr, -1, ker)

def zblur(fr, amt):
    if amt < 0.005: return fr
    acc = fr.astype(np.float32)
    for k in range(1, 5):
        sc = 1 + amt * k / 4; big = cv2.resize(fr, None, fx=sc, fy=sc)
        y0 = (big.shape[0] - H) // 2; x0 = (big.shape[1] - W) // 2; acc += big[y0:y0 + H, x0:x0 + W]
    return (acc / 5).astype(np.uint8)

def whip_k(t, c, l=0.12, r=0.12, km=70):
    if c - l <= t < c: return km * ease_in((t - (c - l)) / l)
    if c <= t < c + r: return km * (1 - ease_out((t - c) / r))
    return 0

def spotlight(fr, m, u):
    if m is None or u <= 0: return fr
    f = fr.astype(np.float32); bg = cv2.GaussianBlur(fr, (0, 0), 6).astype(np.float32) * (1 - 0.45 * u)
    return (f * m + (bg * u + f * (1 - u)) * (1 - m)).astype(np.uint8)

def flash(fr, a): return cv2.addWeighted(fr, 1 - a, np.full_like(fr, 255), a, 0) if a > 0 else fr

def behind(frame, video, m):
    return frame if m is None else (frame.astype(np.float32) * (1 - m) + video.astype(np.float32) * m).astype(np.uint8)

TAG_SPEC, TAG_BRAND = R.build_tag('THÔNG SỐ'), R.build_tag('THƯƠNG HIỆU')

# ---------------- shared body 4.98 – 10.1 ----------------
BODY_CAPS = caption_list(1, 1160) + caption_list(2, 1160)
def body(t, img, sf):
    if t < 5.58: z = lerp(2.9, 1.9, ease_out((t - TW['s1']) / 0.6))
    elif t < TW['yeuto']: z = 1.9
    elif t < TW['quantam']: z = lerp(1.9, 2.05, ease((t - TW['yeuto']) / (TW['quantam'] - TW['yeuto'])))
    else: z = 2.45
    if t >= CUT_FK:
        fr = zoom_img(bclip(FK2, t, CUT_FK), 1 + 0.03 * (t - CUT_FK)); tf = None
    else:
        fr, _, tf = a_frame(img, sf, z, t)
    fr = zblur(fr, 0.09 * (1 - c01((t - TW['s1']) / 0.17)))
    fr = vblur(fr, whip_k(t, CUT_FK))
    # tem card on hand
    t0, t1 = TW['brand'], TW['brand'] + 1.7
    if t0 - 2 / FPS <= t < t1 and tf:
        s, ox, oy = tf; hx = (R.HAND_SRC[0] - ox) * s; hy = (R.HAND_SRC[1] - oy) * s
        ts = t - (t0 - 2 / FPS); sc = (0.55 + 0.45 * back_out(ts / (6 / FPS), 2.0)) * 0.85
        al = c01(ts / (2 / FPS)) * c01((t1 - t) / (4 / FPS))
        card = R.LABEL_CARD.copy() if ts >= 1.5 / FPS else np.dstack([np.full(R.LABEL_CARD.shape[:2] + (3,), 255, np.uint8), R.LABEL_CARD[..., 3]])
        card = cv2.resize(card, None, fx=sc, fy=sc)
        R.paste(fr, card, hx + 70, hy - card.shape[0] / 2 - 30, al)
    draw_caps(fr, t, BODY_CAPS)
    return fr
BODY_SFX = [('whoosh', TW['s1'], -2), ('card_flip', TW['brand'], 0), ('tick', TW['quantam'], 0), ('whoosh', CUT_FK, -2)]

# ---------------- V1 ĐỐI ĐẦU: split-screen open + face in circle ----------------
def v1(t, img, sf):
    if t >= TW['s1']: return body(t, img, sf)
    T_EXP = TW['q']; T_EXP1 = T_EXP + 0.33
    if t < T_EXP1:
        # split background
        bg = np.empty((H, W, 3), np.uint8)
        top = zoom_img(bclip(FK, t, 0, 0.8), 1.0 + 0.04 * t, cy=0.42); bot = zoom_img(bclip(STACK, t, 0, 0.6), 1.0 + 0.04 * t)
        bg[:H // 2] = top[H // 2 - 480 - 140:H // 2 + 480 - 140][:H // 2]; bg[H // 2:] = bot[480:480 + H // 2]
        # highlight the half being said
        if TW['h1'] <= t < TW['h2']: bg[:H // 2] = (bg[:H // 2] * 0.45).astype(np.uint8)
        if TW['h2'] <= t < T_EXP: bg[H // 2:] = (bg[H // 2:] * 0.45).astype(np.uint8)
        cv2.rectangle(bg, (0, H // 2 - 3), (W, H // 2 + 3), (255, 255, 255), -1)
        R.anim_word(bg, TAG_BRAND, 540, 230, t, 0.05, TW['h2'], 'soft')
        R.anim_word(bg, TAG_SPEC, 540, 1690 - 220, t, 0.15, TW['h1'], 'soft')
        if TW['h1'] <= t < T_EXP1: R.anim_word(bg, R.text_img('THÔNG SỐ', 'black', 120), 540, 1430, t, TW['h1'], T_EXP1, 'slam')
        if TW['h2'] <= t < T_EXP1: R.anim_word(bg, R.text_img('THƯƠNG HIỆU', 'black', 110), 540, 420, t, TW['h2'], T_EXP1, 'slam')
        # face circle → expands to full frame
        u = ease_in(c01((t - T_EXP) / (T_EXP1 - T_EXP)))
        cy = lerp(960, 0.42 * H, u); r = lerp(235, 1350, u)
        face, m, _ = a_frame(img, sf, lerp(2.7, 2.2, u), t, target=(540, cy))
        mask = np.zeros((H, W), np.float32); cv2.circle(mask, (540, int(cy)), int(r), 1.0, -1, cv2.LINE_AA)
        mask = cv2.GaussianBlur(mask, (0, 0), 1.2)[..., None]
        fr = (bg * (1 - mask) + face * mask).astype(np.uint8)
        if u < 0.98: cv2.circle(fr, (540, int(cy)), int(r), (255, 255, 255), 8, cv2.LINE_AA)
        if t < T_EXP: draw_caps(fr, t, CAP1_V1, cut_at=TW['h1'])
        return fr
    # question with spotlight + push
    z = lerp(2.2, 2.55, ease_in((t - T_EXP1) / (GAP0 - T_EXP1))) if t < GAP0 else 2.0
    fr, m, _ = a_frame(img, sf, z, t, shake=shake_at(t, [(GAP0, 14)]))
    video = spotlight(fr, m, c01((t - T_EXP1) / 0.2) if t < GAP0 else 0)
    fr = video.copy()
    if t >= GAP0:   # hero stack slam in the silence
        R.anim_word(fr, R.text_img('thông số', 'black', 150), 540, 330, t, GAP0, TW['s1'], 'slam')
        R.anim_word(fr, R.text_img('hay', 'light', 48), 540, 445, t, GAP0 + 0.1, TW['s1'], 'soft')
        R.anim_word(fr, R.text_img('thương hiệu', 'black', 150), 540, 560, t, GAP0 + 0.2, TW['s1'], 'slam')
        fr = behind(fr, video, m)
    draw_caps(fr, t, CAP_Q, cut_at=GAP0 + 0.3)
    return vblur(fr, whip_k(t, TW['s1'], 0.12, 0.0))
CAP1_V1 = [e for e in caption_list(0, 1290, skip=('thông số', 'thương hiệu', 'hay là')) if e[3] < TW['q']]
CAP_Q = [e for e in caption_list(0, 1160) if e[3] >= TW['q'] - 0.01]
SFX_V1 = [('impact_soft', 0.02, -8), ('tick', TW['h1'], 0), ('tick', TW['h2'], 0), ('swell', TW['q'], 0), ('sub_drop', TW['q'] + 0.05, 0),
          ('riser', GAP0, -2), ('impact_soft', GAP0, 0), ('wood_knock', GAP0 + 0.2, 0)]

# ---------------- V2 ĐÓNG BĂNG: freeze + cutout pattern interrupt ----------------
FREEZE = {}
def v2(t, img, sf):
    if t >= TW['s1']: return body(t, img, sf)
    F0, F1 = TW['h1'], TW['q']
    if t < F0:
        fr, m, _ = a_frame(img, sf, lerp(2.0, 2.15, ease(t / F0)), t)
        draw_caps(fr, t, CAP_V2a, cut_at=F0); return fr
    if t < F1:
        if 'f' not in FREEZE:
            fr, m, _ = a_frame(img, sf, 2.15, t)
            g = cv2.cvtColor(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
            g = (cv2.GaussianBlur(g, (0, 0), 3) * 0.42).astype(np.uint8)
            mb = (m[..., 0] > 0.5).astype(np.uint8)
            ring = cv2.dilate(mb, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))).astype(np.float32)
            ring = cv2.GaussianBlur(ring, (0, 0), 1.2)[..., None]
            FREEZE['f'] = (fr, m, g, ring)
        fr0, m, g, ring = FREEZE['f']
        u = ease_out(c01((t - F0) / 0.2))
        sub = (fr0.astype(np.float32) * m + 255 * (ring - m).clip(0, 1)).astype(np.float32)
        alpha = ring
        sc = 1 + 0.05 * u; M = cv2.getRotationMatrix2D((540, 0.42 * H), 0, sc)
        sub = cv2.warpAffine(sub, M, (W, H)); alpha = cv2.warpAffine(alpha, M, (W, H))[..., None]
        bgm = (g.astype(np.float32) * u + fr0.astype(np.float32) * (1 - u))
        fr = (bgm * (1 - alpha) + sub * alpha).astype(np.uint8)
        R.anim_word(fr, TAG_SPEC, 540, 300, t, F0 + 0.03, F1, 'slam')
        R.anim_word(fr, R.text_img('hay là', 'light', 52), 540, 410, t, TW['h1'] + 0.4, F1, 'soft')
        R.anim_word(fr, TAG_BRAND, 540, 520, t, TW['h2'], F1, 'slam')
        return flash(fr, 0.7 if t < F0 + 2 / FPS else 0)
    if t < GAP0:
        z = lerp(2.5, 2.7, (t - F1) / (GAP0 - F1))
        fr, m, _ = a_frame(img, sf, z, t)
        fr = spotlight(fr, m, 1.0); fr = vblur(fr, whip_k(t, F1, 0.0, 0.13, 70))
        draw_caps(fr, t, CAP_Q, cut_at=GAP0); return fr
    # mini drop: 3 b-roll flashes on 0.3 s grid
    k = int((t - GAP0) / 0.3)
    clip, t0 = [(FK, GAP0), (EDGE, GAP0 + 0.3), (STACK, GAP0 + 0.6)][min(k, 2)]
    fr = zoom_img(bclip(clip, t, t0, 1.5), 1.05 + 0.15 * ((t - t0) / 0.3), cy=0.45)
    fr = vblur(fr, max(whip_k(t, GAP0 + 0.3, 0.06, 0.06, 50), whip_k(t, GAP0 + 0.6, 0.06, 0.06, 50), whip_k(t, TW['s1'], 0.1, 0, 60)))
    fr = (fr * 0.75).astype(np.uint8)
    word = ['THƯƠNG HIỆU', 'THÔNG SỐ', '?'][min(k, 2)]
    R.anim_word(fr, R.text_img(word, 'black', 130 if k < 2 else 260), 540, 900, t, t0, t0 + 0.3 if k < 2 else TW['s1'], 'slam')
    return fr
CAP_V2a = [e for e in caption_list(0, 1160) if e[3] < TW['h1']]
SFX_V2 = [('tick', TW['mdf'], -2), ('shutter', TW['h1'], 2), ('impact_soft', TW['h1'], -8), ('card_flip', TW['h1'] + 0.03, 0),
          ('card_flip', TW['h2'], 0), ('whoosh', TW['q'], 0), ('riser', GAP0, -2),
          ('impact_soft', GAP0, -2), ('whoosh', GAP0 + 0.3, -2), ('wood_knock', GAP0 + 0.3, 0), ('whoosh', GAP0 + 0.6, -2), ('wood_knock', GAP0 + 0.6, 0)]

# ---------------- V3 NĂNG LƯỢNG CAO: snap zooms + slams + glitch ----------------
def v3(t, img, sf):
    if t >= TW['s1']: return body(t, img, sf)
    shakes = [(0.0, 16), (TW['mdf'], 12), (TW['h1'], 14), (TW['h2'], 14), (GAP0, 16)]
    sh = shake_at(t, shakes)
    if t < GAP0:
        if t < TW['mdf']: z = 2.8
        elif t < TW['h1']: z = lerp(2.8, 2.0, ease_out((t - TW['mdf']) / 0.2))
        elif t < TW['h2']: z = lerp(2.0, 2.65, ease_out((t - TW['h1']) / (5 / FPS)))
        elif t < TW['q']: z = lerp(2.65, 2.05, ease_out((t - TW['h2']) / (5 / FPS)))
        else: z = lerp(2.05, 2.75, ease_in((t - TW['q']) / (GAP0 - TW['q'])))
        fr, m, _ = a_frame(img, sf, z, t, shake=sh)
        amt = 0.0
        for c in (TW['h1'], TW['h2']):
            if c <= t < c + 0.13: amt = 0.09 * (1 - (t - c) / 0.13)
        fr = zblur(fr, amt); video = fr.copy()
        if TW['mdf'] <= t < TW['h1']: R.anim_word(fr, R.text_img('MDF', 'black', 260), 540, 480, t, TW['mdf'], TW['h1'], 'slam')
        if TW['h1'] <= t < TW['h2']: R.anim_word(fr, R.text_img('THÔNG SỐ', 'black', 150), 540, 470, t, TW['h1'], TW['h2'], 'slam')
        if TW['h2'] <= t < TW['q']: R.anim_word(fr, R.text_img('THƯƠNG HIỆU', 'black', 132), 540, 470, t, TW['h2'], TW['q'], 'slam')
        fr = behind(fr, video, m)
        # glitch on the question
        if TW['q'] <= t < TW['q'] + 3 / FPS or 3.55 <= t < 3.55 + 2 / FPS:
            b, g, r = cv2.split(fr); b = np.roll(b, 14, 1); r = np.roll(r, -14, 1); fr = cv2.merge([b, g, r])
            for _ in range(5):
                y = np.random.randint(0, H - 60); fr[y:y + 40] = np.roll(fr[y:y + 40], np.random.randint(-60, 60), 1)
        if t >= TW['q']:
            draw_caps(fr, t, [e for e in CAP_Q if e[3] < TW['qt'] - 0.01], cut_at=TW['qt'])
            if t >= TW['qt'] - 2 / FPS:   # keyword on red box
                R.anim_word(fr, Q_BOX, 540, 1160, t, TW['qt'], GAP0 + 0.1, 'slam')
        else:
            draw_caps(fr, t, [e for e in caption_list(0, 1160, skip=('MDF', 'thông số', 'thương hiệu')) if e[3] < TW['q']])
        return fr
    # VS slam in the silence
    u = (t - GAP0) / (TW['s1'] - GAP0)
    bg = np.empty((H, W, 3), np.uint8)
    top = zoom_img(bclip(FK, t, GAP0, 1.4), 1.0 + 0.08 * u, cy=0.4); bot = zoom_img(bclip(EDGE, t, GAP0, 1.0), 1.05 + 0.08 * u)
    bg[:H // 2] = top[320:320 + H // 2]; bg[H // 2:] = bot[480:480 + H // 2]
    cv2.rectangle(bg, (0, H // 2 - 3), (W, H // 2 + 3), (255, 255, 255), -1)
    R.anim_word(bg, TAG_BRAND, 540, 300, t, GAP0 + 0.05, TW['s1'], 'slam')
    R.anim_word(bg, TAG_SPEC, 540, 1420, t, GAP0 + 0.15, TW['s1'], 'slam')
    R.anim_word(bg, R.VS_BADGE, 540 + sh[0], 960 + sh[1], t, GAP0, TW['s1'], 'slam')
    return flash(vblur(bg, whip_k(t, GAP0, 0, 0.1, 60) + whip_k(t, TW['s1'], 0.1, 0, 60)), 0.75 if t < GAP0 + 2 / FPS else 0)
def _qbox():
    f = ImageFont.truetype(R.FONTS['black'], 70); txt = 'QUAN TRỌNG?'; bb = f.getbbox(txt)
    w, h = bb[2] - bb[0] + 60, 110; im = Image.new('RGBA', (w, h), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    d.rounded_rectangle([0, 0, w - 1, h - 1], 18, fill=(215, 25, 32, 255))
    d.text((30 - bb[0], (h - (bb[3] - bb[1])) / 2 - bb[1]), txt, font=f, fill=(255, 255, 255, 255))
    return np.array(im)[..., [2, 1, 0, 3]]
Q_BOX = _qbox()
SFX_V3 = [('wood_knock', 0.0, 2), ('impact_soft', TW['mdf'], -4), ('whoosh', TW['h1'], 0), ('wood_knock', TW['h1'], 0),
          ('whoosh', TW['h2'], 0), ('wood_knock', TW['h2'], 0), ('glitch', TW['q'], 4), ('glitch', 3.55, 2), ('ding', TW['qt'], 0),
          ('riser', GAP0, 0), ('impact_soft', GAP0, 0), ('shutter', GAP0, 0)]

# ---------------- V4 ĐIỆN ẢNH: macro open → zoom-out reveal from the board in her hand ----------------
def v4(t, img, sf):
    if t >= TW['s1']: return body(t, img, sf)
    T_REV = TW['q']; T_REV1 = T_REV + 0.6
    bars = int(250 * (1 - ease(c01((t - T_REV) / 0.6))))
    if t < T_REV:
        if t < 1.5: fr = zoom_img(bclip(EDGE, t, 0, 0.6), 1.08 + 0.05 * t)
        else: fr = zoom_img(bclip(STACK, t, 1.5, 0.5), 1.10 + 0.05 * (t - 1.5))
        fr = (fr * 0.82).astype(np.uint8)
        fr = vblur(fr, whip_k(t, 1.5, 0.1, 0.1, 40))
        R.anim_word(fr, MDF_SPACED, 540, 700, t, TW['mdf'], T_REV, 'phrase')
        R.anim_word(fr, R.text_img('thông số', 'black', 150), 540, 900, t, TW['h1'], T_REV, 'soft')
        R.anim_word(fr, R.text_img('hay là', 'light', 48), 540, 1010, t, TW['h1'] + 0.4, T_REV, 'soft')
        R.anim_word(fr, R.text_img('thương hiệu', 'black', 150), 540, 1120, t, TW['h2'], T_REV, 'soft')
        draw_caps(fr, t, [e for e in caption_list(0, 1350, skip=('MDF', 'thông số', 'thương hiệu', 'hay là')) if e[3] < T_REV], cut_at=TW['h1'])
    else:
        u = ease_out(c01((t - T_REV) / 0.6))
        z = lerp(4.0, 2.1, u) if t < T_REV1 else lerp(2.1, 2.35, (t - T_REV1) / (GAP0 + 0.88 - T_REV1))
        hand = (R.HAND_SRC[0], R.HAND_SRC[1] + 140); face = (FACE_X[sf], FACE_Y[sf])
        anc = (lerp(hand[0], face[0], u), lerp(hand[1], face[1], u))
        fr, m, _ = a_frame(img, sf, z, t, anchor=anc, target=(540, lerp(0.5 * H, 0.42 * H, u)))
        fr = zblur(fr, 0.08 * (1 - u))
        draw_caps(fr, t, CAP_Q, cut_at=GAP0 + 0.3)
        if t >= GAP0 + 0.1:
            R.anim_word(fr, R.text_img('MDF chuẩn chọn thế nào?', 'light', 52), 540, 1250, t, GAP0 + 0.4, TW['s1'], 'phrase')
    if bars > 0: fr[:bars] = 0; fr[H - bars:] = 0
    return fr
def _spaced(txt, size=64, track=38):
    f = ImageFont.truetype(R.FONTS['light'], size); w = int(sum(f.getlength(c) for c in txt) + track * (len(txt) - 1)) + 40
    im = Image.new('RGBA', (w, int(size * 1.8)), (0, 0, 0, 0)); d = ImageDraw.Draw(im); x = 20
    for c in txt: d.text((x, int(size * 1.2)), c, font=f, fill=(255, 255, 255, 255), anchor='ls'); x += f.getlength(c) + track
    return np.array(im)[..., [2, 1, 0, 3]]
MDF_SPACED = _spaced('M D F', 70, 30)
SFX_V4 = [('swell', 0.6, -6), ('tick', TW['h1'], -4), ('tick', TW['h2'], -4), ('swell', TW['q'], 0), ('sub_drop', TW['q'] + 0.02, 2),
          ('impact_soft', TW['q'] + 0.02, -10)]

# ---------------- V5 THĂM DÒ: TikTok-style poll ----------------
def _poll():
    w, h = 860, 380; im = Image.new('RGBA', (w + 40, h + 40), (0, 0, 0, 0))
    sh = Image.new('L', im.size, 0); ImageDraw.Draw(sh).rounded_rectangle([20, 26, w + 20, h + 26], 36, fill=110)
    sh = Image.fromarray(cv2.GaussianBlur(np.array(sh), (0, 0), 12)); base = Image.new('RGBA', im.size, (0, 0, 0, 0)); base.putalpha(sh)
    d = ImageDraw.Draw(base); d.rounded_rectangle([20, 20, w + 20, h + 20], 36, fill=(255, 255, 255, 245))
    f = ImageFont.truetype(R.FONTS['bold'], 40); d.text((60, 52), 'Chọn MDF, bạn dựa vào?', font=f, fill=(30, 30, 30, 255))
    return np.array(base)[..., [2, 1, 0, 3]]
def _opt(txt, active=False):
    w, h = 780, 104; im = Image.new('RGBA', (w, h), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    d.rounded_rectangle([0, 0, w - 1, h - 1], 52, fill=(215, 25, 32, 255) if active else (238, 238, 240, 255))
    d.ellipse([26, 30, 70, 74], outline=(255, 255, 255, 255) if active else (160, 160, 165, 255), width=4)
    f = ImageFont.truetype(R.FONTS['xbold'], 44); d.text((100, 52), txt, font=f, fill=(255, 255, 255, 255) if active else (30, 30, 30, 255), anchor='lm')
    return np.array(im)[..., [2, 1, 0, 3]]
POLL = _poll(); O1, O1A, O2, O2A = _opt('THÔNG SỐ'), _opt('THÔNG SỐ', True), _opt('THƯƠNG HIỆU'), _opt('THƯƠNG HIỆU', True)
def v5(t, img, sf):
    if t >= TW['s1']: return body(t, img, sf)
    z = lerp(2.0, 2.2, ease(t / GAP0)) if t < GAP0 else 2.2
    fr, m, _ = a_frame(img, sf, z, t, target=(540, 0.36 * H))
    if t >= TW['q']: fr = spotlight(fr, m, c01((t - TW['q']) / 0.2) * 0.8)
    cy = 1330; out = c01((t - GAP0) / 0.25)
    if t >= TW['mdf'] + 0.3:
        u = back_out(c01((t - TW['mdf'] - 0.3) / 0.25), 1.6); y = cy + 60 * (1 - u) + 700 * ease_in(out)
        R.paste(fr, POLL, 540, y, c01((t - TW['mdf'] - 0.3) / 0.12) * (1 - out))
        blink = int((t - TW['q']) * 6) % 2 if t >= TW['q'] else -1
        if t >= TW['h1'] - 2 / FPS: R.paste(fr, O1A if (TW['h1'] <= t < TW['h1'] + 0.5 or blink == 0) else O1, 540, y + 10, c01((t - TW['h1'] + 2 / FPS) / 0.12) * (1 - out))
        if t >= TW['h2'] - 2 / FPS: R.paste(fr, O2A if (TW['h2'] <= t < TW['h2'] + 0.5 or blink == 1) else O2, 540, y + 128, c01((t - TW['h2'] + 2 / FPS) / 0.12) * (1 - out))
    draw_caps(fr, t, [e[:2] + (e[2], 1010) + e[4:] if False else (e[0], e[1], 1010, e[3], e[4]) for e in caption_list(0, 1010, skip=('thông số', 'thương hiệu', 'hay là'))], cut_at=GAP0 + 0.2)
    if t >= GAP0:
        R.anim_word(fr, R.text_img('Bình luận lựa chọn của bạn', 'light', 50), 540, 1330, t, GAP0 + 0.15, TW['s1'], 'phrase')
    return vblur(fr, whip_k(t, TW['s1'], 0.12, 0, 60))
SFX_V5 = [('card_flip', TW['mdf'] + 0.3, 0), ('tick', TW['h1'], 0), ('tick', TW['h2'], 0), ('swell', TW['q'], -2),
          ('tick', TW['q'] + 1 / 6, -8), ('tick', TW['q'] + 2 / 6, -8), ('tick', TW['q'] + 3 / 6, -8), ('swipe', GAP0, 2)]

VARIANTS = {'V1': ('01_doi_dau', v1, SFX_V1), 'V2': ('02_dong_bang', v2, SFX_V2), 'V3': ('03_nang_luong_cao', v3, SFX_V3),
            'V4': ('04_dien_anh', v4, SFX_V4), 'V5': ('05_tham_do', v5, SFX_V5)}


# ---------------- V6 TAN BIẾN: hook (V3 energy) → presenter disintegrates into dust → empty showroom → B-roll VO ----------------
VAN_F = 123                      # frozen source frame (= plate reference frame)
VAN_Z = 1.55
T_ZO, T_DUST, T_DUST_END = GAP0, GAP0 + 0.17, GAP0 + 1.05
PLATE4K = cv2.imread(f'{ROOT}/work/plate_lama.png')
VAN = {}
def _van_setup(img4k):
    anchor = (FACE_X[VAN_F], FACE_Y[VAN_F])
    fr, m, tf = a_frame(img4k, VAN_F, VAN_Z, 0, anchor=anchor)
    s, ox, oy = tf
    plate = R.grade(R.warp(PLATE4K, s, ox, oy))
    a = m[..., 0]
    ys, xs = np.where(a > 0.5); bx0, bx1, by0, by1 = xs.min(), xs.max(), ys.min(), ys.max()
    rng = np.random.default_rng(3)
    noise = cv2.resize(rng.random((48, 27)).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
    nx = (np.arange(W)[None, :] - bx0) / max(1, bx1 - bx0); ny = (np.arange(H)[:, None] - by0) / max(1, by1 - by0)
    tau = T_DUST + 0.62 * np.clip(0.55 * nx + 0.45 * (1 - ny), 0, 1) + 0.16 * noise     # sweep: lower-left → upper-right
    # particles from the person pixels (grid step 3)
    gy, gx = np.mgrid[0:H:3, 0:W:3]; sel = a[gy, gx] > 0.5
    py, px = gy[sel].astype(np.float32), gx[sel].astype(np.float32)
    col = fr[gy[sel], gx[sel]].astype(np.float32)
    n = len(px)
    P = dict(x=px, y=py, c=np.clip(col * 1.1 + 45, 0, 255), t0=tau[gy[sel], gx[sel]],
             vx=rng.uniform(120, 520, n).astype(np.float32), vy=rng.uniform(-340, -60, n).astype(np.float32),
             life=rng.uniform(0.55, 1.1, n).astype(np.float32), ph=rng.uniform(0, 6.28, n).astype(np.float32))
    VAN.update(fr=fr, a=a, plate=plate, tau=tau, P=P, tf=tf)

def _van_frame(t):
    fr, a, plate, tau, P = VAN['fr'], VAN['a'], VAN['plate'], VAN['tau'], VAN['P']
    vis = np.clip((tau - t) / 0.05, 0, 1) * a                      # person pixels not yet dissolved
    edge = np.clip(1 - np.abs(tau - t) / 0.03, 0, 1) * a          # thin warm rim at the dissolve front
    out = plate.astype(np.float32) * (1 - vis[..., None]) + fr.astype(np.float32) * vis[..., None]
    out += (edge[..., None] * np.array([25, 55, 85], np.float32))
    age = t - P['t0']; on = (age > 0) & (age < P['life'])
    if on.any():
        ag = age[on]; k = ag / P['life'][on]
        x = P['x'][on] + P['vx'][on] * ag + 40 * np.sin(P['ph'][on] + ag * 7) * ag
        y = P['y'][on] + P['vy'][on] * ag - 120 * ag * ag
        al = ((1 - k) ** 1.1)[:, None]
        xi, yi = x.astype(np.int32), y.astype(np.int32)
        ok = (xi >= 0) & (xi < W - 1) & (yi >= 0) & (yi < H - 1)
        xi, yi, al, c = xi[ok], yi[ok], al[ok], P['c'][on][ok]
        for dx, dy, w in ((0, 0, 1.0), (1, 0, 0.9), (0, 1, 0.9), (1, 1, 0.8)):
            out[yi + dy, xi + dx] = out[yi + dy, xi + dx] * (1 - al * w) + c * (al * w)
    return np.clip(out, 0, 255).astype(np.uint8)

def v6(t, img, sf):
    if t < GAP0: return v3(t, img, sf)
    if not VAN:
        VAN['img4k'] = img.copy() if sf == VAN_F else None
    if 'fr' not in VAN:
        _van_setup(VAN['img4k'] if VAN.get('img4k') is not None else img)
    if t < T_DUST:      # snap zoom-out from the hook framing to the wide vanish framing (frozen frame)
        u = ease_out((t - T_ZO) / (T_DUST - T_ZO)); z = lerp(2.75, VAN_Z, u)
        fr, _, _ = a_frame(VAN['img4k'], VAN_F, z, t, anchor=(FACE_X[VAN_F], FACE_Y[VAN_F]))
        return zblur(fr, 0.09 * (1 - u))
    if t < 5.62:
        fr = _van_frame(t)
        zp = 1 + 0.06 * ease((t - T_DUST_END) / 0.6) if t > T_DUST_END else 1.0
        fr = zoom_img(fr, zp, cy=0.45)
        draw_caps(fr, t, [e for e in BODY_CAPS if e[3] < 5.62], cut_at=5.62)
        return vblur(fr, whip_k(t, 5.62, 0.12, 0, 60))
    # VO over product B-roll
    if t < TW['yeuto']: fr = zoom_img(bclip(LABEL, t, 5.62, 0.42), 1.05 + 0.05 * (t - 5.62))
    elif t < TW['quantam']: fr = zoom_img(bclip(EDGE, t, TW['yeuto'], 1.0), 1.05 + 0.05 * (t - TW['yeuto']))
    elif t < CUT_FK: fr = zoom_img(bclip(STACK, t, TW['quantam'], 0.9), 1.05 + 0.05 * (t - TW['quantam']))
    else: fr = zoom_img(bclip(FK2, t, CUT_FK), 1 + 0.03 * (t - CUT_FK))
    k = max(whip_k(t, 5.62, 0, 0.12, 60), whip_k(t, TW['yeuto'], 0.1, 0.1, 50), whip_k(t, TW['quantam'], 0.1, 0.1, 50), whip_k(t, CUT_FK, 0.1, 0.1, 60))
    fr = vblur(fr, k)
    draw_caps(fr, t, BODY_CAPS)
    return fr
SFX_V6 = [s for s in SFX_V3 if s[1] < GAP0 - 0.01] + [('riser', GAP0, -2), ('whoosh', GAP0, 0), ('dust', T_DUST, 4), ('sub_drop', T_DUST, -4),
          ('whoosh', 5.62, -2), ('card_flip', 5.62, -4), ('whoosh', TW['yeuto'], -6), ('tick', TW['quantam'], 0), ('whoosh', CUT_FK, -4)]
VARIANTS['V6'] = ('06_tan_bien', v6, SFX_V6)

# ---------------- audio ----------------
def build_audio(sfx_list, path, kick_gap=False, no_body=False):
    import soundfile as sf_, audio as A
    SR = L.SR; N = int(DUR * SR) + SR
    raw = f'{ROOT}/work/hook_vo.wav'
    subprocess.run(['ffmpeg', '-loglevel', 'error', '-y', '-ss', '0', '-t', str(DUR), '-i', f'{ROOT}/src/1109.mp4', '-ac', '1', '-ar', str(SR),
                    '-af', 'highpass=f=80,dynaudnorm=f=250:g=15:p=0.9:m=8,acompressor=threshold=-20dB:ratio=3:attack=8:release=160:makeup=2,loudnorm=I=-16:TP=-2', raw], check=True)
    vo, _ = sf_.read(raw); v = np.zeros(N); v[:len(vo)] = vo[:N]
    m = np.zeros(N)
    A.place(m, A.pad(A.CHORDS[0][0], 4.4, 0.6, cutoff=700), 0.0)
    A.place(m, A.pad(A.CHORDS[1][0], 3.0, 0.7), GAP0); A.place(m, A.bass_sus(43.65, 2.6, 0.8), GAP0)
    A.place(m, A.pad(A.CHORDS[2][0], 3.2, 0.6), 7.0); A.place(m, A.bass_sus(65.41, 3.1, 0.6), 7.0)
    for i in range(int(DUR / 0.3)): A.place(m, A.shaker(0.5), GAP0 + i * 0.3) if GAP0 + i * 0.3 < DUR else None
    if kick_gap:
        for k in range(3): A.place(m, A.kick_soft(0.9), GAP0 + k * 0.3)
    m = A.verb(m, 0.28)
    s = np.zeros(N)
    for name, th, extra in sfx_list + ([] if no_body else BODY_SFX): L.place_hit(s, L.SYNTH, name, th, extra)
    s = L.verb(s, 0.22)
    vr = np.sqrt(np.mean(v[np.abs(v) > 1e-4] ** 2)); mr = np.sqrt(np.mean(m ** 2)) + 1e-9
    m *= vr * A.db(-13) / mr
    mix = (v + m * A.duck_gain(v) + s * (vr / 0.12))[:int(DUR * SR)]
    fo = int(0.25 * SR); mix[-fo:] *= np.linspace(1, 0, fo)
    sf_.write(f'{ROOT}/work/hook_pre.wav', np.stack([mix, mix], 1) * 0.5, SR, subtype='FLOAT')
    subprocess.run(['ffmpeg', '-loglevel', 'error', '-y', '-i', f'{ROOT}/work/hook_pre.wav', '-af', 'loudnorm=I=-14:TP=-1', path], check=True)

def render_variant(key):
    name, fn, sfx = VARIANTS[key]
    FREEZE.clear(); R._hoff.clear(); VAN.clear()
    vid = f'{ROOT}/work/{name}_v.mp4'; wav = f'{ROOT}/work/{name}.wav'
    enc = subprocess.Popen(['ffmpeg', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                            '-c:v', 'libx264', '-preset', 'medium', '-crf', '17', '-pix_fmt', 'yuv420p', vid], stdin=subprocess.PIPE)
    rd = R.Reader(0, DUR + 0.2, 1)
    for fi in range(NF):
        t = fi / FPS; img = rd.read()
        fr = fn(t, img, fi)
        enc.stdin.write(np.ascontiguousarray(fr).tobytes())
    rd.close(); enc.stdin.close(); enc.wait()
    build_audio(sfx, wav, kick_gap=key in ('V2', 'V3'), no_body=key == 'V6')
    subprocess.run(['ffmpeg', '-loglevel', 'error', '-y', '-i', vid, '-i', wav, '-map', '0:v', '-map', '1:a', '-c:v', 'copy', '-c:a', 'aac', '-b:a', '256k',
                    '-shortest', '-movflags', '+faststart', f'{OUT}/{name}.mp4'], check=True)
    print('done', name, flush=True)

if __name__ == '__main__':
    keys = list(VARIANTS) if sys.argv[1] == 'all' else sys.argv[1:]
    for k in keys: render_variant(k)
