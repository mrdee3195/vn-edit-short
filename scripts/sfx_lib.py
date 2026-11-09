"""SFX library for vn-cinematic-edit.
Two sources:
  1. SYNTH  – sounds generated in code (always available, royalty-free, no files).
  2. FILES  – real recordings in sfx/ + sfx/catalog.json (licensed packs or the client's own recordings).
Every entry has a 'hit' = seconds from file start to the moment that must land on the cut / word,
so the editor can align the hit point (not the file start) to the frame.
"""
import json, os
import numpy as np
from scipy.signal import lfilter, butter, fftconvolve

SR = 48000
rng = np.random.default_rng(11)

def tt(d): return np.arange(int(d * SR)) / SR
def lp(x, fc, o=2): b, a = butter(o, fc / (SR / 2), 'low'); return lfilter(b, a, x)
def hp(x, fc, o=2): b, a = butter(o, fc / (SR / 2), 'high'); return lfilter(b, a, x)
def bp(x, lo, hi): b, a = butter(2, [lo / (SR / 2), hi / (SR / 2)], 'band'); return lfilter(b, a, x)
def fade(sig, a=0.01, r=0.03):
    na, nr = max(1, int(a * SR)), max(1, int(r * SR)); sig = sig.copy()
    sig[:na] *= np.linspace(0, 1, na); sig[-nr:] *= np.linspace(1, 0, nr); return sig
def _ir(d=1.6, decay=0.45):
    t = tt(d); ir = rng.normal(0, 1, len(t)) * np.exp(-t / decay); ir = lp(ir, 5000); ir[:int(0.012 * SR)] = 0
    return ir / np.sqrt((ir ** 2).sum())
IR = _ir()
def verb(x, wet=0.25): return x * (1 - wet) + fftconvolve(x, IR)[:len(x)] * wet * 2.0

# ---------------- v2 sounds (used in test v2) ----------------
def impact_soft(d=1.8):
    t = tt(d); f = 38 + 30 * np.exp(-t * 5)
    sub = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 2.2)
    body = lp(rng.normal(0, 1, len(t)), 400) * np.exp(-t * 5) * 0.35
    return fade((sub + body) * np.minimum(1, t / 0.018), 0.0, 0.3) * 0.8

def swell(d=0.6):
    t = tt(d); u = t / d
    n = lp(rng.normal(0, 1, len(t)), 2500) + 0.3 * hp(rng.normal(0, 1, len(t)), 5000)
    return fade(n * u ** 3 * 0.35, 0.0, 0.04)

def whoosh(d=0.55):
    t = tt(d); u = t / d; env = np.sin(np.pi * u) ** 2
    n = rng.normal(0, 1, len(t)); out = np.zeros(len(t)); step = int(0.012 * SR)
    for i in range(0, len(t), step):
        fc = 500 + 2200 * np.sin(np.pi * u[i])
        seg = bp(n[max(0, i - 1500):i + step], fc * 0.55, fc * 1.6); out[i:i + step] = seg[-len(n[i:i + step]):]
    return lp(out, 4500) * env * 0.6

def riser(d=0.8):
    t = tt(d); u = t / d
    n = rng.normal(0, 1, len(t)); out = np.zeros(len(t)); step = int(0.02 * SR)
    for i in range(0, len(t), step):
        fc = 300 + 6000 * u[i] ** 2
        seg = bp(n[max(0, i - 2000):i + step], fc * 0.6, min(fc * 1.6, SR / 2 - 100)); out[i:i + step] = seg[-len(n[i:i + step]):]
    tone = np.sin(2 * np.pi * np.cumsum(160 + 500 * u ** 2) / SR) * 0.12
    return fade((out * 0.45 + tone) * u ** 2.4, 0.0, 0.03)

def tick():
    t = tt(0.25)
    s = (np.sin(2 * np.pi * 1760 * t) + 0.4 * np.sin(2 * np.pi * 2640 * t)) * np.exp(-t * 28)
    return fade(s * np.minimum(1, t / 0.004) * 0.18, 0.0, 0.05)

def swipe():
    t = tt(0.4); u = t / 0.4
    return lp(hp(rng.normal(0, 1, len(t)), 2500), 9000) * np.sin(np.pi * u) ** 3 * 0.18

# ---------------- new sounds ----------------
def wood_knock():
    """knuckle on a wood panel: modal resonances + soft contact noise (good for MDF / furniture brands)"""
    t = tt(0.5); s = np.zeros(len(t))
    for f, dec, a in ((190, 22, 1.0), (430, 30, 0.6), (870, 45, 0.35), (1520, 70, 0.2)):
        s += a * np.sin(2 * np.pi * f * t) * np.exp(-t * dec)
    s += lp(rng.normal(0, 1, len(t)), 2500) * np.exp(-t * 180) * 0.6
    return fade(s * np.minimum(1, t / 0.0015) * 0.35, 0.0, 0.05)

def sub_drop(d=1.2):
    """cinematic sub-bass drop 80 -> 28 Hz, soft attack (drop / reveal, smoother than impact)"""
    t = tt(d); f = 28 + 52 * np.exp(-t * 3.5)
    s = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 1.8) * np.minimum(1, t / 0.03)
    return fade(s * 0.8, 0.0, 0.2)

def ding():
    """soft bell for price / CTA / 'checked' moments"""
    t = tt(1.6); s = np.zeros(len(t))
    for r, a, dec in ((1.0, 1.0, 2.2), (2.76, 0.45, 3.5), (5.40, 0.25, 6.0), (8.93, 0.1, 9.0)):
        s += a * np.sin(2 * np.pi * 880 * r * t) * np.exp(-t * dec)
    return fade(s * np.minimum(1, t / 0.003) * 0.16, 0.0, 0.2)

def shutter():
    """camera shutter, pairs with the white-flash (F1) effect"""
    out = np.zeros(int(0.2 * SR))
    for d0, g in ((0.0, 1.0), (0.045, 0.7)):
        t = tt(0.03); c = bp(rng.normal(0, 1, len(t)), 1500, 7000) * np.exp(-t * 160) * g
        i = int(d0 * SR); out[i:i + len(c)] += c
    return fade(out * 0.35, 0.0, 0.02)

def glitch(d=0.28):
    """digital glitch, pairs with the RGB-split (G1) effect — use sparingly"""
    out = np.zeros(int(d * SR)); i = 0
    while i < len(out):
        n = int(rng.uniform(0.008, 0.03) * SR)
        f = rng.choice([220, 440, 880, 1760, 3520])
        seg = np.sign(np.sin(2 * np.pi * f * np.arange(n) / SR)) * rng.uniform(0.2, 0.6)
        if rng.random() < 0.4: seg = rng.normal(0, 0.4, n)
        m = min(n, len(out) - i); out[i:i + m] = seg[:m]; i += n + int(rng.uniform(0, 0.01) * SR)
    return fade(lp(out, 7000) * 0.18, 0.002, 0.02)

def card_flip():
    """short paper/card flick, for floating cards (P1) and list items"""
    t = tt(0.14); u = t / 0.14
    return fade(bp(rng.normal(0, 1, len(t)), 2000, 8000) * np.sin(np.pi * u) ** 2 * np.exp(-t * 10) * 0.3, 0.0, 0.02)


def dust(d=1.2):
    """sand / ash disintegration: airy hiss with crackle grains (pairs with the vanish/dust effect)"""
    t = tt(d); u = t / d
    hiss = lp(hp(rng.normal(0, 1, len(t)), 2500), 9000)
    grains = np.zeros(len(t)); idx = rng.integers(0, len(t), int(900 * d)); grains[idx] = rng.uniform(-1, 1, len(idx))
    grains = bp(grains, 2000, 8000) * 3
    env = np.minimum(1, t / 0.08) * (1 - u) ** 1.5
    return fade((hiss * 0.5 + grains) * env * 0.3, 0.0, 0.05)

# ---------------- registry ----------------
# hit: seconds into the sound that must land on the frame; use: when the skill picks it
SYNTH = {
    'impact_soft': dict(fn=impact_soft, hit=0.02, gain_db=-4,  use='drop, màn VS, hero chốt cuối, end card'),
    'sub_drop':    dict(fn=sub_drop,    hit=0.03, gain_db=-6,  use='drop/reveal mềm hơn impact; zoom-out reveal (Z4)'),
    'swell':       dict(fn=swell,       hit=0.60, gain_db=-10, use='hút vào trước chữ hero (đỉnh = chữ)'),
    'riser':       dict(fn=riser,       hit=0.80, gain_db=-9,  use='dẫn vào drop (đỉnh = điểm cắt)'),
    'whoosh':      dict(fn=whoosh,      hit=0.27, gain_db=-14, use='chuyển cảnh whip, snap zoom (Z5)'),
    'swipe':       dict(fn=swipe,       hit=0.20, gain_db=-10, use='thẻ liên hệ, UI trượt vào'),
    'tick':        dict(fn=tick,        hit=0.00, gain_db=-8,  use='punch-in nhẹ, mục danh sách'),
    'card_flip':   dict(fn=card_flip,   hit=0.05, gain_db=-10, use='thẻ nổi trên tay (P1), mục danh sách'),
    'ding':        dict(fn=ding,        hit=0.00, gain_db=-12, use='giá / con số (Z5), dấu tick ✓, CTA'),
    'wood_knock':  dict(fn=wood_knock,  hit=0.00, gain_db=-6,  use='ngành gỗ/nội thất: khung hình 0, rung S1, chữ đập (slam)'),
    'shutter':     dict(fn=shutter,     hit=0.00, gain_db=-10, use='đi cùng lóe sáng F1, freeze-frame (FR)'),
    'dust':        dict(fn=dust,        hit=0.05, gain_db=-8,  use='người tan thành bụi (vanish)'),
    'glitch':      dict(fn=glitch,      hit=0.00, gain_db=-16, use='đi cùng glitch G1 (câu nêu vấn đề) — tối đa 2 lần'),
}

def load_files(folder):
    """sfx/catalog.json: [{"name","file","hit","gain_db","use","category","license"}]. Files override synth names."""
    cat = os.path.join(folder, 'catalog.json')
    if not os.path.exists(cat): return {}
    import soundfile as sf
    out = {}
    for e in json.load(open(cat)):
        x, sr = sf.read(os.path.join(folder, e['file']), always_2d=True); x = x.mean(1)
        if sr != SR:
            from scipy.signal import resample_poly
            x = resample_poly(x, SR, sr)
        out[e['name']] = dict(fn=(lambda x=x: x), hit=e.get('hit', 0.0), gain_db=e.get('gain_db', -10),
                              use=e.get('use', ''), license=e.get('license', ''))
    return out

def library(folder=None):
    lib = dict(SYNTH)
    if folder: lib.update(load_files(folder))
    return lib

def place_hit(buf, lib, name, t_hit, extra_db=0.0):
    """place sound so that its hit point lands exactly on t_hit (seconds)"""
    e = lib[name]; sig = e['fn']()
    i = int(round((t_hit - e['hit']) * SR))
    if i < 0: sig = sig[-i:]; i = 0
    n = min(len(sig), len(buf) - i)
    if n > 0: buf[i:i + n] += sig[:n] * 10 ** ((e['gain_db'] + extra_db) / 20)
