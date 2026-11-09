"""Audio v2: smoother placeholder music + SFX (no clicky booms/pops, no kick under VO, reverb bus).
Usage: python audio.py OUT.wav"""
import sys, json, subprocess
import numpy as np, soundfile as sf
from scipy.signal import lfilter, butter, fftconvolve
import plan as P

SR = 48000
ROOT = '/home/user/project'
N = int(P.DURATION * SR) + 2 * SR
CAP = json.load(open(f'{ROOT}/edit/caption_plan.json'))
rng = np.random.default_rng(7)

def ch(si, text, nth=0): return [c for c in CAP[si]['chunks'] if c['text'] == text][nth]['s']
T = dict(h1=ch(0, 'thông số'), h2=ch(0, 'thương hiệu'), q=ch(0, 'đâu mới là'), card=ch(1, 'thương hiệu'),
         chuadu=ch(4, 'chưa đủ'), cahai=ch(7, 'cả hai.'))
LIST_T = [c['s'] for c in CAP[5]['chunks'] if c['role'] == 'list']
IT = {i['id']: i for i in P.ITEMS}
CUT_A2B = IT['v5']['t0'] + (IT['v5']['a_until_src'] - IT['v5']['src'])

def place(buf, sig, t, gain=1.0):
    i = int(round(t * SR))
    if i < 0: sig = sig[-i:]; i = 0
    n = min(len(sig), len(buf) - i)
    if n > 0: buf[i:i + n] += sig[:n] * gain
def place_end(buf, sig, t_end, gain=1.0): place(buf, sig, t_end - len(sig) / SR, gain)
def db(x): return 10 ** (x / 20)
def lp(x, fc, o=2): b, a = butter(o, fc / (SR / 2), 'low'); return lfilter(b, a, x)
def hp(x, fc, o=2): b, a = butter(o, fc / (SR / 2), 'high'); return lfilter(b, a, x)
def bp(x, lo, hi): b, a = butter(2, [lo / (SR / 2), hi / (SR / 2)], 'band'); return lfilter(b, a, x)
def tt(d): return np.arange(int(d * SR)) / SR
def fade(sig, a=0.01, r=0.03):
    na, nr = max(1, int(a * SR)), max(1, int(r * SR))
    sig = sig.copy(); sig[:na] *= np.linspace(0, 1, na); sig[-nr:] *= np.linspace(1, 0, nr); return sig

def reverb_ir(d=1.6, decay=0.45):
    t = tt(d); ir = rng.normal(0, 1, len(t)) * np.exp(-t / decay)
    ir = lp(ir, 5000); ir[:int(0.012 * SR)] = 0
    return ir / np.sqrt((ir ** 2).sum())
IR = reverb_ir()
def verb(x, wet=0.3): return x * (1 - wet) + fftconvolve(x, IR)[:len(x)] * wet * 2.0

# ---------------- VO ----------------
def build_vo():
    raw = f'{ROOT}/work/src48.wav'
    subprocess.run(['ffmpeg', '-loglevel', 'error', '-y', '-i', f'{ROOT}/src/1109.mp4', '-ac', '1', '-ar', str(SR), raw], check=True)
    x, _ = sf.read(raw)
    vo = np.zeros(N)
    for a, b, o in P.AUDIO:
        place(vo, fade(x[int(a * SR):int(b * SR)], 0.012, 0.012), o)
    sf.write(f'{ROOT}/work/vo_raw.wav', vo, SR)
    subprocess.run(['ffmpeg', '-loglevel', 'error', '-y', '-i', f'{ROOT}/work/vo_raw.wav', '-af',
                    'highpass=f=80,dynaudnorm=f=250:g=15:p=0.9:m=8,acompressor=threshold=-20dB:ratio=3:attack=8:release=160:makeup=2,'
                    'deesser=i=0.4,loudnorm=I=-16:TP=-2:LRA=7', '-ar', str(SR), f'{ROOT}/work/vo_proc.wav'], check=True)
    v, _ = sf.read(f'{ROOT}/work/vo_proc.wav')
    out = np.zeros(N); out[:min(N, len(v))] = v[:N]
    return out

# ---------------- music (soft, no kick under VO) ----------------
def kick_soft(g=1.0):
    t = tt(0.45); f = 42 + 60 * np.exp(-t * 22)
    s = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 7)
    return fade(lp(s, 180), 0.004, 0.05) * g          # no click transient

def clap_soft(g=1.0):
    t = tt(0.35); n = bp(rng.normal(0, 1, len(t)), 1200, 4000)
    return fade(n * np.exp(-t * 14) * 0.25, 0.004, 0.05) * g

def shaker(g=1.0):
    t = tt(0.12); n = hp(rng.normal(0, 1, len(t)), 6000)
    env = np.minimum(1, t / 0.02) * np.exp(-t * 30)
    return n * env * 0.12 * g

def bass_sus(freq, dur, g=1.0):
    t = tt(dur); s = np.sin(2 * np.pi * freq * t) + 0.15 * np.sin(4 * np.pi * freq * t)
    env = np.minimum(1, t / 0.12) * np.minimum(1, (dur - t) / 0.25)
    return lp(s, 220) * env * 0.4 * g

def pad(freqs, dur, g=1.0, cutoff=1300):
    t = tt(dur); s = np.zeros(len(t))
    for f in freqs:
        for det in (-0.3, 0.0, 0.3):
            ph = (f * (1 + det / 100) * t + rng.random()) % 1.0
            s += 2 * ph - 1
        s += 0.6 * np.sin(2 * np.pi * 2 * f * t)        # octave sine shimmer
    s = lp(s / (len(freqs) * 4), cutoff)
    env = np.minimum(1, t / 0.6) * np.minimum(1, (dur - t) / 0.6)
    return s * env * 0.5 * g

CHORDS = [((110.0, 130.81, 164.81), 55.0), ((87.31, 110.0, 130.81), 43.65),
          ((130.81, 164.81, 196.0), 65.41), ((98.0, 123.47, 146.83), 49.0)]
BEAT = 0.6; BAR = 2.4; T_DROP = 4.20

def build_music():
    m = np.zeros(N)
    place(m, pad(CHORDS[0][0], T_DROP - T['h1'] + 0.6, 0.5, cutoff=650), T['h1'] - 0.3)
    end_groove = IT['v10']['t0']
    t = T_DROP; bar = 0
    while t < end_groove - 0.01:
        chord, root = CHORDS[bar % 4]
        drop = bar == 0
        place(m, pad(chord, BAR + 0.7, 0.85 if drop else 0.7), t - 0.05)
        place(m, bass_sus(root, BAR + 0.1, 1.0 if drop else 0.75), t)
        for b in range(4):
            tb = t + b * BEAT
            if tb >= end_groove: break
            if drop:
                place(m, kick_soft(1.0), tb)
                if b in (1, 3): place(m, clap_soft(0.9), tb)
            place(m, shaker(0.9 if drop else 0.6), tb + BEAT / 2)
            place(m, shaker(0.35), tb)
        t += BAR; bar += 1
    cta = np.zeros(N); t = end_groove; k = 0
    while t < IT['v11']['t0'] + 0.1:
        chord, root = CHORDS[k % 4]
        place(cta, pad(chord, BAR + 0.7, 0.7, cutoff=800), t - 0.05); place(cta, bass_sus(root, BAR + 0.1, 0.6), t)
        t += BAR; k += 1
    m += lp(cta, 900) * db(-5)
    place(m, pad(CHORDS[0][0], 2.6, 0.9, cutoff=1000), IT['v11']['t0'] - 0.05)
    m = verb(m, 0.28)
    fs = int((P.DURATION - 1.8) * SR); fe = int(P.DURATION * SR)
    m[fs:fe] *= np.linspace(1, 0, fe - fs) ** 1.5; m[fe:] = 0
    return m

# ---------------- SFX (smooth) ----------------
def impact_soft(g=1.0, d=1.8):
    t = tt(d); f = 38 + 30 * np.exp(-t * 5)
    sub = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 2.2)
    body = lp(rng.normal(0, 1, len(t)), 400) * np.exp(-t * 5) * 0.35
    env = np.minimum(1, t / 0.018)                       # soft 18 ms attack, no click
    return fade((sub + body) * env, 0.0, 0.3) * 0.8 * g

def swell(d=0.7, g=1.0):
    """reverse-reverb style suck-in, ends exactly at its end time"""
    t = tt(d); u = t / d
    n = lp(rng.normal(0, 1, len(t)), 2500) + 0.3 * hp(rng.normal(0, 1, len(t)), 5000)
    return fade(n * u ** 3 * 0.35, 0.0, 0.04) * g

def whoosh(d=0.55, g=1.0):
    t = tt(d); u = t / d; env = np.sin(np.pi * u) ** 2
    n = rng.normal(0, 1, len(t)); out = np.zeros(len(t)); step = int(0.012 * SR)
    for i in range(0, len(t), step):
        fc = 500 + 2200 * np.sin(np.pi * u[i])
        seg = bp(n[max(0, i - 1500):i + step], fc * 0.55, fc * 1.6)
        out[i:i + step] = seg[-len(n[i:i + step]):]
    return lp(out, 4500) * env * 0.6 * g

def riser(d, g=1.0):
    t = tt(d); u = t / d
    n = rng.normal(0, 1, len(t)); out = np.zeros(len(t)); step = int(0.02 * SR)
    for i in range(0, len(t), step):
        fc = 300 + 6000 * u[i] ** 2
        seg = bp(n[max(0, i - 2000):i + step], fc * 0.6, min(fc * 1.6, SR / 2 - 100))
        out[i:i + step] = seg[-len(n[i:i + step]):]
    tone = np.sin(2 * np.pi * np.cumsum(160 + 500 * u ** 2) / SR) * 0.12
    return fade((out * 0.45 + tone) * u ** 2.4, 0.0, 0.03) * g

def tick(g=1.0):
    t = tt(0.25)
    s = (np.sin(2 * np.pi * 1760 * t) + 0.4 * np.sin(2 * np.pi * 2640 * t)) * np.exp(-t * 28)
    return fade(s * np.minimum(1, t / 0.004) * 0.18, 0.0, 0.05) * g

def swipe(g=1.0):
    t = tt(0.4); u = t / 0.4
    return lp(hp(rng.normal(0, 1, len(t)), 2500), 9000) * np.sin(np.pi * u) ** 3 * 0.18 * g

def build_sfx():
    s = np.zeros(N)
    place_end(s, swell(0.6), T['h1'], db(-10))
    place(s, tick(), T['h2'], db(-8))
    place(s, riser(4.20 - 3.40), 3.40, db(-9))
    place(s, impact_soft(), 4.20, db(-2)); place(s, whoosh(0.5), 4.20 - 0.25, db(-12))
    place(s, whoosh(0.45), 4.80 - 0.22, db(-14))
    place(s, impact_soft(0.6, 1.2), 5.40, db(-8)); place(s, whoosh(0.45), 5.40 - 0.22, db(-14))
    place(s, whoosh(0.55), 6.60 - 0.27, db(-12))
    place(s, tick(), T['card'], db(-6))
    place(s, whoosh(0.5), CUT_A2B - 0.25, db(-17))
    place(s, whoosh(0.6), 18.17 - 0.3, db(-14))
    place_end(s, swell(0.55), T['chuadu'], db(-12))
    for tl in LIST_T: place(s, tick(0.8), tl, db(-14))
    place(s, whoosh(0.5), IT['v9']['t0'] - 0.25, db(-16))
    place_end(s, swell(0.55), T['cahai'], db(-11)); place(s, impact_soft(0.7, 1.2), T['cahai'] + 0.05, db(-12))
    place(s, swipe(), IT['v10']['t0'] + 0.08, db(-10))
    place(s, whoosh(0.5), IT['v11']['t0'] - 0.25, db(-16)); place(s, impact_soft(0.7), IT['v11']['t0'], db(-10))
    return verb(s, 0.25)

def duck_gain(vo, depth_db=-12):
    h = int(0.01 * SR); fr = len(vo) // h
    rms = np.sqrt(np.mean(vo[:fr * h].reshape(fr, h) ** 2, axis=1) + 1e-12)
    env = np.zeros(fr); a_att, a_rel = np.exp(-1 / 6), np.exp(-1 / 45)
    for i in range(fr):
        c = a_att if (i and rms[i] > env[i - 1]) else a_rel
        env[i] = c * env[i - 1] + (1 - c) * rms[i] if i else rms[i]
    g = 1 - (1 - db(depth_db)) * np.clip(env / 0.02, 0, 1)
    g = lp(np.repeat(g, h), 8)
    out = np.ones(len(vo)); out[:len(g)] = g
    return out

if __name__ == '__main__':
    vo = build_vo(); music = build_music(); sfx = build_sfx()
    vrms = np.sqrt(np.mean(vo[np.abs(vo) > 1e-4] ** 2)); mrms = np.sqrt(np.mean(music[int(6.6 * SR):int(36 * SR)] ** 2))
    music *= (vrms * db(-12)) / mrms
    mix = vo + music * duck_gain(vo) + sfx * (vrms / 0.12)
    mix = mix[:int(P.DURATION * SR)]
    sf.write(f'{ROOT}/work/mix_pre.wav', np.stack([mix, mix], 1) * 0.5, SR, subtype='FLOAT')
    subprocess.run(['ffmpeg', '-loglevel', 'error', '-y', '-i', f'{ROOT}/work/mix_pre.wav', '-af',
                    'loudnorm=I=-14:TP=-1:LRA=9', '-ar', str(SR), sys.argv[1]], check=True)
    print('ok')
