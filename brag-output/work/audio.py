"""Cortex brag soundtrack: D major, 120 BPM, 23s. Music + in-key SFX, one mix."""
import numpy as np
from scipy.signal import fftconvolve, butter, sosfilt
from scipy.io import wavfile

SR = 48000
DUR = 23.0
N = int(SR * DUR)
BEAT = 0.5
rng = np.random.default_rng(7)

def hz(m):  # midi -> Hz
    return 440.0 * 2 ** ((m - 69) / 12)

def env_adsr(n, a, d, s, r, sr=SR):
    a, d, r = int(a * sr), int(d * sr), int(r * sr)
    e = np.ones(n) * s
    e[:a] = np.linspace(0, 1, a, endpoint=False) if a else e[:a]
    e[a:a + d] = np.linspace(1, s, max(0, min(d, n - a)))[: max(0, min(d, n - a))]
    if r:
        e[-r:] *= np.linspace(1, 0, r)
    return e

def lp(x, fc, order=2):
    return sosfilt(butter(order, fc, 'low', fs=SR, output='sos'), x)

def hp(x, fc, order=2):
    return sosfilt(butter(order, fc, 'high', fs=SR, output='sos'), x)

def place(buf, sig, t, gain=1.0, pan=0.0):
    i = int(t * SR)
    if i >= buf.shape[1]:
        return
    sig = sig[: buf.shape[1] - i]
    l, r = np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)
    buf[0, i:i + len(sig)] += sig * gain * l * 1.414
    buf[1, i:i + len(sig)] += sig * gain * r * 1.414

# --- voices ---
def pad_voice(m, dur):
    n = int(dur * SR); t = np.arange(n) / SR
    x = np.zeros(n)
    for det in (-0.08, 0.0, 0.07):
        f = hz(m + det)
        ph = rng.random() * 6.28
        # soft saw via few harmonics
        for k in range(1, 7):
            x += np.sin(2 * np.pi * f * k * t + ph * k) / k * (0.9 ** k)
    x = lp(x, 1400)
    return x * env_adsr(n, 0.6, 0.3, 0.8, 0.8) * 0.08

def pluck(m, dur=0.6, bright=1.0):
    n = int(dur * SR); t = np.arange(n) / SR
    f = hz(m)
    x = np.sin(2 * np.pi * f * t) + 0.35 * np.sin(2 * np.pi * 2 * f * t) * np.exp(-t * 18) + 0.15 * np.sin(2 * np.pi * 3 * f * t) * np.exp(-t * 30)
    return x * np.exp(-t * (6.5 / bright)) * env_adsr(n, 0.003, 0, 1, 0.02)

def bell(m, dur=1.6):
    n = int(dur * SR); t = np.arange(n) / SR
    f = hz(m)
    x = (np.sin(2 * np.pi * f * t) * np.exp(-t * 2.5)
         + 0.5 * np.sin(2 * np.pi * f * 2.0 * t) * np.exp(-t * 4)
         + 0.25 * np.sin(2 * np.pi * f * 3.01 * t) * np.exp(-t * 7))
    return x * env_adsr(n, 0.004, 0, 1, 0.05)

def bass(m, dur):
    n = int(dur * SR); t = np.arange(n) / SR
    f = hz(m)
    x = np.sin(2 * np.pi * f * t) + 0.3 * np.sin(2 * np.pi * 2 * f * t)
    x = np.tanh(x * 1.3)
    return lp(x, 500) * env_adsr(n, 0.01, 0.15, 0.7, 0.06)

def kick():
    n = int(0.35 * SR); t = np.arange(n) / SR
    f = 42 + 90 * np.exp(-t * 28)
    ph = 2 * np.pi * np.cumsum(f) / SR
    return np.sin(ph) * np.exp(-t * 9) * 0.9

def hat():
    n = int(0.06 * SR); t = np.arange(n) / SR
    return hp(rng.standard_normal(n), 7000) * np.exp(-t * 70) * 0.25

def clap():
    n = int(0.22 * SR); t = np.arange(n) / SR
    x = lp(hp(rng.standard_normal(n), 900), 5000) * np.exp(-t * 22)
    return x * 0.35

def tick(m):
    # typing: tiny click + faint in-key blip
    n = int(0.05 * SR); t = np.arange(n) / SR
    c = hp(rng.standard_normal(n), 3000) * np.exp(-t * 260) * 0.5
    b = np.sin(2 * np.pi * hz(m) * t) * np.exp(-t * 90) * 0.35
    return c + b

def riser(dur):
    n = int(dur * SR); t = np.arange(n) / SR
    x = rng.standard_normal(n)
    out = np.zeros(n); blk = 2400
    for i in range(0, n, blk):
        fc = 300 + 7000 * (i / n) ** 2
        out[i:i + blk] = lp(x[i:i + blk], fc)
    sweep = np.sin(2 * np.pi * np.cumsum(hz(50) * 2 ** (2 * t / dur)) / SR) * 0.3
    return (out * 0.5 + sweep) * (t / dur) ** 2

def boom():
    n = int(2.2 * SR); t = np.arange(n) / SR
    f = hz(26) + 40 * np.exp(-t * 10)
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 2.2)
    nz = lp(rng.standard_normal(n), 1200) * np.exp(-t * 6) * 0.25
    return (x + nz) * 0.9

def swell(dur):
    n = int(dur * SR); t = np.arange(n) / SR
    x = lp(rng.standard_normal(n), 2500) * (t / dur) ** 3
    return x * 0.25

# --- arrangement ---
music = np.zeros((2, N)); drums = np.zeros((2, N)); sfx = np.zeros((2, N))
D, E, Fs, G, A, B, Cs = 62, 64, 66, 67, 69, 71, 73
CH = {  # chord tones (midi) + bass root
    'Bm': ([B - 12, D, Fs], 47), 'D': ([D, Fs, A], 38), 'A': ([Cs - 12 + 12, E, A - 12 + 12], 45),
    'G': ([G - 12, B - 12, D], 43), 'Dmaj9': ([D, Fs, A, E + 12], 38),
}
CH['A'] = ([A - 12, Cs, E], 45)

# hook: sparse Bm pad, filtered, tension
for m in CH['Bm'][0]:
    place(music, lp(pad_voice(m, 4.2), 900), 0.0, 0.8)
place(music, bass(35, 3.6) * 0.5, 0.2, 0.6)

prog = ['D', 'A', 'Bm', 'G', 'D', 'A', 'Bm', 'G']
for k, name in enumerate(prog):
    t0 = 4.0 + k * 2.0
    tones, root = CH[name]
    for m in tones:
        place(music, pad_voice(m, 2.25), t0, 0.9, pan=(m % 5 - 2) * 0.15)
    for b8 in range(4 * 2):  # 8th-note bass
        tb = t0 + b8 * 0.25
        if 15.5 <= tb < 17.5 and b8 % 2:
            continue
        place(music, bass(root, 0.22), tb, 0.55 if b8 % 2 == 0 else 0.35)
    # arp from 8.0
    if t0 >= 8.0:
        arp = [tones[0] + 12, tones[1] + 12, tones[2] + 12, tones[1] + 12]
        for s16 in range(16):
            ts = t0 + s16 * 0.125
            if 15.5 <= ts < 17.4:
                continue
            place(music, pluck(arp[s16 % 4], 0.35, 0.6), ts, 0.11, pan=0.35 if s16 % 2 else -0.35)

# outro: Dmaj9 resolve at 20.0 with swell into it
place(music, swell(0.6), 19.4, 0.8)
for m in CH['Dmaj9'][0]:
    place(music, pad_voice(m, 3.0), 20.0, 1.0, pan=(m % 5 - 2) * 0.2)
place(music, bass(38, 2.6), 20.0, 0.6)

# drums: 4.0 -> 15.5, back 17.5 -> 20.0
def drum_on(t):
    return (4.0 <= t < 15.5) or (17.5 <= t < 20.0)
for i in range(int(DUR / BEAT)):
    tb = i * BEAT
    if drum_on(tb):
        place(drums, kick(), tb, 0.55)
        if tb >= 8.0 and i % 2 == 1:
            place(drums, clap(), tb, 0.45, pan=0.05)
    th = tb + 0.25
    if drum_on(th) and th >= 6.0:
        place(drums, hat(), th, 0.5, pan=0.3)

# --- SFX (in key, soft) ---
scale = [D + 12, E + 12, Fs + 12, A + 12, B + 12]
def typing(t0, t1, nchars, seed):
    r = np.random.default_rng(seed)
    for c in range(nchars):
        tc = t0 + (t1 - t0) * c / nchars + r.uniform(-0.004, 0.004)
        place(sfx, tick(scale[r.integers(0, 5)] + 12), tc, 0.11, pan=r.uniform(-0.2, 0.2))
typing(0.2, 1.15, 28, 1)
place(sfx, pluck(B + 12, 0.8), 1.3, 0.16)                   # AI reply
place(sfx, boom() * 0.35, 1.5, 0.5)                          # headline lands
for k, m in enumerate([Fs + 12, B + 12, D + 24]):             # ghost cards stack
    place(sfx, bell(m, 1.2), 1.5 + k * 0.12, 0.09, pan=(k - 1) * 0.3)
place(sfx, riser(0.95), 3.05, 0.35)                          # into reveal
place(sfx, boom(), 4.0, 0.55)                                # burst impact
place(sfx, bell(D + 24, 2.0), 4.0, 0.08)
for k, m in enumerate([D + 12, Fs + 12, A + 12]):             # labels
    pass
for k in range(5):                                           # labels pop
    place(sfx, pluck([A + 12, B + 12, D + 24, E + 24, Fs + 24][k], 0.5), 5.3 + k * 0.18, 0.07, pan=(k - 2) * 0.2)
typing(8.4, 9.2, 16, 2)                                      # search typing
for r_ in range(5):                                          # results cascade
    place(sfx, pluck([D + 24, E + 24, Fs + 24, A + 24, B + 24][r_], 0.5, 0.8), 9.3 + r_ * 0.07, 0.07, pan=-0.3)
place(sfx, tick(A + 24) * 1.5, 11.5, 0.2)                    # click
place(sfx, bell(A + 12, 1.8), 11.52, 0.12)
for e in range(5):                                           # edges grow
    place(sfx, pluck([D + 24, Fs + 24, A + 24, B + 24, D + 36][e], 0.6), 12.7 + e * 0.12, 0.06, pan=0.3)
typing(15.75, 16.5, 22, 3)                                   # callback typing
place(sfx, bell(Fs + 12, 1.6), 16.65, 0.12)                  # tool call
for k in range(3):
    place(sfx, pluck([A + 12, D + 24, Fs + 24][k], 0.5), 16.9 + k * 0.13, 0.08, pan=-0.2)
place(sfx, swell(0.5), 17.0, 0.5)
place(sfx, bell(D + 24, 2.5), 17.5, 0.1)                     # memory reply
place(sfx, boom() * 0.5, 20.0, 0.45)                         # outro
place(sfx, bell(A + 12, 3.0), 20.0, 0.1)
place(sfx, bell(D + 24, 3.0), 20.2, 0.08)

# --- mix: shared room so everything sits in one space ---
def ir(sec=2.4, decay=3.2):
    n = int(sec * SR); t = np.arange(n) / SR
    out = []
    for s in (11, 12):
        r = np.random.default_rng(s).standard_normal(n) * np.exp(-t * decay)
        out.append(lp(r, 6000))
    return np.array(out) / 30

IR = ir()
def verb(x, send):
    return np.array([fftconvolve(x[c], IR[c])[:N] for c in range(2)]) * send

music_hp = np.array([hp(music[c], 40) for c in range(2)])
sfx_f = np.array([lp(sfx[c], 9000) for c in range(2)])
drums_f = np.array([lp(drums[c], 11000) for c in range(2)])

# gentle sidechain duck of music on kicks
duck = np.ones(N)
for i in range(int(DUR / BEAT)):
    tb = i * BEAT
    if drum_on(tb):
        a = int(tb * SR); n = int(0.22 * SR)
        duck[a:a + n] = np.minimum(duck[a:a + n], 1 - 0.3 * np.exp(-np.arange(n) / SR * 14))

mix = (music_hp * duck * 0.7 + verb(music_hp, 0.3)
       + drums_f * 0.8 + verb(drums_f, 0.12)
       + sfx_f * 0.9 + verb(sfx_f, 0.5))

# master: fade in/out, soft clip, normalize
t = np.arange(N) / SR
mix *= np.clip(t / 0.05, 0, 1) * np.clip((DUR - t) / 1.6, 0, 1) ** 1.5
mix *= 10 ** (-14.5 / 20) / np.sqrt(np.mean(mix ** 2))
pk = 0.85
mix = np.where(np.abs(mix) > pk, np.sign(mix) * (pk + (1 - pk) * np.tanh((np.abs(mix) - pk) / (1 - pk))), mix)
wavfile.write('soundtrack.wav', SR, (mix.T * 0.92 * 32767).astype(np.int16))
print('ok', mix.shape, float(np.sqrt(np.mean(mix ** 2))))
