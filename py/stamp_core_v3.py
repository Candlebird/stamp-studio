"""Palette-knife painting for Postage Stamp V3, pure numpy (no scipy/skimage/PIL),
so the same code runs inside GIMP's Python and from the command line.

This paints in passes on a surface holding both color and paint height. The
painter looks at a slightly loosened version of the photo, puts down a toned
ground, then works big to small. Each style is a recipe of passes.

A blade's mark has a life: it lands as a lump with a straight skewed edge, its
width follows the hand's pressure, the paint runs out at its own point and then
catches only on high spots, and it ends in a ragged lift-off with a raised lip.
Different parts of the knife make different marks: the flat for planes, the tip
for dabs, the edge for lines. The finished surface is then lit from the side.
"""
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

# Palettes are gradient-map stops, dark -> light, sRGB 0..255.
PALETTES = {
    "desert": [(52, 34, 38), (118, 52, 40), (178, 84, 52), (214, 140, 92),
               (232, 196, 150), (246, 232, 206)],
    "dusk":   [(34, 32, 58), (84, 58, 92), (160, 82, 84), (220, 128, 90),
               (240, 190, 140), (250, 236, 210)],
    "forest": [(24, 40, 36), (48, 82, 60), (98, 122, 74), (168, 164, 104),
               (220, 206, 156), (244, 238, 214)],
    "coast":  [(22, 40, 62), (40, 86, 110), (70, 140, 150), (150, 194, 180),
               (226, 220, 190), (248, 244, 230)],
}
PALETTE_NAMES = ["none"] + list(PALETTES)
_F = np.float32
_LUMA = np.array([0.299, 0.587, 0.114], _F)

# A stroke's sample packs into one int64 so a single scatter records it:
# stroke id | position along the stroke (1/4 px) | position across it (0..255).
_U_BITS, _V_BITS = 16, 8
_U_MAX = (1 << _U_BITS) - 1
_SAMPLE = 0.7          # raster sample spacing in px; < 1 leaves no holes
_CHUNK = 1000000       # samples per scatter, bounds memory


# ---------------------------------------------------------------- blurring

def _win_sum(a, r):
    """Sum over a 2r+1 window along axis 0, edge-padded. Adding whole rows in a
    Python loop is several times faster than cumsum(0), which walks columns."""
    c = np.pad(a, ((r + 1, r),) + ((0, 0),) * (a.ndim - 1), mode="edge")
    for i in range(1, len(c)):
        c[i] += c[i - 1]
    k = 2 * r + 1
    return c[k:] - c[:-k]


def _box(a, r):
    s = _win_sum(a, r)
    s = _win_sum(np.ascontiguousarray(s.swapaxes(0, 1)), r).swapaxes(0, 1)
    return s / _F((2 * r + 1) ** 2)


def _shrink(a, f):
    h, w = a.shape[:2]
    gh, gw = -(-h // f), -(-w // f)
    p = np.pad(a, ((0, gh * f - h), (0, gw * f - w)) + ((0, 0),) * (a.ndim - 2),
               mode="edge")
    return p.reshape((gh, f, gw, f) + a.shape[2:]).mean((1, 3), dtype=_F)


def _grow(a, f, h, w):
    """Bilinear upsample of a _shrink result back to h x w."""
    def taps(n, m):
        p = (np.arange(n, dtype=_F) + _F(0.5)) / _F(f) - _F(0.5)
        i = np.clip(np.floor(p).astype(np.int64), 0, m - 1)
        return i, np.minimum(i + 1, m - 1), np.clip(p - i, 0, 1).astype(_F)
    y0, y1, ty = taps(h, a.shape[0])
    x0, x1, tx = taps(w, a.shape[1])
    ty = ty.reshape((-1, 1) + (1,) * (a.ndim - 2))
    tx = tx.reshape((1, -1) + (1,) * (a.ndim - 2))
    rows = a[y0] * (1 - ty) + a[y1] * ty
    return rows[:, x0] * (1 - tx) + rows[:, x1] * tx


def _smooth(a, r):
    """Gaussian-like blur of radius ~r. Wide blurs run on a coarse grid."""
    if r < 1:
        return a
    r = int(round(r))
    if r < 6:
        return _box(_box(a, max(1, r // 2 + r % 2)), max(1, r // 2))
    f = max(2, r // 3)
    return _grow(_box(_box(_shrink(a, f), 2), 1), f, a.shape[0], a.shape[1])


# ---------------------------------------------------------------- noise

def _noise(rng, h, w, r):
    """Smooth noise in 0..1, roughly uniform, feature size ~r px."""
    f = max(1, int(r))
    n = rng.standard_normal((-(-h // f) + 3, -(-w // f) + 3)).astype(_F)
    n = _box(n, 1)
    n = _grow(n, f, h, w) if f > 1 else n[:h, :w]
    n -= n.mean()
    n /= max(float(n.std()), 1e-6)
    return np.clip(n * _F(0.22) + _F(0.5), 0, 1)


def _bristle_tile(rng, n=512, m=128):
    """Tileable bristle texture: streaks that run along the stroke (axis 0).
    The long axis uses wrap-padded smoothing so tile seams can't show."""
    t = rng.standard_normal((n, m)).astype(_F)
    for _ in range(3):
        for k in (1, 2, 4, 8, 16):
            t = (t + np.roll(t, k, 0) + np.roll(t, -k, 0)) / _F(3)
    t = t + _F(0.45) * np.roll(t, 1, 1) + _F(0.45) * np.roll(t, -1, 1)
    t -= t.mean()
    t /= max(float(t.std()), 1e-6)
    return np.clip(t * _F(0.25) + _F(0.5), 0, 1)


# ---------------------------------------------------------------- direction

def _tensor(lum, r):
    g = _smooth(lum, r)
    gy = np.zeros_like(g)
    gx = np.zeros_like(g)
    gy[1:-1] = g[2:] - g[:-2]
    gx[:, 1:-1] = g[:, 2:] - g[:, :-2]
    return np.stack([gx * gx, gx * gy, gy * gy], -1)


def _flow(J, rad, coarse, wobble, angle=np.deg2rad(62)):
    """Stroke directions for one layer: the tangent of the smoothed structure
    tensor. Flat areas (sky, walls) fall back on the wider tensor and then on a
    fixed hand angle, like a painter's habitual diagonal."""
    t = _smooth(J, rad) + _F(0.5) * coarse
    bias = _F(1.5e-5)
    c, s = np.cos(angle + np.pi / 2), np.sin(angle + np.pi / 2)
    a = t[..., 0] + bias * _F(c * c)
    b = t[..., 1] + bias * _F(c * s)
    d = t[..., 2] + bias * _F(s * s)
    th = 0.5 * np.arctan2(2 * b, a - d) + _F(np.pi / 2) + wobble
    return np.cos(th).astype(_F), np.sin(th).astype(_F)


# ---------------------------------------------------------------- strokes

def _seeds(err, spacing, thresh, rng):
    """Jittered grid; in each cell take the worst-painted pixel of a few random
    picks, and drop the cell if the canvas there is already good enough."""
    h, w = err.shape
    gy = np.arange(0, h, spacing)
    gx = np.arange(0, w, spacing)
    cy, cx = [a.ravel() for a in np.meshgrid(gy, gx, indexing="ij")]
    n = len(cy)
    picks = 3
    y = np.minimum((cy + rng.random((picks, n)) * spacing).astype(np.int64), h - 1)
    x = np.minimum((cx + rng.random((picks, n)) * spacing).astype(np.int64), w - 1)
    e = err[y, x]
    best = e.argmax(0)
    i = np.arange(n)
    y, x, e = y[best, i], x[best, i], e[best, i]
    keep = e > thresh
    return x[keep], y[keep]


def _trace(sx, sy, col, ref, tx, ty, grid, step, nmax, nmin, tol, spin, inertia=0.55,
           limit=None):
    """Walk every stroke at once, both ways from its seed, along the flow field.
    A stroke ends when the photo under it stops matching its color. tx, ty sit
    on a grid `grid` times coarser than the image; spin turns each stroke off
    the field by its own angle, so big strokes cross like a scrubbing hand.
    limit gives a stroke its own forward length and makes it pull one way only.
    Returns the path (2K+1, n, 2) with the seed at K, and the steps taken each way."""
    h, w = ref.shape[:2]
    fw = tx.shape[1]
    n = len(sx)
    K = nmax
    path = np.empty((2 * K + 1, n, 2), _F)
    path[K, :, 0], path[K, :, 1] = sx, sy
    steps = []
    refl = ref.reshape(-1, 3)
    sc, ss = np.cos(spin).astype(_F), np.sin(spin).astype(_F)
    keep, turn = _F(inertia), _F(1 - inertia)
    txl, tyl = tx.ravel(), ty.ravel()
    for sign in (1, -1):
        x, y = sx.astype(_F), sy.astype(_F)
        i0 = (sy // grid) * fw + sx // grid
        dx = (txl[i0] * sc - tyl[i0] * ss) * sign
        dy = (txl[i0] * ss + tyl[i0] * sc) * sign
        alive = np.ones(n, bool)
        count = np.zeros(n, np.int32)
        for k in range(1, K + 1):
            xi = np.clip((x + _F(0.5)).astype(np.int64), 0, w - 1)
            yi = np.clip((y + _F(0.5)).astype(np.int64), 0, h - 1)
            i = (yi // grid) * fw + xi // grid
            fx = txl[i] * sc - tyl[i] * ss
            fy = txl[i] * ss + tyl[i] * sc
            flip = np.where(fx * dx + fy * dy < 0, _F(-1), _F(1))
            ndx = keep * dx + turn * fx * flip             # inertia: no kinks
            ndy = keep * dy + turn * fy * flip
            m = np.maximum(np.hypot(ndx, ndy), _F(1e-4))
            dx = np.where(alive, ndx / m, dx)
            dy = np.where(alive, ndy / m, dy)
            nx = x + dx * _F(step) * alive
            ny = y + dy * _F(step) * alive
            inside = (nx >= 0) & (nx <= w - 1) & (ny >= 0) & (ny <= h - 1)
            xi = np.clip((nx + _F(0.5)).astype(np.int64), 0, w - 1)
            yi = np.clip((ny + _F(0.5)).astype(np.int64), 0, h - 1)
            diff = np.abs(refl[yi * w + xi] - col).sum(1)
            ok = inside & ((diff < tol) | (k <= nmin))
            if limit is not None:
                ok &= k <= (limit if sign > 0 else 0)
            alive &= ok
            x = np.where(alive, nx, x)
            y = np.where(alive, ny, y)
            count += alive
            path[K + sign * k, :, 0] = x
            path[K + sign * k, :, 1] = y
            if not alive.any():      # steps past a stroke's end are never read
                break
        steps.append(count)
    return path, steps[1], steps[0]


# ---------------------------------------------------------------- finishing

def _gradient_map(lum, stops):
    stops = np.asarray(stops, float) / 255.0
    pos = np.linspace(0, 1, len(stops))
    return np.stack([np.interp(lum, pos, stops[:, c]) for c in range(3)], -1).astype(_F)


def _grade(out, contrast, vibrance, palette, palette_strength):
    # paint has no true black: open the deepest shadows a little
    out = out + _F(0.045) * (1 - out) ** 4
    lum = out @ _LUMA
    s = lum * lum * (3 - 2 * lum)                       # gentle S-curve
    l2 = lum * _F(1 - contrast) + s * _F(contrast)
    out = out * (l2 / np.maximum(lum, _F(1e-4)))[..., None]
    g = (out @ _LUMA)[..., None]                        # vibrance
    sat = out.max(-1, keepdims=True) - out.min(-1, keepdims=True)
    out = g + (out - g) * (1 + _F(vibrance) * (1 - np.clip(sat * 2, 0, 1)))
    if palette in PALETTES and palette_strength > 0:     # split-tone tint
        l = np.clip(out @ _LUMA, 0, 1)
        tint = _gradient_map(l, PALETTES[palette])
        wgt = (np.abs(l - _F(0.5)) * 2) ** _F(1.5) * _F(palette_strength)
        out = out * (1 - wgt[..., None]) + tint * wgt[..., None]
    return np.clip(out, 0, 1)


# ---------------------------------------------------------------- painting a pass

class _Surface:
    """The painting so far: color and paint height build up pass by pass, so a
    stroke can drag wet paint, skip over canvas tooth, or sit on old ridges."""

    def __init__(self, img, seed, scale):
        self.h, self.w = img.shape[:2]
        self.scale = scale
        self.rng = np.random.default_rng(seed)
        self.tile = _bristle_tile(self.rng)
        self.C = None
        self.H = np.zeros((self.h, self.w), _F)
        self.tooth = None
        self.grid = 2 if min(self.h, self.w) >= 64 else 1
        self.J = None
        self.coarse = None
        self.wobble = None
        self.focus = None


def _mix_paint(col, P, rng, vary):
    """A stroke's paint: the photo's color as a painter mixes it on the palette,
    never exactly. broken turns the hue, for side by side colors that blend in
    the eye."""
    n = len(col)
    L = col @ _LUMA
    a = col[:, 0] - L
    b = col[:, 2] - L
    ang = (rng.standard_normal(n) * P["broken"] * vary).astype(_F)
    ca, sa = np.cos(ang), np.sin(ang)
    ch = _F(P["chroma"])
    a2 = (a * ca - b * sa) * ch + rng.standard_normal(n).astype(_F) * _F(P["fleck"] * vary)
    b2 = (a * sa + b * ca) * ch + rng.standard_normal(n).astype(_F) * _F(P["fleck"] * vary)
    L2 = L * (1 + rng.standard_normal(n).astype(_F) * _F(P["jitter"] * vary))
    r = L2 + a2
    bl = L2 + b2
    g = (L2 - _F(0.299) * r - _F(0.114) * bl) / _F(0.587)
    return np.clip(np.stack([r, g, bl], -1), 0, 1)


def _unpack(top):
    idx = np.flatnonzero(top.ravel() >= 0)
    t = top.ravel()[idx]
    sid = t >> (_U_BITS + _V_BITS)
    u = ((t >> _V_BITS) & _U_MAX).astype(_F) * _F(0.25)
    v = (t & 255).astype(_F) * _F(1 / 127.5) - 1
    return idx, sid, u, v


def _sample_tile(S, off, sid, u, v, r, hair, coarse=1):
    """Bristle value per pixel. hair is the pitch of the hairs in pixels."""
    th, tw = S.tile.shape
    across = (v * r * _F(3.0 / (hair * coarse))).astype(np.int64)
    along = (u * _F(0.8 / (S.scale * coarse))).astype(np.int64)
    return S.tile[(along + off[sid, 0]) % th, (across + off[sid, 1]) % tw]


def _error(S, P, ref, lum_ref):
    """Where the canvas still misses the subject. Smoothing before abs lets
    color jitter average out, and the focus map loosens the outskirts."""
    R = P["R"]
    err = np.abs(_smooth(S.C - ref, R)).sum(-1)
    err = err / (_F(0.35) + _smooth(lum_ref, R)) * _F(0.8)
    f = _F(P["focus"])
    return err * (1 - f + f * S.focus)


# ---------------------------------------------------------------- seeing

def _see(img, S, warp, mass, temp, sat):
    """What the painter sees: the drawing drifts a little, shadows group into
    simple masses, lights warm up and shadows cool down."""
    h, w = S.h, S.w
    sc = S.scale
    if warp > 0:
        amp = warp * sc
        dx = (_noise(S.rng, h, w, 70 * sc) - _F(0.5)) * _F(6 * amp) \
            + (_noise(S.rng, h, w, 18 * sc) - _F(0.5)) * _F(3 * amp)
        dy = (_noise(S.rng, h, w, 70 * sc) - _F(0.5)) * _F(6 * amp) \
            + (_noise(S.rng, h, w, 18 * sc) - _F(0.5)) * _F(3 * amp)
        x = np.clip((np.arange(w)[None, :] + dx + _F(0.5)).astype(np.int64), 0, w - 1)
        y = np.clip((np.arange(h)[:, None] + dy + _F(0.5)).astype(np.int64), 0, h - 1)
        img = img[y, x]
    if mass > 0:
        soft = _smooth(img, 6 * sc)
        m = np.clip((_F(0.40) - soft @ _LUMA) / _F(0.3), 0, 1) * _F(mass)
        img = img + (soft - img) * m[..., None]
    lum = (img @ _LUMA)[..., None]
    img = lum + (img - lum) * _F(1 + sat)
    shift = (lum - _F(0.45)) * _F(temp)
    img = img + shift * np.array([0.09, 0.0, -0.10], _F)
    return np.clip(img, 0, 1)


def _focus_map(S, img):
    """Where the eye goes: busy areas toward the middle of the picture."""
    e = np.sqrt(S.J[..., 0] + S.J[..., 2])
    e = _smooth(e, 40 * S.scale / S.grid)
    e = np.clip(e / max(float(np.percentile(e, 85)), 1e-6), 0, 1)
    gh, gw = e.shape
    y = (np.arange(gh, dtype=_F) / max(1, gh - 1) - _F(0.5))[:, None]
    x = (np.arange(gw, dtype=_F) / max(1, gw - 1) - _F(0.5))[None, :]
    c = np.exp(-(x * x + y * y) / _F(0.16))
    f = _F(0.3) + _F(0.7) * np.clip(e * (_F(0.35) + _F(0.65) * c) * _F(1.3), 0, 1)
    return _grow(f, S.grid, S.h, S.w) if S.grid > 1 else f


# ---------------------------------------------------------------- surfaces

def _weave(rng, h, w, period):
    """Linen canvas, 0..1: two thread directions with uneven threads."""
    y = np.arange(h, dtype=_F)[:, None]
    x = np.arange(w, dtype=_F)[None, :]
    k = _F(np.pi / period)
    wob = (_noise(rng, h, w, period * 9) - _F(0.5)) * _F(period * 1.6)
    wob2 = (_noise(rng, h, w, period * 9) - _F(0.5)) * _F(period * 1.6)
    over = (np.floor((x + wob) / _F(period)) + np.floor((y + wob2) / _F(period))) % 2
    warp_t = np.abs(np.sin((x + wob) * k))        # threads running up and down
    weft_t = np.abs(np.sin((y + wob2) * k))       # threads running across
    t = np.where(over > 0, warp_t * (_F(0.55) + _F(0.45) * weft_t),
                 weft_t * (_F(0.55) + _F(0.45) * warp_t))
    t = t + (_noise(rng, h, w, 2) - _F(0.5)) * _F(0.35)
    return np.clip(t, 0, 1)


def _light(S, relief, gloss):
    """Raking light from the upper left, over the real surface: bristle grooves
    and ridges shade, thick paint glints, and valleys sit in shadow."""
    H = S.H
    C = S.C
    gy = np.zeros_like(H)
    gx = np.zeros_like(H)
    gy[1:-1] = (H[2:] - H[:-2]) * _F(relief)
    gx[:, 1:-1] = (H[:, 2:] - H[:, :-2]) * _F(relief)
    # soft-limit slopes: a stroke's edge is a ridge, not a black outline
    gx = gx / (1 + np.abs(gx) * _F(1.6))
    gy = gy / (1 + np.abs(gy) * _F(1.6))
    inv = 1 / np.sqrt(1 + gx * gx + gy * gy)
    lx, ly, lz = -0.55, -0.62, 0.56
    diff = (-gx * _F(lx) - gy * _F(ly) + _F(lz)) * inv / _F(lz)
    shade = _F(0.42) + _F(0.58) * np.clip(diff, 0, 1.5)
    hx, hy, hz = -0.30, -0.34, 0.89
    spec = np.clip((-gx * _F(hx) - gy * _F(hy) + _F(hz)) * inv, 0, 1) ** _F(36)
    cav = H - _box(H, max(2, int(3 * S.scale)))
    shade = shade * (1 + np.clip(cav * _F(0.5 * relief), -0.22, 0.08))
    return C * shade[..., None] + (spec * _F(gloss))[..., None]


# ---------------------------------------------------------------- recipes

def _sizes(big, small, ratio=0.6):
    radii = []
    r = big
    while r > small * 1.3:
        radii.append(r)
        r *= ratio
    radii.append(small)
    return radii


def _lerp(a, b, t):
    return a + (b - a) * t


# ---------------------------------------------------------------- the knife

def _petal(tt):
    """Width of a pressed dab along its length: a round shoulder near where the
    blade landed, drawn out to a soft point where it lifted."""
    s = np.sin(_F(np.pi) * np.clip(tt, 0, 1) ** _F(0.66))
    return np.maximum(s, 0) ** _F(0.62)


def _knife_footprints(c, path, nb, nf, step, rad, ids, length, cap, nu, nv, vn, edge,
                      w, h, shape):
    """Like _footprints, but a blade's mark is not symmetric. It lands with a
    straight, skewed edge, its width follows the hand's pressure, and it ends
    in a ragged lift-off."""
    skew, pinch, tilt, wave, rag, flare, rough, petal = shape
    K = (path.shape[0] - 1) // 2
    counts = nu[c].astype(np.intp)          # repeat counts must be native ints (32-bit in the browser)
    sid = np.repeat(c, counts)
    firsts = np.cumsum(counts) - counts
    k = np.arange(counts.sum()) - np.repeat(firsts, counts)
    capS = cap[sid]
    u = k.astype(_F) * _F(_SAMPLE) - capS
    t = u / _F(step) - nb[sid]
    lo = (-nb[sid]).astype(np.int64)
    hi = np.maximum(nf[sid].astype(np.int64) - 1, lo)
    i = np.clip(np.floor(t).astype(np.int64), lo, hi)
    fr = t - i
    p0 = path[K + i, sid]
    seg = path[K + i + 1, sid] - p0
    px = p0[:, 0] + seg[:, 0] * fr
    py = p0[:, 1] + seg[:, 1] * fr
    sl = np.maximum(np.hypot(seg[:, 0], seg[:, 1]), _F(1e-4))
    r = rad[sid]
    nx, ny = -seg[:, 1] / sl * r, seg[:, 0] / sl * r
    X = px[:, None] + nx[:, None] * vn + _F(0.5)
    Y = py[:, None] + ny[:, None] * vn + _F(0.5)
    rows = len(edge)
    L = length[sid] + 2 * capS
    u0 = u + capS
    tt = u0 / L
    # landing: one straight edge of the blade, set down at an angle
    land = u0[:, None] > capS[:, None] * (_F(0.15) + skew[sid][:, None] * (vn + 1))
    # lift-off: the paint tears away unevenly, in fingers
    tail = (L - u0)[:, None] > capS[:, None] * (_F(0.1) + rag[sid][:, None]
                                                * edge[(sid * 131) % rows] * _F(2.4))
    # pressure: the mark narrows as the hand eases and drifts to one side
    half = 1 - pinch[sid] * tt * tt + wave[sid] * np.sin(tt * _F(5.0) + sid.astype(_F))
    half = half * (1 - flare[sid] * np.clip(1 - tt / _F(0.3), 0, 1) ** 2)   # blade rolls down
    mid = tilt[sid] * tt
    # the pressed side of the blade cuts clean; the other side breaks up
    hand = np.where(sid % 2 == 0, _F(1), _F(-1))[:, None]
    wander = edge[((k // 2 + sid * 37)) % rows]
    loose = np.clip(vn * hand + _F(0.15), 0, 1)
    side = wander * (_F(0.03) + loose * (rough[sid] * (_F(0.3) + tt))[:, None])
    if petal:
        lean = _petal(tt) * (1 + _F(0.04) * np.sin(tt * _F(7) + sid.astype(_F)))
        keep = np.abs(vn) < lean[:, None]
    else:
        keep = land & tail & (np.abs(vn - mid[:, None]) < half[:, None] - side)
    keep &= (X >= 0) & (X < w) & (Y >= 0) & (Y < h)
    idx = Y[keep].astype(np.int64) * w + X[keep].astype(np.int64)
    pu = np.minimum((u0 * _F(4)).astype(np.int64), _U_MAX)
    head = (ids[sid] << (_U_BITS + _V_BITS)) | (pu << _V_BITS)
    vq = ((vn + 1) * _F(127.5)).astype(np.int64)
    return idx, (head[:, None] | vq)[keep]


def _knife_rasterize(top, path, nb, nf, step, rad, tile, shape):
    h, w = top.shape
    flat = top.reshape(-1)
    length = (nb + nf).astype(_F) * _F(step)
    nv = int(np.ceil(2.0 * rad.max() / _SAMPLE)) + 1
    vn = np.linspace(-1, 1, nv, dtype=_F)
    edge = np.ascontiguousarray(tile[:, (np.arange(nv) * (tile.shape[1] // nv))])
    cap = rad * _F(0.7)
    nu = np.ceil((length + 2 * cap) / _F(_SAMPLE)).astype(np.int64) + 1
    ids = np.arange(len(rad), dtype=np.int64)
    cuts = np.searchsorted(np.cumsum(nu) * nv, np.arange(_CHUNK, int(nu.sum()) * nv, _CHUNK))
    chunks = [c for c in np.split(np.arange(len(rad)), np.unique(cuts)) if len(c)]
    workers = min(os.cpu_count() or 1, 8)
    with ThreadPoolExecutor(workers) as ex:
        for i in range(0, len(chunks), workers):
            for idx, val in ex.map(
                    lambda c: _knife_footprints(c, path, nb, nf, step, rad, ids, length,
                                                cap, nu, nv, vn, edge, w, h, shape),
                    chunks[i:i + workers]):
                flat[idx] = val
    return length + 2 * cap


def _new_knife(R, fine, **kw):
    p = dict(R=R, fine=fine, kind="lay", run=3.0, spacing=1.2, spin=0.3, inertia=0.9,
             tol=1.0, thresh=1.0, focus=1.0, size_var=0.3, len_var=0.4,
             out=(0.6, 1.05), skew=(0.1, 0.9), pinch=(0.0, 0.5), tilt=0.2, wave=0.05,
             rag=(0.5, 1.5), thick=1.0, lump=0.9, lip=0.5, ridge=0.5, ridge_bias=0.6,
             chatter=0.04, scrape=0.3, marble=0.7, ribbon=3.0, wet=0.2, flatten=0.9,
             stick=1.0, jitter=0.04, broken=0.0, fleck=0.0, chroma=1.05, hair=5.0,
             light=0.0, min_run=2.2, flare=(0.0, 0.6), rough=(0.25, 0.8), respect=0.0,
             tone=0.2, fit=0.5)
    p.update(kw)
    return p


def _spread(rng, n, lohi):
    lo, hi = lohi
    return (lo + (hi - lo) * rng.random(n)).astype(_F)


def _knife_marks(S, P, ref, err, thresh, detail, length_mul, vary):
    """Set one pass of blade marks down on the surface."""
    R, fine, rng = P["R"], P["fine"], S.rng
    h, w = S.h, S.w
    tx, ty = _flow(S.J, max(1.0, R * 1.2 / S.grid), S.coarse, S.wobble)
    if P.get("level"):
        # the ground lies flat: lower in the picture, sweeps run level
        m = np.clip((np.arange(tx.shape[0], dtype=_F) / tx.shape[0] - _F(0.5)) / _F(0.15), 0, 1)
        m = (m * _F(P["level"]))[:, None]
        sgn = np.where(tx < 0, _F(-1), _F(1))
        tx, ty = tx * sgn * (1 - m) + m, ty * sgn * (1 - m)
        nrm = np.maximum(np.hypot(tx, ty), _F(1e-4))
        tx, ty = tx / nrm, ty / nrm
    sx, sy = _seeds(err, max(1, int(round(R * P["spacing"]))), thresh, rng)
    n = len(sx)
    if n == 0:
        return
    order = rng.permutation(n)
    sx, sy = sx[order], sy[order]
    petal = P["kind"] == "petal"
    if petal:
        # a dab that won't fit inside its shape isn't made; a smaller one will be
        i0 = (sy // S.grid) * tx.shape[1] + sx // S.grid
        dx, dy = tx.ravel()[i0], ty.ravel()[i0]
        fit = np.ones(n, bool)
        for ax, ay in ((1.6, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0), (0.8, 0.8), (0.8, -0.8)):
            px = np.clip(sx + (dx * ax - dy * ay) * R, 0, w - 1).astype(np.int64)
            py = np.clip(sy + (dy * ax + dx * ay) * R, 0, h - 1).astype(np.int64)
            fit &= np.abs(S.sharp[py, px] - S.sharp[sy, sx]).sum(1) < _F(P["fit"])
        sx, sy = sx[fit], sy[fit]
        n = len(sx)
        if n == 0:
            return
    col = _mix_paint(ref[sy, sx], P, rng, vary)
    if P["light"]:
        col = np.clip(col * _F(1 + P["light"]), 0, 1)
    rad = np.clip(R * np.exp(rng.standard_normal(n) * P["size_var"]), 0.8, R * 3).astype(_F)
    step = max(1.0, R * 0.4)
    # a knife pulls one way from where it lands, each swipe its own length
    run = P["run"] * max(0.1, length_mul) * np.exp(rng.standard_normal(n) * P["len_var"])
    lim = np.maximum(1, (run * rad / step)).astype(np.int32)
    nmax = int(min(lim.max(), max(4, 26 * P["run"] * R / step / 4)))
    lim = np.minimum(lim, nmax)
    tol = (0.16 + 0.34 * (1 - fine) * (1 - 0.6 * detail)) * P["tol"]
    # vivid paint is the first thing a painter stops from smearing around
    if P["kind"] != "hero" and not petal:
        chroma = col.max(1) - col.min(1)
        lim = np.maximum(1, (lim * (1.1 - np.clip(chroma * 1.4, 0, 0.75))).astype(np.int32))
    spin = rng.standard_normal(n) * P["spin"] + np.pi * (rng.random(n) < 0.5)
    # a blade's mark is always longer than it is wide, even if it crosses an edge
    nmin = max(2, int(round(P["min_run"] * R / step)))
    lim = np.maximum(lim, nmin)
    nmax = max(nmax, nmin)
    path, nb, nf = _trace(sx, sy, col, ref, tx, ty, S.grid, step, nmax, nmin, tol, spin,
                          P["inertia"], lim)
    K = (path.shape[0] - 1) // 2
    ar = np.arange(n)
    far = path[K + nf, ar]
    col2 = _mix_paint(ref[np.clip(far[:, 1].astype(np.int64), 0, h - 1),
                          np.clip(far[:, 0].astype(np.int64), 0, w - 1)], P, rng, vary)
    # the third paint on the blade: a lighter or darker scrap of the same mix
    col3 = np.clip(col * (1 + rng.standard_normal((n, 1)).astype(_F) * _F(0.16)
                          + rng.standard_normal((n, 3)).astype(_F) * _F(0.05)) + _F(0.02), 0, 1)
    shape = (_spread(rng, n, P["skew"]), _spread(rng, n, P["pinch"]),
             (rng.standard_normal(n) * P["tilt"]).astype(_F),
             (rng.random(n) * P["wave"]).astype(_F), _spread(rng, n, P["rag"]),
             _spread(rng, n, P["flare"]), _spread(rng, n, P["rough"]), petal)
    top = np.full((h, w), -1, np.int64)
    total = _knife_rasterize(top, path, nb, nf, step, rad, S.tile, shape)
    idx, sid, u, v = _unpack(top)
    if len(idx) == 0:
        return

    # --- what the blade leaves behind ---------------------------------------
    off = rng.integers(0, 1 << 16, (n, 2))
    r = rad[sid]
    tt = np.clip(u / total[sid], 0, 1)
    hair = P["hair"] * S.scale
    fine_b = _sample_tile(S, off, sid, u, v, r, hair)
    b2 = _sample_tile(S, off, sid, u, v, r, hair * 3, coarse=3)
    if petal:
        # one clean press: creamy and opaque, a little lighter on the side the
        # blade leaned on, with a raised rim all the way round
        rel = np.clip(np.abs(v) / np.maximum(_petal(tt), _F(0.05)), 0, 1)
        hand = np.where(sid % 2 == 0, _F(1), _F(-1))
        tone = v * hand * _F(0.7) + (_F(0.45) - tt) * _F(0.6)
        swirl = np.clip((b2 - _F(0.5)) * _F(2.5) + _F(0.5), 0, 1) * _F(P["marble"])
        pix = col[sid] + (col2[sid] - col[sid]) * swirl[:, None]
        pix = pix * (1 + tone * _F(P["tone"]))[:, None]
        rim = np.exp(-(1 - rel) / _F(0.16))
        hgt = _F(P["thick"]) * (_F(0.8) + _F(0.25) * (1 - rel * rel)
                                + rim * _F(P["ridge"]) + (fine_b - _F(0.5)) * _F(0.12)
                                + np.exp(-(1 - tt) / _F(0.1)) * _F(P["lip"]))
        S.C.reshape(-1, 3)[idx] = pix
        S.H.ravel()[idx] = hgt
        return
    tooth = S.tooth.ravel()[idx]
    Cf = S.C.reshape(-1, 3)
    Hf = S.H.ravel()
    under = Cf[idx]
    Hn = Hf[idx]
    relief = _box(S.H, max(2, int(3 * S.scale))).ravel()[idx]
    high = np.clip(_F(0.5) + (Hn - relief) * _F(1.6), 0, 1)     # peaks of old paint

    # the load: fresh, then gone at its own point along the swipe. A short
    # touch never runs dry; only long pulls do
    out = _spread(rng, n, P["out"]) * _F(1 + 0.9 * fine ** 2)
    rem = np.clip(1 - tt / out[sid], 0, 1)
    grit = S.grit.ravel()[idx]
    # as it thins, paint streaks along the swipe and catches on old ridges; it
    # is there or it isn't, never see-through
    stick = (high * _F(0.3) + grit * _F(0.12) + b2 * _F(0.23) + fine_b * _F(0.35)) * _F(P["stick"])
    a = np.clip((rem * _F(2.0) + stick - _F(0.72)) / _F(0.07), 0, 1)
    if P["respect"]:
        # a painter stops a swipe at the shape's edge: paint won't sit on
        # what it doesn't belong to
        off_by = np.abs(ref.reshape(-1, 3)[idx] - col[sid]).sum(1)
        a = a * np.clip((_F(P["respect"]) - off_by) / _F(0.12), 0, 1)

    # unmixed paints ride the blade in ribbons, running along the swipe
    if P["kind"] == "scratch":
        pix = S.under.reshape(-1, 3)[idx]
    else:
        rib = np.clip((b2 - _F(0.5)) * _F(P["ribbon"]) + _F(0.5), 0, 1) * _F(P["marble"])
        b3 = _sample_tile(S, off[:, ::-1], sid, u, v, r, hair * 2)
        rib2 = np.clip((b3 - _F(0.5)) * _F(P["ribbon"] * 1.5) + _F(0.25), 0, 1) * _F(P["marble"])
        pix = col[sid] + (col2[sid] - col[sid]) * rib[:, None]
        pix += (col3[sid] - pix) * rib2[:, None]
        pix += (under - pix) * (_F(P["wet"]) * (1 - rem))[:, None]

    # thickness: a lump at the landing, thinning as the load goes, a raised
    # lip where the blade lifted, and squeezed-out edges, heavier on one side
    bias = np.where(rng.random(n) < 0.5, _F(1), _F(-1))[sid] * _F(P["ridge_bias"])
    aside = np.exp(-(1 - np.abs(v)) / _F(0.09)) * (1 + bias * np.sign(v))
    land = np.exp(-tt / _F(0.07))
    lift = np.exp(-(1 - tt) / _F(0.05))
    # the blade skips now and then, in patches, never as an even ripple
    th_, tw_ = S.tile.shape
    ch = S.tile[(off[sid, 1] + sid) % th_, (u * _F(1.4 / S.scale)).astype(np.int64) % tw_]
    ch = (ch - _F(0.5)) * _F(3) * np.clip((S.patch.ravel()[idx] - _F(0.5)) * _F(5), 0, 1) \
        * _F(P["chatter"])
    body = _F(0.34) + _F(0.66) * rem ** _F(1.3)
    hgt = _F(P["thick"]) * (body * (1 + ch + (fine_b - _F(0.5)) * _F(P["scrape"]))
                            + land * _F(P["lump"]) + lift * _F(P["lip"])
                            + aside * _F(P["ridge"]))
    if P["kind"] == "scratch":
        Cf[idx] = under + (pix - under) * a[:, None]
        Hf[idx] = Hn * (1 - a * _F(0.7))
        return
    Cf[idx] = under + (pix - under) * a[:, None]
    Hf[idx] = Hn * (1 - a * _F(P["flatten"])) + a * hgt


def _knife_paint(S, rec, base, lum, detail, length_mul, vary, say):
    """Run a knife recipe. Marks are chosen by what's under them: flat of the
    blade on broad areas, tip dabs on busy texture, edge lines on long lines."""
    h, w = S.h, S.w
    S.sharp = _smooth(base, 2 * S.scale)     # judges fit: blur would hide edges
    S.grit = _noise(S.rng, h, w, 3 * S.scale) * _F(0.6) + _noise(S.rng, h, w, 9 * S.scale) * _F(0.4)
    S.grit = np.clip((S.grit - _F(0.5)) * _F(2) + _F(0.5), 0, 1)
    S.patch = _noise(S.rng, h, w, 45 * S.scale)
    e = S.J[..., 0] + S.J[..., 2]
    j = _smooth(S.J, 3 * S.scale / S.grid)
    tr = j[..., 0] + j[..., 2] + _F(1e-9)
    coh = np.sqrt((j[..., 0] - j[..., 2]) ** 2 + 4 * j[..., 1] ** 2) / tr
    en = np.clip(np.sqrt(_smooth(e, 4 * S.scale / S.grid))
                 / max(float(np.percentile(np.sqrt(e), 90)), 1e-6), 0, 1)
    if S.grid > 1:
        coh, en = _grow(coh, S.grid, h, w), _grow(en, S.grid, h, w)
    busy = en * (1 - coh)
    line = en * coh
    # eyes read contrast as a ratio, so dark foliage counts as busy too
    rel_busy = busy / (_F(0.3) + _smooth(lum, 6 * S.scale))
    organic = np.clip((_smooth(rel_busy, 7 * S.scale) - _F(0.22)) / _F(0.2), 0, 1)
    steps = rec["steps"]
    for i, P in enumerate(steps):
        if P["kind"] == "snapshot":
            S.under = S.C.copy()
            continue
        if P["kind"] == "iron":
            # rims are soft rolls of paint; once, this irons out raster teeth
            S.H[:] = _box(S.H, 1) * _F(0.6) + S.H * _F(0.4)
            continue
        if P["kind"] == "blend":
            # melt the underpainting so the dabs sit on something soft
            S.C = np.ascontiguousarray(_smooth(S.C, P["R"]))
            S.H *= _F(0.3)
            continue
        ref = _smooth(base, P["R"] * 0.5)
        if i == 0:
            err, thresh = np.ones((h, w), _F), 0.0
        elif P["kind"] == "hero":
            # confident swipes laid late across calm areas, sparse and uneven
            calm = np.clip(1 - busy * _F(2.2), 0, 1)
            err = calm * (_F(0.45) + _F(0.55) * _noise(S.rng, h, w, P["R"] * 5))
            thresh = 0.5 * P["thresh"]
        elif P["kind"] == "petal":
            if P.get("accent"):
                err = np.maximum(ref @ _LUMA - _smooth(lum, P["R"] * 5), 0) * S.focus * organic
                thresh = 0.07 * P["thresh"]
            else:
                # dabs belong to things that grow; structures stay flat planes
                err = (_error(S, P, ref, lum) + _F(0.3)) * organic
                thresh = (0.2 - 0.12 * detail) * P["thresh"]
        elif P["kind"] == "define":
            # restate the drawing where the picture still misses, near the focus
            err = _error(S, P, ref, lum)
            err = err * (_F(0.25) + _F(0.75) * S.focus) * (_F(0.5) + en)
            if P.get("structure"):
                err = err * (1 - organic)
            thresh = (0.2 - 0.14 * detail) * P["thresh"]
        elif P["kind"] == "dabs":
            err, thresh = busy * S.focus, (0.3 - 0.2 * detail) * P["thresh"]
        elif P["kind"] in ("lines", "scratch"):
            err, thresh = line * S.focus, (0.3 - 0.2 * detail) * P["thresh"]
        else:
            err = _error(S, P, ref, lum)
            thresh = (0.42 - 0.32 * detail) * (0.6 + 0.4 * (1 - P["fine"])) * P["thresh"]
        _knife_marks(S, P, ref, err, thresh, detail, length_mul, vary)
        say((i + 1) / float(len(steps)))


def _hero(big, f=1.0, **kw):
    p = dict(kind="hero", run=5.0, spacing=2.4, spin=0.22, tol=0.55, thick=1.25,
             size_var=0.4, len_var=0.5, out=(0.45, 1.0), lump=1.1, lip=0.7, ridge=0.8,
             marble=0.85, min_run=3.0, jitter=0.03, wet=0.1, respect=0.3,
             rag=(0.9, 2.2), rough=(0.5, 1.1), flare=(0.2, 0.8), pinch=(0.15, 0.7))
    p.update(kw)
    return _new_knife(big * f, 0.45, **p)


def _define(small, sc, f=1.0, **kw):
    p = dict(kind="define", run=2.6, spacing=1.05, spin=0.1, thick=1.15, tol=0.7,
             out=(1.0, 1.4), lump=0.7, lip=0.35, ridge=0.35, marble=0.3, jitter=0.012,
             min_run=1.8, size_var=0.25, len_var=0.4, wet=0.0, respect=0.42,
             rough=(0.1, 0.5), rag=(0.3, 1.0))
    p.update(kw)
    return _new_knife(max(small * 1.5, 2.6 * sc) * f, 1.0, **p)


def _recipe_butter(big, small, sc):
    """Thick paint tapped on and spread like butter, every swipe its own."""
    sizes = _sizes(big * 1.5, max(small * 2.3, 3.8 * sc), 0.6)
    n = len(sizes)
    steps = []
    for i, R in enumerate(sizes):
        f = i / max(1, n - 1)
        steps.append(_new_knife(R, f, run=_lerp(5.0, 3.6, f), spacing=_lerp(1.4, 1.2, f),
                                spin=_lerp(0.36, 0.16, f), thick=_lerp(0.8, 1.2, f),
                                focus=_lerp(0.5, 1.0, f), thresh=_lerp(1.0, 0.9, f),
                                size_var=0.32, len_var=0.45, lump=1.2, ridge=0.7))
    steps.append(_hero(big, 1.15))
    steps.append(_define(small, sc))
    return dict(engine="knife", ground=(0.34, 0.25, 0.19), ground_mix=0.4, warp=1.2,
                mass=0.6, temp=0.6, sat=0.12, relief=1.6, gloss=0.16, weave=0.06, steps=steps)


def _recipe_slab(big, small, sc):
    """Big flat interlocking planes in short swipes, nothing fussed over."""
    sizes = _sizes(big * 2.2, max(big * 0.42, small * 3.0, 4.0 * sc), 0.62)
    n = len(sizes)
    steps = []
    for i, R in enumerate(sizes):
        f = i / max(1, n - 1)
        steps.append(_new_knife(R, f, run=_lerp(4.4, 3.4, f), spacing=_lerp(1.25, 1.1, f),
                                min_run=3.0,
                                spin=_lerp(0.5, 0.2, f), thick=_lerp(0.9, 1.15, f), tol=0.75,
                                thresh=_lerp(1.0, 0.8, f),
                                out=(0.65, 1.1), marble=0.3, chatter=0.03, scrape=0.12,
                                size_var=0.35, len_var=0.35, jitter=0.05, chroma=1.12,
                                skew=(0.0, 0.6), rag=(0.3, 1.0), pinch=(0.0, 0.3), lip=0.8))
    steps.append(_hero(big, 1.5, run=5.0, spacing=3.0, marble=0.35, spin=0.3))
    steps.append(_define(small, sc, 1.9, run=2.4, spin=0.2, thresh=1.3, marble=0.15))
    return dict(engine="knife", ground=(0.2, 0.17, 0.16), ground_mix=0.5, warp=2.0,
                mass=0.9, temp=0.7, sat=0.2, relief=1.5, gloss=0.14, weave=0.05, steps=steps)


def _recipe_sgraffito(big, small, sc):
    """All the parts of the knife: flat planes, tip dabs, edge lines, and
    lines scratched back through the wet paint."""
    sizes = _sizes(big * 1.4, max(small * 2.7, 4.6 * sc), 0.6)
    n = len(sizes)
    steps = []
    for i, R in enumerate(sizes):
        f = i / max(1, n - 1)
        steps.append(_new_knife(R, f, run=_lerp(4.6, 3.4, f), spacing=_lerp(1.35, 1.2, f),
                                spin=_lerp(0.34, 0.16, f), thick=_lerp(0.7, 1.05, f),
                                thresh=_lerp(1.0, 0.9, f)))
        if i == 0:
            steps.append(dict(kind="snapshot"))
    tip = max(small * 1.5, 2.6 * sc)
    edge = max(small * 0.8, 1.1 * sc)
    steps.append(_new_knife(tip, 1.0, kind="dabs", run=0.9, spacing=1.15, spin=0.9,
                            thick=1.3, out=(1.2, 2.0), lump=0.9, lip=0.3, ridge=0.2,
                            size_var=0.4, len_var=0.5, tol=2.0, marble=0.4))
    steps.append(_hero(big, 1.1))
    steps.append(_define(small, sc))
    steps.append(_new_knife(edge, 1.0, kind="lines", run=16, spacing=9.0, spin=0.03,
                            inertia=0.96, thick=1.1, tol=1.2, out=(0.7, 1.3), lump=0.2,
                            thresh=2.2,
                            lip=0.2, ridge=0.0, skew=(0.0, 0.2), rag=(0.2, 0.6),
                            size_var=0.25, len_var=0.5, light=0.08, marble=0.2))
    steps.append(_new_knife(edge * 0.9, 1.0, kind="scratch", run=14, spacing=18.0, spin=0.06,
                            inertia=0.97, tol=9.0, out=(0.8, 1.4), thresh=2.6, size_var=0.2,
                            len_var=0.5, skew=(0.0, 0.2), rag=(0.2, 0.6)))
    return dict(engine="knife", ground=(0.45, 0.24, 0.14), ground_mix=0.55, warp=1.2,
                mass=0.5, temp=0.6, sat=0.12, relief=1.5, gloss=0.14, weave=0.06, steps=steps)


def _recipe_petal(big, small, sc):
    """Vivid thick dabs on a soft blended ground: each press a clean rounded
    shape with a raised rim, long flat sweeps across calm areas, and thin dark
    lines cut in with the blade's edge."""
    steps = [_new_knife(big * 1.6, 0.0, run=4.0, spacing=1.2, out=(2.0, 3.0), thick=0.3,
                        marble=0.5, stick=1.2, min_run=2.5, rag=(0.2, 0.6)),
             _new_knife(big * 0.9, 0.3, run=4.0, spacing=1.2, out=(2.0, 3.0), thick=0.3,
                        marble=0.5, stick=1.2, min_run=2.5, rag=(0.2, 0.6)),
             dict(kind="blend", R=big * 0.35),
             _hero(big, 0.75, run=9.0, spacing=1.7, spin=0.1, thick=0.9, out=(0.9, 1.6),
                   respect=0.34, marble=0.7, rag=(0.3, 1.0), rough=(0.1, 0.4), level=0.85),
             _new_knife(big * 0.42, 0.6, run=5.0, spacing=1.2, spin=0.12, out=(1.0, 1.5),
                        thick=0.95, respect=0.4, marble=0.5, tol=0.8, rough=(0.1, 0.4),
                        rag=(0.3, 0.9), level=0.7, min_run=2.6),
             _define(small, sc, 1.1, run=3.6, spin=0.06, marble=0.2, thresh=1.1, structure=True)]
    sizes = _sizes(big * 1.15, max(small * 2.0, 4.2 * sc), 0.62)
    n = len(sizes)
    for i, R in enumerate(sizes):
        f = i / max(1, n - 1)
        steps.append(_new_knife(R, f, kind="petal", run=2.1, min_run=1.5, spacing=_lerp(1.35, 1.15, f),
                                spin=0.75, inertia=0.86, tol=9.0, size_var=0.28, len_var=0.25,
                                thick=_lerp(1.0, 1.25, f), ridge=0.55, lip=0.35, marble=0.55,
                                jitter=0.05, chroma=1.12, tone=0.22, fit=0.42))
    steps.append(dict(kind="iron"))
    steps.append(_new_knife(max(small * 0.9, 1.3 * sc), 1.0, kind="lines", run=16, spacing=16.0,
                            spin=0.03, inertia=0.95, thick=1.0, tol=1.3, out=(1.2, 1.8), lump=0.2,
                            lip=0.2, ridge=0.1, thresh=3.0, light=-0.1, marble=0.1,
                            size_var=0.3, len_var=0.5, rag=(0.2, 0.6)))
    steps.append(_new_knife(sizes[max(0, n - 2)], 1.0, kind="petal", accent=True, run=2.1,
                            min_run=1.5, spacing=2.4, spin=0.8, inertia=0.86, tol=9.0,
                            thick=1.5, ridge=0.6, lip=0.4, marble=0.3, light=0.1, tone=0.15,
                            jitter=0.03, chroma=1.1, size_var=0.3, len_var=0.25))
    return dict(engine="knife", ground=(0.16, 0.14, 0.15), ground_mix=0.3, warp=1.2, mass=0.75,
                temp=0.5, sat=0.75, punch=0.3, glow=0.72, deep=0.6, relief=1.5, gloss=0.3, weave=0.0,
                steps=steps)


_RECIPES = {
    "dab": _recipe_petal,
    "butter": _recipe_butter,
    "slab": _recipe_slab,
    "sgraffito": _recipe_sgraffito,
}
STYLE_NAMES = list(_RECIPES)


def stylize(rgb, style="dab", brush=6, detail=0.6, stroke_length=1.0,
            color_variation=0.35, texture=0.5, canvas=0.5, palette="none",
            palette_strength=0.3, vibrance=0.2, contrast=0.08, seed=1,
            progress=None):
    """rgb: HxWx3 uint8 photo -> HxWx3 uint8 painting.

    style            one of STYLE_NAMES
    brush            largest brush size (relative to image size)
    detail           0..1, how fine the last strokes get and how much of the
                     picture's small structure they chase
    stroke_length    multiplier on how far strokes run
    color_variation  0..1, how much each stroke's color strays from the photo
    texture          0..1, how strongly the paint surface catches the light
    canvas           0..1, how much of the canvas or paper shows
    palette          'none' or a PALETTES key, applied as a split-tone tint
    palette_strength 0..1
    seed             different seed = different stroke placement
    """
    say = progress or (lambda f: None)
    detail = float(np.clip(detail, 0, 1))
    img = rgb.astype(_F) * _F(1 / 255.0)
    h, w, _ = img.shape
    scale = max(h, w) / 1600.0
    big = max(3.0, brush * 2.5 * scale)
    small = max(1.15, (3.2 - 2.2 * detail) * scale)
    vary = color_variation / 0.35
    S = _Surface(img, seed, scale)
    rec = _RECIPES[style](big, small, scale)

    # direction fields are smooth, so they live on a half-size grid
    J = _tensor(img @ _LUMA, max(1.0, 1.5 * scale))
    S.J = _shrink(J, S.grid) if S.grid > 1 else J
    S.coarse = _smooth(S.J, big * 3 / S.grid)
    S.wobble = (_noise(S.rng, S.J.shape[0], S.J.shape[1], big * 4 / S.grid) - _F(0.5)) * _F(0.6)
    S.focus = _focus_map(S, img)
    base = _see(img, S, rec["warp"], rec["mass"], rec["temp"], rec["sat"])
    lum = base @ _LUMA
    if rec.get("punch"):
        # deep darks against glowing lights, with the middle tones lifted first
        lift = lum ** _F(rec.get("glow", 1.0))
        s_curve = lift * lift * (3 - 2 * lift)
        l2 = lift + (s_curve - lift) * _F(rec["punch"] * 2)
        base = np.clip(base * (np.clip(l2, 0.01, 1) / np.maximum(lum, _F(0.01)))[..., None], 0, 1)
        lum = base @ _LUMA
        # no dead blacks: shadows take a deep version of the color around them
        soft = _smooth(base, 45 * scale)
        sl = (soft @ _LUMA)[..., None]
        deep = sl + (soft - sl) * _F(2.4)
        deep = np.clip(deep / np.maximum(deep.max(-1, keepdims=True), _F(0.05)), 0, 1) * _F(0.36)
        # shadows are never one flat color: their depth and temperature drift
        drift = _noise(S.rng, h, w, 26 * scale)[..., None]
        deep = deep * (_F(0.6) + _F(0.8) * drift) \
            * (1 + (drift - _F(0.5)) * np.array([-0.5, 0.0, 0.9], _F))
        w_ = np.clip((_F(0.2) - lum) / _F(0.2), 0, 1)[..., None] * _F(rec["deep"])
        base = base + (deep - base) * w_
        lum = base @ _LUMA
    say(0.1)

    S.tooth = _weave(S.rng, h, w, max(3.0, 4.6 * scale))
    ground = np.array(rec["ground"], _F)
    S.C = _smooth(base, big * 2)
    # contiguous: passes write through a flat view of this array
    S.C = np.ascontiguousarray(S.C + (ground - S.C) * _F(rec["ground_mix"]))
    S.under = S.C.copy()
    _knife_paint(S, rec, base, lum, detail, stroke_length, vary,
                 lambda f: say(0.1 + 0.75 * f))

    # --- the finished surface ----------------------------------------------
    # thin paint lets the weave through; thick paint buries it
    thin = np.exp(-np.maximum(S.H, 0) * _F(2.4))
    S.H = S.H + (S.tooth - _F(0.5)) * thin * _F(rec["weave"] * canvas * 2)
    S.C = _box(S.C, 1) * _F(0.3) + S.C * _F(0.7)              # soften pixel stairs
    S.H = _box(S.H, 1) * _F(0.5) + S.H * _F(0.5)
    say(0.9)
    relief = rec["relief"] * (texture / 0.5) * scale ** 0.5
    out = _light(S, relief, rec["gloss"] * (texture / 0.5))
    out = _grade(np.clip(out, 0, 1), contrast, vibrance, palette, palette_strength)
    say(1.0)
    return (np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8)


# ---------------------------------------------------------------- stamp frame

def frame_geometry(art_w, art_h, border_frac=0.07):
    """Sizes for the white perforated border around art of art_w x art_h."""
    b = max(6, int(round(min(art_w, art_h) * border_frac)))
    hole_r = b * 0.32
    spacing = b * 1.05
    return dict(border=b, hole_r=hole_r, spacing=spacing,
                width=art_w + 2 * b, height=art_h + 2 * b)


def perforation_mask(w, h, hole_r, spacing):
    """HxW float mask: 1 inside the stamp, 0 in the half-circle bites along
    every edge, anti-aliased. Only the edge strips are computed."""
    mask = np.ones((h, w))
    t = int(np.ceil(hole_r)) + 2
    nx = max(2, int(round((w - 1) / spacing)))
    ny = max(2, int(round((h - 1) / spacing)))
    xs = np.linspace(0, w - 1, nx + 1)
    ys = np.linspace(0, h - 1, ny + 1)

    def strip(y0, y1, x0, x1, centers):
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float64)
        d = np.full(yy.shape, np.inf)
        for cx, cy in centers:
            d = np.minimum(d, np.hypot(xx - cx, yy - cy))
        mask[y0:y1, x0:x1] = np.minimum(mask[y0:y1, x0:x1],
                                        np.clip(d - hole_r + 0.5, 0, 1))

    strip(0, t, 0, w, [(x, 0) for x in xs])
    strip(h - t, h, 0, w, [(x, h - 1) for x in xs])
    strip(0, h, 0, t, [(0, y) for y in ys] + [(x, y) for x in (0,) for y in (0, h - 1)])
    strip(0, h, w - t, w, [(w - 1, y) for y in ys] + [(w - 1, 0), (w - 1, h - 1)])
    return mask


def compose_stamp(art, border_frac=0.07, paper=(250, 248, 242), inner_line=True,
                  grain=2.5, seed=1, scrim=0.2):
    """Put painted art (HxWx3 uint8) on a perforated paper border.
    Returns (RGBA uint8 array, geometry dict)."""
    ah, aw, _ = art.shape
    g = frame_geometry(aw, ah, border_frac)
    W, H, b = g["width"], g["height"], g["border"]
    rgb = np.empty((H, W, 3), np.float64)
    rgb[:] = paper
    if grain:  # faint paper fibre texture, softened so it isn't pixel noise
        rng = np.random.default_rng(seed)
        n = _box(rng.standard_normal((H, W)).astype(_F), 1)
        rgb += (n * grain * 2.2)[..., None]
    art = art.astype(np.float64)
    if scrim > 0:  # darken the bottom of the art so the caption reads
        t = np.clip((np.arange(ah) / ah - 0.62) / 0.38, 0, 1)
        art = art * (1 - scrim * t * t * (3 - 2 * t))[:, None, None]
    rgb[b:b + ah, b:b + aw] = art
    if inner_line:  # thin darker hairline where the art meets the paper
        lw = max(1, b // 14)
        col = np.array(paper) * 0.82
        rgb[b - lw:b, b - lw:b + aw + lw] = col
        rgb[b + ah:b + ah + lw, b - lw:b + aw + lw] = col
        rgb[b - lw:b + ah + lw, b - lw:b] = col
        rgb[b - lw:b + ah + lw, b + aw:b + aw + lw] = col
    alpha = perforation_mask(W, H, g["hole_r"], g["spacing"]) * 255
    out = np.dstack([rgb, alpha])
    return (np.clip(out, 0, 255) + 0.5).astype(np.uint8), g
