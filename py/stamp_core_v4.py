"""Vintage travel-poster stamps for Postage Stamp V4, pure numpy (no scipy,
skimage or PIL), so the same code runs inside GIMP's Python and from the
command line.

Tiny printed stamps can't show brushwork, so this posterizes instead: the photo
is reduced to flat shapes in a bold palette (warm lights, cool teal shadows,
cream instead of white, deep ink instead of black), with crisp edges, a soft
light gradient inside each shape and fine print grain. A fine printer (600 DPI)
can hold a lot of small shapes, so Detail decides how much of the picture's
structure survives: at high Detail the shapes are found at full resolution.
"""
import numpy as np

_F = np.float32
_LUMA = np.array([0.299, 0.587, 0.114], _F)
INK = np.array([0.07, 0.12, 0.16], _F)
CREAM = np.array([0.96, 0.92, 0.80], _F)


# ---------------------------------------------------------------- filters

def _win_sum(a, r):
    """Sum over a 2r+1 window along axis 0, edge-padded."""
    c = np.pad(a, ((r + 1, r),) + ((0, 0),) * (a.ndim - 1), mode="edge")
    for i in range(1, len(c)):
        c[i] += c[i - 1]
    k = 2 * r + 1
    return c[k:] - c[:-k]


def _box(a, r):
    s = _win_sum(a, r)
    s = _win_sum(np.ascontiguousarray(s.swapaxes(0, 1)), r).swapaxes(0, 1)
    return s / _F((2 * r + 1) ** 2)


def _resize(a, h, w):
    """Bilinear resize to h x w (any factor)."""
    def taps(n, m):
        p = (np.arange(n, dtype=_F) + _F(0.5)) * _F(m / n) - _F(0.5)
        i = np.clip(np.floor(p).astype(np.int64), 0, m - 1)
        return i, np.minimum(i + 1, m - 1), np.clip(p - i, 0, 1).astype(_F)
    y0, y1, ty = taps(h, a.shape[0])
    x0, x1, tx = taps(w, a.shape[1])
    ty = ty.reshape((-1, 1) + (1,) * (a.ndim - 2))
    tx = tx.reshape((1, -1) + (1,) * (a.ndim - 2))
    rows = a[y0] * (1 - ty) + a[y1] * ty
    return rows[:, x0] * (1 - tx) + rows[:, x1] * tx


def _shrink(img, longest):
    """Area-average down so the longest side is about `longest`."""
    h, w = img.shape[:2]
    f = max(1, int(round(max(h, w) / float(longest))))
    if f == 1:
        return img.astype(_F)
    h2, w2 = (h // f) * f, (w // f) * f
    return img[:h2, :w2].reshape(h2 // f, f, w2 // f, f, -1).mean((1, 3), dtype=_F)


def _guided(img, r, eps):
    """Edge-preserving smoothing: a guided filter steered by brightness. Flat
    areas melt together; real edges stay put."""
    g = img @ _LUMA
    mg = _box(g, r)
    vg = _box(g * g, r) - mg * mg
    out = np.empty_like(img)
    for c in range(3):
        p = img[..., c]
        mp = _box(p, r)
        a = (_box(g * p, r) - mg * mp) / (vg + _F(eps))
        b = mp - a * mg
        out[..., c] = _box(a, r) * g + _box(b, r)
    return out


def _noise(rng, h, w, r):
    """Smooth noise, mean 0, std about 1, feature size ~r px."""
    f = max(1, int(r))
    n = rng.standard_normal((-(-h // f) + 2, -(-w // f) + 2)).astype(_F)
    if f > 1:
        n = _resize(_box(n, 1), (n.shape[0]) * f, (n.shape[1]) * f)[:h, :w]
    else:
        n = n[:h, :w]
    n -= n.mean()
    return n / max(float(n.std()), 1e-6)


# ---------------------------------------------------------------- shapes

def _features(x):
    """Opponent color space, so hue counts as much as brightness. Brightness
    is square-rooted, like the eye, so shadows get as many tones as lights."""
    L = np.sqrt(np.maximum(x @ _LUMA, 0))
    a = x[..., 0] - x[..., 1]
    b = (x[..., 0] + x[..., 1]) / 2 - x[..., 2]
    return np.stack([L * 1.25, a * 1.3, b * 1.3], -1)


def _kmeans(px, k, rng, iters=14):
    # vivid pixels are sampled more often, so a few flowers can win a color
    vivid = np.abs(px[:, 1]) + np.abs(px[:, 2])
    p = (_F(0.1) + vivid) / (_F(0.1) + vivid).sum(dtype=np.float64)
    sample = px[rng.choice(len(px), min(len(px), 30000), replace=False, p=p)]
    k = min(k, len(sample))
    # k-means++ start: spread the first guesses across the colors
    cent = [sample[rng.integers(len(sample))]]
    d = ((sample - cent[0]) ** 2).sum(1)
    for _ in range(1, k):
        cent.append(sample[rng.choice(len(sample), p=d / max(d.sum(), 1e-12))
                           if d.sum() > 0 else rng.integers(len(sample))])
        d = np.minimum(d, ((sample - cent[-1]) ** 2).sum(1))
    cent = np.array(cent, _F)
    for it in range(iters):
        if it == iters // 2:
            # swap the most crowded-together color for the worst-served pixels
            d = ((sample[:, None] - cent[None]) ** 2).sum(-1)
            near = d.min(1)
            cd = ((cent[:, None] - cent[None]) ** 2).sum(-1) + np.eye(len(cent)) * 1e9
            worst = sample[near.argmax()]
            if near.max() > cd.min() * 0.5:
                cent[np.unravel_index(cd.argmin(), cd.shape)[0]] = worst
        lab = ((sample[:, None] - cent[None]) ** 2).sum(-1).argmin(1)
        for j in range(k):
            sel = sample[lab == j]
            if len(sel):
                cent[j] = sel.mean(0)
    return cent


def _assign(px, cent):
    cc = (cent ** 2).sum(1)
    return (cc - 2 * (px @ cent.T)).argmin(1)


def _majority(lab, k, r, passes):
    """Each pixel takes the most common label around it: removes specks and
    rounds off ragged shapes."""
    for _ in range(passes):
        best = np.full(lab.shape, -1, _F)
        new = np.zeros_like(lab)
        for j in range(k):
            c = _box((lab == j).astype(_F), r)
            win = c > best
            best[win] = c[win]
            new[win] = j
        lab = new
    return lab


# ---------------------------------------------------------------- color

def _hsv(c):
    mx, mn = c.max(1), c.min(1)
    d = mx - mn
    h = np.zeros(len(c), _F)
    r, g, b = c[:, 0], c[:, 1], c[:, 2]
    nz = d > 1e-6
    i = nz & (mx == r)
    h[i] = ((g - b)[i] / d[i]) % 6
    i = nz & (mx == g) & (mx != r)
    h[i] = (b - r)[i] / d[i] + 2
    i = nz & (mx == b) & (mx != r) & (mx != g)
    h[i] = (r - g)[i] / d[i] + 4
    return h * _F(60), np.where(mx > 0, d / np.maximum(mx, 1e-6), 0).astype(_F), mx


def _rgb(h, s, v):
    h = (h % 360) / _F(60)
    i = np.floor(h).astype(int) % 6
    f = h - np.floor(h)
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    table = [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)]
    out = np.zeros((len(h), 3), _F)
    for k, (a, b, c) in enumerate(table):
        m = i == k
        out[m] = np.stack([a[m], b[m], c[m]], 1)
    return out


def _toward(h, target, pull):
    """Turn hue h toward target by `pull` (0..1), the short way round."""
    d = (target - h + 180) % 360 - 180
    return h + d * pull


def _poster_colors(c, warmth, saturation):
    """Turn the photo's average colors into a printer's palette: a few
    confident hues, warm lights against cool shadows, cream for white and
    deep ink for black."""
    L = c @ _LUMA
    h, s, v = _hsv(c)
    # the poster's hues: sky and water go teal-blue, foliage a deep blue
    # green, rock and sunlight a warm orange
    blue = (h > 175) & (h < 260)
    green = (h > 70) & (h <= 175)
    warm = (h <= 55) | (h >= 330)
    h = np.where(blue, _toward(h, 200, 0.45), h)
    h = np.where(green, _toward(h, 140, 0.35), h)
    h = np.where(warm & (L > 0.3), _toward(h, 28, 0.25), h)
    grey = np.clip((_F(0.2) - s) / _F(0.2), 0, 1)
    s = np.clip(s * _F(1 + 0.9 * saturation), 0, 0.92)
    out = _rgb(h, s, v)
    L2 = out @ _LUMA
    out = out * (L / np.maximum(L2, _F(1e-3)))[:, None]     # keep each value
    # greys pick up the light's temperature instead of staying dead: warm in
    # the light, cool in the shade
    temp = np.where(L > 0.55, 1, -1)[:, None] * np.array([0.05, 0.015, -0.06], _F)
    out = out + temp * (grey * _F(0.6 + 0.8 * saturation))[:, None]
    out = out + ((L - _F(0.5)) * _F(warmth))[:, None] * np.array([0.12, 0.04, -0.12], _F)
    w = np.clip((_F(0.18) - L) / _F(0.18), 0, 1)[:, None] * _F(0.4)
    out = out + (INK - out) * w
    w = np.clip((L - _F(0.8)) / _F(0.18), 0, 1)[:, None] * _F(0.7)
    out = out + (CREAM - out) * w
    return np.clip(out, 0.03, 0.98)


def _spread_values(img, amount):
    """Push lights and darks apart before simplifying, so shapes read at
    a glance."""
    L = img @ _LUMA
    s = L * L * (3 - 2 * L)
    L2 = L + (s - L) * _F(amount)
    return np.clip(img * (L2 / np.maximum(L, _F(1e-3)))[..., None], 0, 1)


# ---------------------------------------------------------------- main

def stylize(rgb, colors=20, detail=0.8, simplify=0.3, lines=0.5, warmth=0.5,
            saturation=0.5, shading=0.35, grain=0.5, seed=1, progress=None):
    """rgb: HxWx3 uint8 photo -> HxWx3 uint8 poster art, same size.

    colors      number of flat colors (4..32); fewer is bolder
    detail      0..1, how much small structure survives (leaves, bricks,
                windows); high values find shapes at full resolution
    simplify    0..1, how much smaller shapes get merged and smoothed
    lines       0..1, fine printed linework for detail too small to get its own
                color (mortar, leaf veins, struts)
    warmth      0..1, warm sunlit lights against cool shadows
    saturation  0..1, color boost
    shading     0..1, soft light gradient kept inside each shape
    grain       0..1, fine print grain
    seed        changes the palette guesses and the grain
    """
    say = progress or (lambda f: None)
    rng = np.random.default_rng(seed)
    h, w = rgb.shape[:2]
    img = rgb.astype(_F) * _F(1 / 255.0)
    detail = float(np.clip(detail, 0, 1))
    simplify = float(np.clip(simplify, 0, 1))
    colors = int(np.clip(colors, 2, 32))

    # --- find the shapes -----------------------------------------------------
    small = _shrink(img, 640 * 2 ** (2 * detail))
    gh, gw = small.shape[:2]
    z = max(gh, gw) / 1200.0                     # sizes below are for 1200 px
    small = _spread_values(small, 0.35 - 0.2 * detail)
    if detail > 0:
        # lift small structure so leaves and bricks earn their own colors
        rr = max(1, int(round(3 * z)))
        small = np.clip(small + (small - _box(_box(small, rr), rr)) * _F(0.7 * detail), 0, 1)
    r = max(1, int(round((1.5 + 5 * simplify) * z)))
    for _ in range(1 + int(round(2 * simplify))):
        small = np.clip(_guided(small, r, 0.0015 + 0.012 * simplify), 0, 1)
    say(0.25)
    f = _features(small).reshape(-1, 3)
    cent = _kmeans(f, colors, rng)
    k = len(cent)
    lab = _assign(f, cent).reshape(gh, gw)
    lab = _majority(lab, k, max(1, int(round((0.6 + 2.5 * simplify) * z))), 2)
    say(0.5)
    # each shape's color is the mean of the photo it covers
    base = np.zeros((k, 3), _F)
    count = np.bincount(lab.ravel(), minlength=k).astype(_F)
    for c in range(3):
        base[:, c] = np.bincount(lab.ravel(), small[..., c].ravel(), minlength=k) / np.maximum(count, 1)
    pal = _poster_colors(base, warmth, saturation)

    # --- redraw sharp at full size ------------------------------------------
    # each shape's mask is enlarged and softened, then the strongest wins:
    # edges come out as smooth curves instead of blocky pixels
    if (gh, gw) == (h, w):
        full = lab
    else:
        rb = max(1, int(round(min(h, w) / gh * 0.9)))
        best = np.full((h, w), -1, _F)
        full = np.zeros((h, w), np.int64)
        for j in range(k):
            if count[j] == 0:
                continue
            m = _box(_resize((lab == j).astype(_F), h, w), rb)
            win = m > best
            best[win] = m[win]
            full[win] = j
    say(0.75)
    out = pal[full]
    if shading > 0:
        # a soft light gradient within each shape, like a printed fade
        Ls = _resize(_box(small @ _LUMA, max(1, int(round((3 + 8 * simplify) * z)))), h, w)
        Lm = (base @ _LUMA)[full]
        out = out * (1 + (Ls - Lm) * _F(1.1 * shading))[..., None]
    if lines > 0:
        # thin dark and light lines from the photo itself, printed over the
        # flat colors like a key plate
        z_full = max(h, w) / 1200.0
        L = _box(img @ _LUMA, 1) if z_full >= 1.5 else img @ _LUMA
        around = _box(_box(L, max(1, int(round(2 * z_full)))), max(1, int(round(2 * z_full))))
        d = around - L
        t = _F(0.035 - 0.015 * lines)
        dark = np.clip((d - t) / _F(0.06), 0, 1) * _F(0.55 * lines)
        light = np.clip((-d - t) / _F(0.06), 0, 1) * _F(0.45 * lines)
        out = out * (1 - dark)[..., None]
        out = out + (CREAM - out) * light[..., None]
    # anti-alias only the shape boundaries
    edge = np.zeros((h, w), bool)
    edge[1:] |= full[1:] != full[:-1]
    edge[:-1] |= full[1:] != full[:-1]
    edge[:, 1:] |= full[:, 1:] != full[:, :-1]
    edge[:, :-1] |= full[:, 1:] != full[:, :-1]
    out = np.where(edge[..., None], _box(out, 1), out)
    if grain > 0:
        g = (rng.standard_normal((h, w)).astype(_F) * _F(0.045)
             + _noise(rng, h, w, 4) * _F(0.02)) * _F(grain)
        out = out * (1 + g)[..., None]
    say(1.0)
    return (np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8)


# ---------------------------------------------------------------- stamp frame

def perforation_mask(w, h, hole_r, spacing):
    """HxW mask: 1 inside the stamp, 0 in the round bites along every edge."""
    mask = np.ones((h, w), _F)
    t = int(np.ceil(hole_r)) + 2
    xs = np.linspace(0, w - 1, max(2, int(round((w - 1) / spacing))) + 1)
    ys = np.linspace(0, h - 1, max(2, int(round((h - 1) / spacing))) + 1)

    def strip(y0, y1, x0, x1, centers):
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(_F)
        d = np.full(yy.shape, np.inf, _F)
        for cx, cy in centers:
            d = np.minimum(d, np.hypot(xx - cx, yy - cy))
        mask[y0:y1, x0:x1] = np.minimum(mask[y0:y1, x0:x1], np.clip(d - hole_r + 0.5, 0, 1))

    strip(0, t, 0, w, [(x, 0) for x in xs])
    strip(h - t, h, 0, w, [(x, h - 1) for x in xs])
    strip(0, h, 0, t, [(0, y) for y in ys])
    strip(0, h, w - t, w, [(w - 1, y) for y in ys])
    return mask


def compose_stamp(art, border_frac=0.08, paper=(228, 223, 196)):
    """Put art (HxWx3 uint8) on a cream perforated border. Returns RGBA uint8."""
    ah, aw = art.shape[:2]
    b = max(6, int(round(min(aw, ah) * border_frac)))
    W, H = aw + 2 * b, ah + 2 * b
    rgb = np.empty((H, W, 3), _F)
    rgb[:] = paper
    rgb[b:b + ah, b:b + aw] = art
    alpha = perforation_mask(W, H, b * 0.36, b * 1.1) * 255
    return (np.clip(np.dstack([rgb, alpha]), 0, 255) + 0.5).astype(np.uint8)
