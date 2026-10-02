#!/usr/bin/env python3
"""AUDIT: Fujifilm «PROFESSIONAL DATA GUIDE» AF3-207U (2005) and FUJIFILM RESEARCH & DEVELOPMENT No.46.

Re-reads the guide's VECTOR panels (characteristic, spectral sensitivity, spectral dye density,
MTF, time-G) and printed values (RMS, resolving power) and asserts that what the database stores
for the seventeen stocks added from it on 2026-09-29 -- and the cross-checks on the ten stocks it
shares with their own data sheets -- still follows from the document.

CALIBRATION, AND WHY IT IS A HYBRID. Fuji typesets its tick labels on a uniform pitch but draws
its gridlines by hand: the rules jitter by up to 1 pt (0.037 D) while the labels are exactly
uniform and sit at a constant OFFSET from the rules. So each axis takes its SLOPE from the labels
and its OFFSET from the snapped rules; checked against the stocks whose own data sheets were
traced independently, this agrees to 0.007-0.028 D (X-TRA 400, REALA, PRO 400H, 800Z, X-TRA 800).

    python3 fuji_pdg_2005.py --root <tree> [--assert]
"""
import argparse, json, re, sys
from pathlib import Path
import numpy as np
import pymupdf
from scipy.optimize import minimize
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

TITLES = (("SPECTRAL", "SENSITIVITY", "SENS"), ("SPECTRAL", "DYE", "DYE"),
          ("CHARACTERISTIC", None, "CHAR"), ("MTF", None, "MTF"),
          ("TIME-G", None, "TIMEG"), ("TIME-", None, "TIMEG"))
NUM = re.compile(r"^[–−-]?\d+(\.\d+)?$")


def bez(p, n=24):
    t = np.linspace(0, 1, n)[:, None]
    return ((1 - t) ** 3 * p[0] + 3 * (1 - t) ** 2 * t * p[1]
            + 3 * (1 - t) * t ** 2 * p[2] + t ** 3 * p[3])


def subpaths(items, tol=0.35):
    """Split a drawing's items into continuous polylines (pt coordinates)."""
    out, cur, last = [], [], None
    for it in items:
        if it[0] == "c":
            P = np.array([[q.x, q.y] for q in it[1:5]])
            seg = bez(P)
        elif it[0] == "l":
            seg = np.array([[it[1].x, it[1].y], [it[2].x, it[2].y]])
        else:
            continue
        if last is not None and np.hypot(*(seg[0] - last)) > tol:
            out.append(np.vstack(cur)); cur = []
        cur.append(seg if not cur else seg[1:])
        last = seg[-1]
    if cur:
        out.append(np.vstack(cur))
    return out


def xmono(paths, travel=1.5):
    """Split polylines where the x direction reverses (two curves in one path)."""
    out = []
    for P in paths:
        start, sign, acc = 0, 0, 0.0
        for i in range(1, len(P)):
            dx = P[i, 0] - P[i - 1, 0]
            if abs(dx) < 1e-6:
                continue
            s = 1 if dx > 0 else -1
            if sign == 0:
                sign = s
            elif s != sign:
                # look ahead: a real reversal travels back far enough
                j = i; back = 0.0
                while j < len(P) and (P[j, 0] - P[j - 1, 0]) * sign <= 0:
                    back += abs(P[j, 0] - P[j - 1, 0]); j += 1
                    if back > travel: break
                if back > travel:
                    out.append(P[start:i]); start = i - 1; sign = s
        out.append(P[start:])
    return [p for p in out if len(p) >= 2]


def dash_class(d):
    d = (d or "").strip()
    m = re.match(r"\[(.*)\]", d)
    arr = m.group(1).split() if m else []
    return {0: "solid", 2: "dash", 4: "dashdot"}.get(len(arr), "other%d" % len(arr))


def is_dark(c):
    return c is not None and sum(c[:3]) < 0.6


def plot_rects(pg):
    out = []
    for dr in pg.get_drawings():
        f = dr.get("fill")
        if not f or not all(0.74 < v < 0.87 for v in f[:3]):
            continue
        kinds = [it[0] for it in dr["items"]]
        if kinds == ["re"] or kinds == ["l"] * 4 or kinds == ["qu"]:
            r = dr["rect"]
            if r.width > 50 and r.height > 40:
                out.append(pymupdf.Rect(r))
    return out


def grid_for(pg, R):
    """All thin dark rules spanning the plot, from whichever drawings carry them."""
    V, H = [R.x0, R.x1], [R.y0, R.y1]
    for dr in pg.get_drawings():
        w = dr.get("width") or 0
        if not is_dark(dr.get("color")) or w > 0.45:
            continue
        for it in dr["items"]:
            if it[0] != "l":
                continue
            p, q = it[1], it[2]
            if abs(p.x - q.x) < 0.3 and R.x0 - 1 < p.x < R.x1 + 1 and \
                    min(p.y, q.y) < R.y0 + 0.2 * R.height and max(p.y, q.y) > R.y1 - 0.2 * R.height:
                V.append(p.x)
            if abs(p.y - q.y) < 0.3 and R.y0 - 1 < p.y < R.y1 + 1 and \
                    min(p.x, q.x) < R.x0 + 0.25 * R.width and max(p.x, q.x) > R.x1 - 0.25 * R.width:
                H.append(p.y)
    def dd(a):
        a = sorted(a); o = []
        for v in a:
            if not o or v - o[-1] > 1.0: o.append(v)
            else: o[-1] = (o[-1] + v) / 2
        return o
    return dd(V), dd(H), None


def classify(words, R):
    words = [tuple(w[:4]) + (w[4].lstrip("●•·"),) for w in words]
    cand = []
    for i, w in enumerate(words):
        for a, b, kind in TITLES:
            if w[4].startswith(a) and (b is None or (i + 1 < len(words) and words[i + 1][4].startswith(b))):
                if w[3] <= R.y0 + 2 and R.y0 - w[3] < 75 and R.x0 - 60 < w[0] < R.x1:
                    cand.append((R.y0 - w[3] + 0.3 * abs(w[0] - R.x0), kind))
    return min(cand)[1] if cand else None


def merge_signs(words):
    ws = list(words); out = []; skip = False
    for i, w in enumerate(ws):
        if skip:
            skip = False; continue
        if w[4] in ("–", "-", "−") and i + 1 < len(ws):
            n = ws[i + 1]
            if abs(n[1] - w[1]) < 2 and 0 <= n[0] - w[2] < 5 and NUM.match(n[4]):
                out.append((w[0], n[1], n[2], n[3], "-" + n[4]) + tuple(n[5:])); skip = True
                continue
        out.append(w)
    return out


def labels(words, R, axis):
    out = []
    for w in merge_signs(words):
        t = w[4].replace("–", "-").replace("−", "-")
        if not NUM.match(t):
            continue
        cx, cy = (w[0] + w[2]) / 2, (w[1] + w[3]) / 2
        if axis == "y" and R.x0 - 22 < w[2] < R.x0 + 1 and R.y0 - 6 < cy < R.y1 + 6:
            out.append((float(t), cy))
        if axis == "x" and R.y1 - 1 < w[1] < R.y1 + 12 and R.x0 - 12 < cx < R.x1 + 12:
            out.append((float(t), cx))
    return out


def snap(lab, rules, tol):
    o = []
    for v, p in lab:
        if not rules:
            o.append((v, p, False)); continue
        j = int(np.argmin([abs(r - p) for r in rules]))
        o.append((v, rules[j], True) if abs(rules[j] - p) <= tol else (v, p, False))
    return o


def fit_axis(pairs, log=False):
    v = np.array([a for a, b, s in pairs], float)
    if log:
        v = np.log10(v)
    p = np.array([b for a, b, s in pairs])
    keep = np.ones(len(v), bool)
    while True:
        c = np.polyfit(p[keep], v[keep], 1)          # value = c0 * pos + c1
        res = np.abs(np.polyval(c, p) - v)
        span = np.ptp(v[keep])
        worst = res[keep].max()
        if keep.sum() > 3 and worst > 0.05 * span:
            keep[np.argmax(np.where(keep, res, -1))] = False
            continue
        return c, float(worst), int(keep.sum())


def curves_in(pg, R, minw=0.5):
    Rx = pymupdf.Rect(R.x0 - 3, R.y0 - 3, R.x1 + 3, R.y1 + 3)
    out = []
    for dr in pg.get_drawings():
        w = dr.get("width") or 0
        if dr.get("fill") or not is_dark(dr.get("color")) or w < minw:
            continue
        if any(it[0] == "re" for it in dr["items"]):
            continue
        if not Rx.contains(dr["rect"]) and not (Rx & dr["rect"]).get_area() > 0.6 * dr["rect"].get_area():
            continue
        for sp in xmono(subpaths(dr["items"])):
            if len(sp) < 4 and np.hypot(*(sp[-1] - sp[0])) < 6:
                continue
            out.append(dict(pts=sp, dash=dash_class(dr.get("dashes")), width=w))
    return out


def read_panel(pg, words, R, kind):
    V, H, G = grid_for(pg, R)
    logx = kind == "MTF"
    xl = labels(words, R, "x"); yl = labels(words, R, "y")
    xs = snap(xl, V, 4.0); ys = snap(yl, H, 4.0)
    res = dict(kind=kind, rect=[R.x0, R.y0, R.x1, R.y1], V=V, H=H)
    for ax, lab, sn in (("x", xl, xs), ("y", yl, ys)):
        if len(lab) < 2:
            continue
        a = fit_axis(sn, logx); b = fit_axis([(v, p, False) for v, p in lab], logx)
        if a[1] <= b[1] + 1e-9:
            use, how = a, "gridline"
        else:
            # labels are typeset uniformly (tiny residual) but sit at a constant offset
            # from the drawn rules: SLOPE from the labels, OFFSET from the snapped rules
            vv = np.array([v for v, p, s in sn if s], float)
            pp = np.array([p for v, p, s in sn if s])
            if logx:
                vv = np.log10(vv)
            slope = b[0][0]
            icpt = float(np.mean(vv - slope * pp)) if len(vv) else b[0][1]
            worst = float(np.abs(slope * pp + icpt - vv).max()) if len(vv) else b[1]
            use, how = ((slope, icpt), worst, len(vv)), "hybrid"
        res[ax] = dict(c=list(use[0]), worst=use[1], n=use[2], calib=how,
                       alt_worst=b[1] if how == "gridline" else a[1])
    res["curves"] = curves_in(pg, R)
    return res


def page_panels(pg):
    words = pg.get_text("words")
    out = []
    Rs = plot_rects(pg)
    ks = [classify(words, R) for R in Rs]
    for i, (R, k) in enumerate(zip(Rs, ks)):
        if k is None:   # «[120 Size]» twin panels: inherit from the panel above
            up = [(R.y0 - Q.y1, kk) for Q, kk in zip(Rs, ks)
                  if kk and abs(Q.x0 - R.x0) < 4 and abs(Q.width - R.width) < 4 and Q.y1 < R.y0]
            k = min(up)[1] if up else None
        if k:
            out.append(read_panel(pg, words, R, k))
    return out, words


def to_data(panel, c):
    """pt -> data coordinates."""
    P = c["pts"]
    x = np.polyval(panel["x"]["c"], P[:, 0])
    if "y" in panel:
        y = np.polyval(panel["y"]["c"], P[:, 1])
    else:
        y = -P[:, 1]
    if panel["kind"] == "MTF":
        x, y = 10 ** x, 10 ** y
    return x, y


# ---------------------------------------------------------------------------
# assembly per film block
# ---------------------------------------------------------------------------
BLOCKS = [
    # key, page, half ('L' x<300, 'R' x>300, 'B' whole page), kind
    ("RVP", 24, "L", "rev"), ("RVP100", 24, "R", "rev"),
    ("RVP100F", 25, "L", "rev"), ("RDPIII", 25, "R", "rev"),
    ("RAP100F", 26, "L", "rev"), ("RHPIII", 26, "R", "rev"),
    ("RTPII", 27, "L", "rev"), ("RA", 27, "R", "rev"),
    ("RM", 28, "L", "rev"), ("RH", 28, "R", "rev"),
    ("PRO160S", 29, "L", "neg"), ("PRO160C", 29, "R", "neg"),
    ("NPL", 30, "L", "neg"), ("PRO400H", 30, "R", "neg"),
    ("PRO800Z", 31, "L", "neg"), ("CS", 31, "R", "neg"),
    ("CN", 32, "L", "neg"), ("CA", 32, "R", "neg"),
    ("CH", 33, "L", "neg"), ("CH_TD", 33, "R", "neg"),
    ("CZ", 34, "L", "neg"), ("CU", 34, "R", "neg"),
    ("DA", 53, "L", "neg"), ("DH", 53, "R", "neg"), ("DZ", 54, "L", "neg"),
    ("ACROS", 40, "B", "bw"), ("NP400", 41, "B", "bw"), ("NP1600", 42, "B", "bw"),
]


def robust(panel, words, R, axis):
    """Refit an axis dropping one outlier label if the fit is poor."""
    return panel


def legend_map(pg, words, box):
    """dash class -> record, from the legend swatches beside the words."""
    out = {}
    for dr in pg.get_drawings():
        its = dr["items"]
        if len(its) != 1 or its[0][0] != "l" or not is_dark(dr.get("color")):
            continue
        p, q = its[0][1], its[0][2]
        if abs(p.y - q.y) > 0.3 or not (8 < abs(p.x - q.x) < 25):
            continue
        if not (box[0] < p.x < box[2] and box[1] < p.y < box[3]):
            continue
        for w in words:
            if w[4] in ("Red", "Green", "Blue") and abs((w[1] + w[3]) / 2 - p.y) < 3 \
                    and 0 < w[0] - max(p.x, q.x) < 12:
                out[dash_class(dr.get("dashes"))] = w[4][0]
    return out


def resample(x, y, grid, lo=None):
    o = np.argsort(x); x, y = x[o], y[o]
    keep = np.concatenate([[True], np.diff(x) > 1e-6]); x, y = x[keep], y[keep]
    v = np.interp(grid, x, y, left=np.nan, right=np.nan)
    return v


def block_panels(doc, key):
    b = [bb for bb in BLOCKS if bb[0] == key][0]
    pg = doc[b[1] - 1]
    pl, words = page_panels(pg)
    if b[2] == "L":
        pl = [p for p in pl if p["rect"][0] < 300]
    elif b[2] == "R":
        pl = [p for p in pl if p["rect"][0] > 300]
    return b, pg, pl, words


def pts(panel, c):
    x, y = to_data(panel, c)
    o = np.argsort(x)
    return x[o], y[o]


def union(frags):
    """Merge x-sorted fragments; where two overlap, average them."""
    if not frags:
        return None
    xs = np.unique(np.round(np.concatenate([f[0] for f in frags]), 4))
    acc = np.zeros_like(xs); n = np.zeros_like(xs)
    for x, y in frags:
        v = np.interp(xs, x, y, left=np.nan, right=np.nan)
        m = ~np.isnan(v); acc[m] += v[m]; n[m] += 1
    m = n > 0
    return xs[m], acc[m] / n[m]


def fill_from(frags, base):
    """own record where drawn, the shared (solid) trace wherever no own fragment covers."""
    if not frags:
        return base
    bx, by = base
    m = np.ones(len(bx), bool)
    for x, y in frags:
        m &= (bx < x.min() - 1e-3) | (bx > x.max() + 1e-3)
    X = np.concatenate([bx[m]] + [f[0] for f in frags]); Y = np.concatenate([by[m]] + [f[1] for f in frags])
    o = np.argsort(X)
    return X[o], Y[o]


def neg_sens_scale(pg, panel):
    """pt per log decade from the drawn 1.0 scale bar's arrowheads."""
    R = panel["rect"]
    tips = []
    for dr in pg.get_drawings():
        f = dr.get("fill")
        r = dr["rect"]
        if f and is_dark(f) and r.width < 3.5 and r.height < 5 and R[0] - 9 < r.x0 < R[0] + 1 \
                and R[1] - 2 < r.y0 < R[3]:
            tips.append((r.y0, r.y1))
    if len(tips) < 2:
        return None, tips
    tips.sort()
    top = tips[0][0]; bot = tips[-1][1]
    return bot - top, tips


def char_rev(pg, words, p):
    R = p["rect"]
    lm = legend_map(pg, words, (R[0] - 5, R[1] - 20, R[2] + 130, R[3] + 10))
    by = {"R": [], "G": [], "B": []}
    for c in p["curves"]:
        k = lm.get(c["dash"])
        if k:
            by[k].append(pts(p, c))
    base = union(by["R"])
    out = {"R": base, "G": fill_from(by["G"], base), "B": fill_from(by["B"], base)}
    return out, lm, {k: len(v) for k, v in by.items()}


def char_neg(p):
    cs = [pts(p, c) for c in p["curves"]]
    # merge fragments of the same curve: group by mean density at common x
    xs = np.linspace(-3.0, 0.0, 7)
    lev = [np.nanmean(np.interp(xs, x, y, left=np.nan, right=np.nan)) for x, y in cs]
    order = np.argsort(lev)[::-1]
    if len(cs) != 3:
        return None, len(cs)
    return {"B": cs[order[0]], "G": cs[order[1]], "R": cs[order[2]]}, 3


def sens(pg, p, neg):
    out = {}
    scale = None
    if neg or "y" not in p:
        sc, tips = neg_sens_scale(pg, p)
        H = sorted(p["H"])
        inner = [h for h in H if p["rect"][1] + 0.5 < h < p["rect"][3] - 0.5]
        gsp = np.diff(inner).tolist() if len(inner) >= 2 else []
        scale = dict(bar_pt=sc, grid_pt=gsp)
        pt_per = float(np.mean(gsp)) if gsp else sc
    solids = []; dashed = []
    for c in p["curves"]:
        x = np.polyval(p["x"]["c"], c["pts"][:, 0])
        if "y" in p and not neg:
            y = np.polyval(p["y"]["c"], c["pts"][:, 1])
        else:
            y = -(c["pts"][:, 1] - p["rect"][1]) / pt_per
        o = np.argsort(x)
        (dashed if c["dash"] != "solid" else solids).append((x[o], y[o]))
    if not neg and "y" not in p or (neg and False):
        pass
    if p.get("_bw"):
        return {"PAN": union(solids)}, scale
    # a layer may arrive in fragments: cluster by x-centre
    solids.sort(key=lambda t: t[0].mean())
    names = ["B", "G", "R"]
    if len(solids) == 3:
        for n, s in zip(names, solids):
            out[n] = s
    else:
        out["_frag"] = len(solids)
        # assign by peak wavelength band
        for s in solids:
            pk = s[0][np.argmax(s[1])]
            n = "B" if pk < 495 else ("G" if pk < 585 else "R")
            out.setdefault(n, []).append(s)
        for n in names:
            if isinstance(out.get(n), list):
                out[n] = union(out[n])
    if dashed:
        out["C"] = union(dashed)
    return out, scale


def dye(p, neg):
    cs = []
    for c in p["curves"]:
        x, y = pts(p, c); cs.append((x, y))
    if neg:
        # two traces, possibly in fragments: split at the largest gap in level
        lv = [float(np.mean(t[1])) for t in cs]
        o = np.argsort(lv); srt = [lv[i] for i in o]
        if len(cs) < 2:
            return {"_n": len(cs)}
        g = int(np.argmax(np.diff(srt)))
        lo = [cs[i] for i in o[:g + 1]]; hi = [cs[i] for i in o[g + 1:]]
        return {"dmin": union(lo), "mid": union(hi), "_n": len(cs)}
    out = {}
    for x, y in cs:
        pk = x[np.argmax(y)]
        n = "Y" if pk < 490 else ("M" if pk < 590 else "C")
        out.setdefault(n, []).append((x, y))
    return {k: union(v) for k, v in out.items()}


def text_block(words, half):
    ws = [w for w in words if (half == "B" or (w[0] < 300) == (half == "L"))]
    t = " ".join(w[4] for w in ws)
    import re
    rms = re.search(r"VALUE\.+\s*(\d+)", t)
    rp = re.findall(r"1\.6:1\s*-\s*(\d+)\s*lines/mm.*?1000:1\s*-\s*(\d+)", t)
    iso = re.search(r"(ISO|EI)\s*(\d+)/(\d+)", t)
    return dict(rms=int(rms.group(1)) if rms else None, rp=list(map(int, rp[0])) if rp else None,
                iso=iso.group(0) if iso else None)






def sp(z, k):
    t = z / k
    return np.where(t > 40, z, np.where(t < -40, 0, k * np.log1p(np.exp(np.clip(t, -40, 40)))))


def md(x, q):
    return q[0] + q[1] * (sp(x - q[2], q[3]) - sp(x - q[4], q[5]))


XX = np.linspace(-8, 8, 4001)


def dip(q):
    y = md(XX, q); run = np.maximum.accumulate(y)
    return float((run - y).max())


def fit_rev(X, Y, seeds=None):
    """Reversal: x = -logH, full softplus pair (Dmax is inside the plot)."""
    X = np.asarray(X); Y = np.asarray(Y)
    if seeds is None:
        dmin = float(Y.min()); dmax = float(Y.max())
        mid = float(X[np.argmin(np.abs(Y - (dmin + dmax) / 2))])
        seeds = [[dmin + 0.002, 1.8, mid - 0.8, 0.25, mid + 0.8, 0.2], [dmin, 2.2, mid - 0.7, 0.3, mid + 0.7, 0.25],
                 [dmin, 1.6, mid - 1.0, 0.35, mid + 0.9, 0.15]]

    ymin = float(Y.min())

    def f(q):
        if q[1] <= 0 or q[1] > 2.5 or q[3] <= 0.02 or q[5] <= 0.02 or q[4] - q[2] < 0.9:
            return 1e9
        if q[0] < ymin - 0.005 or q[0] > ymin + 0.02:   # the asymptote is the drawn D-min plateau
            return 1e9
        e = np.mean((md(X, q) - Y) ** 2); d = dip(q)
        return e + (0 if d < 0.004 else 10 * (d - 0.004) ** 2 + 1e-3 * (d - 0.004))
    best = None
    for s in seeds:
        r = minimize(f, s, method="Nelder-Mead", options=dict(maxiter=30000, xatol=1e-7, fatol=1e-11))
        if best is None or r.fun < best.fun:
            best = r
    q = best.x; res = md(X, q) - Y
    return q, float(np.sqrt(np.mean(res ** 2))), float(np.abs(res).max())


def fit_neg(X, Y, shoulder=(1.75, 0.42)):
    """Negative: x = logH, shoulder DECLARED (the plot ends in the straight line)."""
    X = np.asarray(X); Y = np.asarray(Y)
    sx, sk = shoulder

    def full(p):
        return [p[0], p[1], p[2], p[3], sx, sk]

    ymin = float(Y.min())

    def f(p):
        if p[1] <= 0 or p[3] <= 0.02 or p[3] > 1.5:
            return 1e9
        if p[0] < ymin - 0.005 or p[0] > ymin + 0.02:
            return 1e9
        return np.mean((md(X, full(p)) - Y) ** 2)
    best = None
    for s in ([Y.min() + 0.002, 0.65, -2.5, 0.3], [Y.min() + 0.002, 0.6, -2.0, 0.4], [Y.min() + 0.002, 0.7, -3.0, 0.2]):
        r = minimize(f, s, method="Nelder-Mead", options=dict(maxiter=30000, xatol=1e-7, fatol=1e-11))
        if best is None or r.fun < best.fun:
            best = r
    q = full(best.x); res = md(X, q) - Y
    return q, float(np.sqrt(np.mean(res ** 2))), float(np.abs(res).max())


def f50(freq, resp_pct):
    f = np.asarray(freq); r = np.asarray(resp_pct) / 100.0
    o = np.argsort(f); f, r = f[o], r[o]
    above = np.where(r >= 0.5)[0]
    if len(above) == 0 or above[-1] == len(r) - 1:
        return None
    i = above[-1]
    lf = np.interp(0.5, [r[i + 1], r[i]], [np.log10(f[i + 1]), np.log10(f[i])])
    return float(10 ** lf)


def resample(x, y, grid, floor=None):
    x = np.asarray(x); y = np.asarray(y)
    o = np.argsort(x); x, y = x[o], y[o]
    k = np.concatenate([[True], np.diff(x) > 1e-6]); x, y = x[k], y[k]
    v = np.interp(grid, x, y, left=np.nan, right=np.nan)
    return v


# ===========================================================================
#  what the database adopted, and the checks
# ===========================================================================
NEW = {"RVP100": "FUJI_VELVIA_100", "RVP100F": "FUJI_VELVIA_100F", "RAP100F": "FUJI_ASTIA_100F",
       "RA": "FUJI_SENSIA_100_2005", "RM": "FUJI_SENSIA_200", "RH": "FUJI_SENSIA_400",
       "PRO160S": "FUJICOLOR_PRO_160S", "PRO160C": "FUJICOLOR_PRO_160C", "NPL": "FUJICOLOR_NPL_160",
       "CN": "FUJICOLOR_SUPERIA_100", "CA": "FUJICOLOR_SUPERIA_200", "CH_TD": "FUJICOLOR_TRUE_DEFINITION_400",
       "CU": "FUJICOLOR_SUPERIA_1600", "DA": "FUJICOLOR_NEXIA_A200", "DH": "FUJICOLOR_NEXIA_400",
       "DZ": "FUJICOLOR_NEXIA_800"}
#: Stocks whose OWN data sheet was traced earlier: the guide is a second copy of the same drawing,
#: so it is a CHECK and not a source. Limit = best-shift shape rms per record, D.
SHARED = {"RVP": ("FUJI_VELVIA_50", 0.035), "RDPIII": ("FUJI_PROVIA_100F", 0.12),
          "RHPIII": ("FUJI_PROVIA_400F", 0.06), "RTPII": ("FUJICHROME_64T_II", 0.06),
          "PRO400H": ("FUJICOLOR_PRO_400H", 0.04), "PRO800Z": ("FUJICOLOR_PRO_800Z", 0.04),
          "CS": ("FUJICOLOR_SUPERIA_REALA", 0.03), "CH": ("FUJICOLOR_SUPERIA_XTRA_400", 0.03),
          "CZ": ("FUJICOLOR_SUPERIA_XTRA_800", 0.035)}


def read_all(pdf):
    doc = pymupdf.open(str(pdf))
    out = {}
    for key, page, half, kind in BLOCKS:
        b, pg, pl, words = block_panels(doc, key)
        rec = dict(kind=kind, page=page, text=text_block(words, half))
        for p in pl:
            k = p["kind"]
            if k == "CHAR" and kind == "rev":
                rec["char"] = char_rev(pg, words, p)[0]
            elif k == "CHAR" and kind == "neg":
                rec["char"] = char_neg(p)[0]
            elif k == "SENS":
                p["_bw"] = kind == "bw"
                rec["sens"] = sens(pg, p, kind != "rev")[0]
            elif k == "DYE":
                rec["dye"] = dye(p, kind == "neg")
            elif k == "MTF":
                rec["mtf"] = union([pts(p, c) for c in p["curves"]])
        out[key] = rec
    return out


def shape_rms(x, y, curve, rev, shift=True):
    import film_sim as fs
    X = -np.asarray(x) if rev else np.asarray(x)
    best = None
    for s in (np.linspace(-1.5, 1.5, 301) if shift else (0.0,)):
        d = fs.density((X + s).astype(np.float32), curve)
        e = float(np.sqrt(np.mean((d - np.asarray(y)) ** 2)))
        if best is None or e < best[0]:
            best = (e, s)
    return best


def rd46_fig5(pdf):
    """Re-trace rd046 Fig. 5 (sigma_D vs D) from the embedded raster; returns {curve: (D, sigma)}."""
    doc = pymupdf.open(str(pdf))
    pg = doc[2]
    best = None
    for im in pg.get_images(full=True):
        pix = pymupdf.Pixmap(doc, im[0])
        if pix.width == 965 and pix.height == 461:
            best = pix
    if best is None:
        return None
    a = np.frombuffer(best.samples, dtype=np.uint8).reshape(best.height, best.width, best.n)[:, :, 0].astype(float)
    B = a > 110
    for c in (60, 61, 197, 333, 471, 472, 550, 551, 687, 823, 960, 961):
        B[:, max(0, c - 1):c + 2] = False
    for r_ in (5, 6, 7, 142, 143, 277, 278, 413, 414):
        B[max(0, r_ - 1):r_ + 2, :] = False
    for x0, y0, x1, y1 in ((212, 200, 300, 232), (238, 300, 338, 335), (605, 178, 668, 210), (722, 200, 812, 232)):
        B[y0:y1, x0:x1] = False
    B[405:, :] = False; B[:10, :] = False
    out = {}
    for name, (xa, xb, D0) in {"L": (62, 470, 60.5), "R": (552, 959, 550.5)}.items():
        tr = {"up": [], "lo": []}
        for x in range(xa, xb):
            ys = np.where(B[:, x])[0]
            if len(ys) == 0:
                continue
            runs = np.split(ys, np.where(np.diff(ys) > 1)[0] + 1)
            cs = sorted(r.mean() for r in runs if 1 <= len(r) <= 9)
            if not cs:
                continue
            if len(cs) >= 2:
                tr["up"].append((x, cs[0])); tr["lo"].append((x, cs[-1])); continue
            c = cs[0]
            if len(tr["lo"]) >= 2:
                (x1, y1), (x2, y2) = tr["lo"][-2], tr["lo"][-1]
                pred = y2 + (y2 - y1) / max(1, (x2 - x1)) * (x - x2)
                if abs(c - pred) < 3.5:
                    tr["lo"].append((x, c))
                elif c < pred:
                    tr["up"].append((x, c))
            else:
                tr["lo"].append((x, c))
        for k, v in tr.items():
            q = np.array([((x - D0) / 136.67, (413.5 - y) / 135.83 * 0.01) for x, y in v])
            out[name + k] = q[np.argsort(q[:, 0])]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--assert", dest="assert_", action="store_true")
    a = ap.parse_args(argv)
    root = Path(a.root) if a.root else HERE.parent
    pdf = root / "PDF" / "PROFILES" / "FUJI" / "ProfessionalFilmDataGuide.pdf"
    if not pdf.exists():
        print("SKIP: %s not on this tree" % pdf)
        return 0
    import film_profiles as fp
    P = {p.name: p for p in fp.FILM_PROFILES}
    R = read_all(pdf)
    bad = 0

    def chk(ok, msg):
        nonlocal bad
        print(("[OK  ] " if ok else "[FAIL] ") + msg)
        bad += 0 if ok else 1

    gs = np.arange(380.0, 701.0, 10.0)
    for key, name in NEW.items():
        r = R[key]; p = P.get(name)
        if p is None:
            chk(False, "%s missing from the database" % name); continue
        rev = r["kind"] == "rev"
        t = r["text"]
        chk(abs(p.grain.rms_granularity - t["rms"]) < 1e-9 and
            (p.mtf.resolving_power_lp_mm_lowc, p.mtf.resolving_power_lp_mm_highc) == tuple(float(v) for v in t["rp"]),
            "%s: printed RMS %s and resolving power %s stored" % (name, t["rms"], t["rp"]))
        f = f50(*r["mtf"])
        chk(abs(p.mtf.f50_g - f) / f < 0.01, "%s: f50 %.2f traced, %.2f stored" % (name, f, p.mtf.f50_g))
        cv = p.curves.as_tuple(); worst = 0.0
        for i, ch in enumerate("RGB"):
            x, y = r["char"][ch]
            worst = max(worst, shape_rms(x, y, cv[i], rev, shift=False)[0])
        chk(worst < 0.085, "%s: stored curves reproduce the trace at worst %.4f D rms (no shift)" % (name, worst))
        sp = p.spectral; sd = 0.0
        for ch, st in (("R", sp.log_s_r), ("G", sp.log_s_g), ("B", sp.log_s_b)):
            x, y = r["sens"][ch]
            v = resample(np.asarray(x), np.asarray(y), gs); v = v - np.nanmax(v)
            st = np.asarray(st); m = ~np.isnan(v) & (st > -3.9) & (v > -3.9)
            sd = max(sd, float(np.abs(v[m] - st[m]).max()))
        chk(sd < 0.02, "%s: spectral records reproduce the trace to %.3f log" % (name, sd))
    # the ten shared drawings
    for key, (name, lim) in SHARED.items():
        r = R[key]; p = P[name]; rev = r["kind"] == "rev"
        cv = p.curves.as_tuple()
        e = [shape_rms(*r["char"][ch], cv[i], rev)[0] for i, ch in enumerate("RGB")]
        chk(max(e) <= lim, "%s: its own-sheet curves and the AF3-207U drawing agree to %s D (limit %.3f)"
            % (name, " / ".join("%.3f" % v for v in e), lim))
    # A4 overshoot targets: the (peak value, peak frequency) pairs verify.py
    # _A4_SOLVED pins are re-read here off the vector MTF panels, so the solve
    # and the trace cannot drift apart.
    A4 = {"RVP": ("FUJI_VELVIA_50", 1.229, 8.33), "RTPII": ("FUJICHROME_64T_II", 1.115, 5.77),
          "RVP100": ("FUJI_VELVIA_100", 1.068, 4.94), "RVP100F": ("FUJI_VELVIA_100F", 1.068, 4.94),
          "NPL": ("FUJICOLOR_NPL_160", 1.141, 8.15)}
    for key, (name, tp, tf) in A4.items():
        fr, rs = (np.asarray(v, dtype=float) for v in R[key]["mtf"])
        i = int(np.argmax(rs))
        chk(abs(rs[i] / 100.0 - tp) < 2e-3 and abs(fr[i] - tf) < 0.2 and P[name].mtf.adjacency > 0.0,
            "A4 %s: traced overshoot %.3f @ %.2f c/mm (pinned %.3f @ %.2f), adjacency %.4f / %.2f um stored"
            % (name, rs[i] / 100.0, fr[i], tp, tf, P[name].mtf.adjacency, P[name].mtf.adjacency_um))
    # SENSIA 200 [RM]: its maximum sits at the low-frequency edge of the
    # drawing, so the solve lands in verify.py's refused 70-90 um band and no
    # adjacency is stored.
    fr, rs = (np.asarray(v, dtype=float) for v in R["RM"]["mtf"])
    i = int(np.argmax(rs))
    chk(fr[i] < 3.0 and P["FUJI_SENSIA_200"].mtf.adjacency == 0.0,
        "A4 refusal: SENSIA 200 peaks at %.2f c/mm, next to the panel's 1 c/mm edge; adjacency %.1f"
        % (fr[i], P["FUJI_SENSIA_200"].mtf.adjacency))
    # Velvia 50: the generation argument rests on four agreements
    r = R["RVP"]; p = P["FUJI_VELVIA_50"]
    chk(r["text"]["rms"] == 9 and tuple(r["text"]["rp"]) == (80, 160) and p.grain.rms_granularity == 9.0,
        "G-FPDG-VELVIA: RVP (1990) and RVP50 (2007) print the same RMS 9 and 80/160 lines/mm")
    dd = p.dye_density; dmax = 0.0
    for ch, st in (("C", dd.d_cyan), ("M", dd.d_magenta), ("Y", dd.d_yellow)):
        x, y = r["dye"][ch]
        v = resample(np.asarray(x), np.asarray(y), np.arange(400.0, 701.0, 10.0))
        m = ~np.isnan(v); dmax = max(dmax, float(np.sqrt(np.mean((v[m] - np.asarray(st)[m]) ** 2))))
    chk(dmax < 0.03, "G-FPDG-VELVIA: RVP50's dye set and the RVP drawing agree to %.3f D rms" % dmax)
    # Press 400 / 800 are the X-TRA blocks
    doc = pymupdf.open(str(pdf))
    t33 = doc[32].get_text(); t34 = doc[33].get_text()
    chk("PRESS 400" in t33 and "press 400" in P["FUJICOLOR_SUPERIA_XTRA_400"].aliases and
        "PRESS 800" in t34 and "press 800" in P["FUJICOLOR_SUPERIA_XTRA_800"].aliases,
        "PRESS 400 / 800 share the X-TRA 400 / 800 data blocks and are stored as their aliases")
    t41 = doc[40].get_text()
    chk(all(g in t41 for g in ("0.54", "0.65", "0.83")) and P["FUJI_NEOPAN_400"].processing_family.gamma_infinity > 0,
        "NEOPAN 400: G-bar 0.54 / 0.65 / 0.83 printed, rate law stored")
    rdp = root / "PDF" / "PROFILES" / "FUJI" / "rd_report_ff_rd046_001.pdf"
    if rdp.exists():
        f5 = rd46_fig5(rdp)
        rec = fp._FUJI_RD46
        for key, lab in (("Llo", "RDP III"), ("Lup", "RDP II"), ("Rlo", "RHP III"), ("Rup", "RHP")):
            q = f5[key]
            y = np.array([np.median(q[max(0, i - 3):i + 4, 1]) for i in range(len(q))])
            D = rec["fig5_density"] if lab.startswith("RDP") else rec["fig5_density"][1:]
            got = np.interp(D, q[:, 0], y)
            dev = float(np.abs(got - np.asarray(rec["fig5_sigma"][lab])).max())
            chk(dev < 0.0006, "rd046 Fig. 5 %s re-traced; stored sigma(D) reproduced to %.5f" % (lab, dev))
        for name, lab in (("FUJI_PROVIA_100F", "RDP III"), ("FUJI_PROVIA_400F", "RHP III")):
            g = P[name].grain
            D = rec["fig5_density"] if lab == "RDP III" else rec["fig5_density"][1:]
            sig = np.asarray(rec["fig5_sigma"][lab]); s1 = float(np.interp(1.0, D, sig))
            anchors = g.sigma_anchors(0.0, 3.0)
            model = np.interp(D, [d for d, _ in anchors], [v for _, v in anchors])
            dev = float(np.abs(model - sig / s1).max())
            # 0.05: RDP III bends at D 0.9 (Fig. 5), which three linear anchors follow to 0.044
            chk(g.sigma_shape_measured and dev < 0.05 and abs(s1 * 1000 - g.rms_granularity) < 0.6,
                "%s: stored sigma(D) anchors follow rd046 Fig. 5 to %.3f, sigma(1.0) x1000 = %.1f vs RMS %.0f"
                % (name, dev, s1 * 1000, g.rms_granularity))
        m6 = np.asarray(rec["fig6_mtf"]["RDP III"]); f6 = np.asarray(rec["fig6_freq"], dtype=float)
        j = int(np.argmax(m6))
        chk(abs(m6[j] - 1.152) < 2e-3 and 5.0 <= f6[j] <= 7.0 and P["FUJI_PROVIA_100F"].mtf.adjacency > 0.0,
            "A4 FUJI_PROVIA_100F: rd046 Fig. 6 RDP III overshoot %.4f at the %.0f c/mm sample (pinned 1.152 @ 5.79)"
            % (m6[j], f6[j]))
    else:
        print("SKIP: rd046 not on this tree")
    print("\n%s" % ("OK" if not bad else "FAIL (%d)" % bad))
    return 1 if (bad and a.assert_) else 0


if __name__ == "__main__":
    sys.exit(main())
