"""Dash-aware mutual-exclusion tracer for multi-track plots.

Written 2026-08-16 for the KODAK VISION3 granularity sheets, where three DASHED
sigma_D curves share one frame with three solid density curves. A plain
nearest-neighbour tracer needs a gap tolerance large enough to cross the dashes,
and that same tolerance lets it cross onto a NEIGHBOURING track -- the failure
that produced a sigma ratio of 44.3 on 5219 before this module existed. That is
the same class of bug as the merged tracks recorded in DIGITIZATION_QUEUE.md
batch 8, made worse by the dashes.

STATUS 2026-08-17: the VISION3 sigma(D) extraction is DONE and adopted, using
the three additions at the bottom of this module. `trace_tracks` and
`check_ordering` are kept unchanged because they are still the right primitives,
but on their own they were not sufficient, and the reason is recorded below.

WHAT `trace_tracks` + `check_ordering` COULD NOT DO, and what replaced it:

  1. CROSS-FAMILY MIGRATION WAS INVISIBLE. `check_ordering` validates order
     *within* the density trio and *within* the granularity trio. A track that
     leaves one family and joins the other violates neither, so three separate
     attempts produced clean-looking traces of hybrid curves. Fixed twice over:
     `family_split_by_style` makes the migration IMPOSSIBLE (the two families
     are separated into different ink masks before any tracing happens), and
     `check_cross_family` asserts afterwards that it did not happen anyway.
  2. THE CROSSINGS ARE TANGENTIAL, AND DECIDABLE IN ONLY ONE DIRECTION. Where a
     rising density curve passes through a granularity curve AT THAT CURVE'S
     MAXIMUM (log E ~ 1.1 on 5203), a LEFTWARD trace sees both branches descend
     from the junction with similar slope -- neither proximity nor slope can
     choose. A RIGHTWARD trace from the left plateau sees the density branch
     rising and the granularity branch at slope ~ 0, and separates cleanly.
     THE DIRECTION RULE, therefore: seed the density curves at their LEFT-EDGE
     dmin plateau and trace RIGHTWARD ONLY. Never seed them at mid-width and
     trace outward -- that is what failed three times.
  3. NEAREST-NEIGHBOUR STEPPING IS TOO WEAK. `trace_predictive` predicts from a
     local linear fit over the last N accepted points, extrapolated from the
     last REAL point. (Extrapolating from the previous *prediction* double-counts
     the slope and kills tracks mid-plot; that bug cost a full debugging round.)

SEEDING, corrected: identity-by-dmin at the left edge is necessary but NOT
sufficient on its own. Two of the four sheets do not even show six separated
runs there, because the granularity curves lie on top of the density curves at
the left edge -- 5203 gives 5 ink runs for 6 curves, 5213 gives 4. The style
split has to come first; then the left-edge seeding works on all four.

Rules this module enforces, so that a future pass cannot silently regress:
  * one ink run may be claimed by at most one track in a column (exclusion);
  * a track may step by at most `max_step` pixels between adjacent columns;
  * a gap is bridged only while it is shorter than `max_bridge`, which must be
    set from the MEASURED dash period, never guessed. On the VISION3 sheets the
    measured pattern is ~7 px of ink and ~6 px of gap, so 12 is the ceiling;
  * the tracks' vertical ordering is ASSERTED by `check_ordering`, not assumed.

Calibration notes for these particular plots, measured and reusable:
  * density axis 0.0-3.0 across the frame; on 5207 that is 147.7 px per density
    unit, confirmed by 19 evenly spaced minor ticks at 0.2 D intervals;
  * the right-hand granularity axis is logarithmic, sigma = 0.001 AT THE FRAME
    BOTTOM, and the tick comb just outside the right frame line resolves as a
    full 1,2,3..9,10,20..100 log ladder. Measured px per decade, from that comb:
    5203 139.00, 5207 139.75, 5213 139.00, 5219 140.25. Each sheet's two decades
    agree to <= 1 px and the within-decade ticks reproduce log10 to <= 0.5 px.
    (The earlier "~150 px per decade" note was a rough read; use these.)
  * CORRECTION to a claim this docstring used to make. It said the sigma_shape_*
    ratios are normalised at D = 1.0 "so a multiplicative error in that log
    calibration cancels exactly", and concluded that absolute sigma accuracy is
    not a blocker. That is only half true and it licences skipping a measurement
    that must not be skipped. With sigma = C * 10**((y0 - y) / P), a ratio is
    10**((y2 - y1) / P): an error in C or in y0 does cancel, an error in P does
    NOT. P has to be measured per sheet -- see the figures above. The sensitivity
    is mild (+/-1 % on P moves the dmax/mid ratio by 0.5 %) but it is not zero.
"""
from collections import deque

import numpy as np


def column_runs(ink, x, y0, y1):
    """Centres of the vertical ink runs in one column, top to bottom."""
    col = ink[:, x]
    out, cur = [], []
    for y in range(y0 + 2, y1 - 1):
        if col[y]:
            cur.append(y)
        elif cur:
            out.append((cur[0] + cur[-1]) / 2.0)
            cur = []
    if cur:
        out.append((cur[0] + cur[-1]) / 2.0)
    return out


def trace_tracks(ink, x_range, y0, y1, seeds, exclude=None,
                 max_step=3.0, max_bridge=12, exclude_tol=3.0):
    """Follow N tracks outward from `seeds`, both directions, with exclusion.

    seeds   -- {name: (x, y)}; all seeds must share the same x.
    exclude -- {name: {x: y}} of curves that must NOT be picked up (e.g. the
               solid density curves when tracing the dashed granularity ones).
    Returns {name: {x: y}}.
    """
    exclude = exclude or {}
    out = {k: {} for k in seeds}

    def excluded(x, y):
        for d in exclude.values():
            yy = d.get(x)
            if yy is not None and abs(yy - y) <= exclude_tol:
                return True
        return False

    for direction in (+1, -1):
        state = {k: dict(y=float(v[1]), miss=0, alive=True) for k, v in seeds.items()}
        x = seeds[list(seeds)[0]][0]
        while x_range[0] <= x <= x_range[1]:
            cands = [c for c in column_runs(ink, x, y0, y1) if not excluded(x, c)]
            pairs = sorted(
                ((abs(c - state[k]['y']), k, c)
                 for k in state if state[k]['alive'] for c in cands),
                key=lambda t: t[0])
            claimed, taken = set(), set()
            for dist, k, c in pairs:
                if k in claimed or c in taken or dist > max_step:
                    continue
                claimed.add(k)
                taken.add(c)
                state[k]['y'] = c
                state[k]['miss'] = 0
                out[k][x] = c
            for k in state:
                if state[k]['alive'] and k not in claimed:
                    state[k]['miss'] += 1
                    if state[k]['miss'] > max_bridge:
                        state[k]['alive'] = False
            if not any(state[k]['alive'] for k in state):
                break
            x += direction
    return out


def check_ordering(tracks, order):
    """Columns where the tracks' vertical order breaks: (violations, shared).

    Call this before trusting any trace. Zero violations is necessary, not
    sufficient -- a track can still be short. Report coverage as well.
    """
    names = [n for n in order if n in tracks]
    if len(names) < 2:
        return 0, 0
    shared = set(tracks[names[0]])
    for n in names[1:]:
        shared &= set(tracks[n])
    bad = 0
    for x in shared:
        ys = [tracks[n][x] for n in names]
        if any(ys[i] >= ys[i + 1] for i in range(len(ys) - 1)):
            bad += 1
    return bad, len(shared)


# ---------------------------------------------------------------------------
# 2026-08-17 additions. See the STATUS block at the top for why each exists.
# ---------------------------------------------------------------------------
def check_cross_family(density, granularity, min_margin=3.0):
    """Assert that no track has migrated between the two curve families.

    This is the check `check_ordering` structurally cannot perform: it validates
    order *within* a family, so a track that leaves the density trio and joins
    the granularity trio violates nothing it looks at. On the VISION3 sheets
    that exact migration survived three passes undetected.

    density, granularity -- {name: {x: y}}, as returned by the tracers.
    min_margin -- how many pixels of clear space a granularity track must keep
        from every density track in a column to count as separated.

    Returns (violations, shared_columns, worst_margin_px, worst_at). A violation
    is a column where some granularity/density pair sits closer than
    `min_margin`, i.e. where the two families are not resolved and a swap could
    have happened silently. `worst_at` is (column, density_name,
    granularity_name) for the tightest approach, so it can be inspected on the
    page image rather than argued about.

    ZERO VIOLATIONS IS NOT A PROOF OF CORRECTNESS. It says only that the
    families never came close enough for a swap to be possible in a traced
    column. Genuine crossings DO occur on these plots, and there the honest
    answer is that the column is undecidable and must be excluded, not that the
    trace is fine. Read the overlay as well -- always.
    """
    worst, worst_at, bad, shared = float("inf"), None, 0, 0
    gnames = [g for g in granularity if granularity[g]]
    dnames = [d for d in density if density[d]]
    cols = set()
    for g in gnames:
        cols |= set(granularity[g])
    for x in sorted(cols):
        pairs = [(dn, gn) for dn in dnames for gn in gnames
                 if x in density[dn] and x in granularity[gn]]
        if not pairs:
            continue
        shared += 1
        hit = False
        for dn, gn in pairs:
            m = abs(density[dn][x] - granularity[gn][x])
            if m < worst:
                worst, worst_at = m, (x, dn, gn)
            if m < min_margin:
                hit = True
        bad += hit
    return bad, shared, (None if worst == float("inf") else worst), worst_at


def _components(mask, connectivity=8):
    """Label connected components; yield (label_image, {label: (w, h, count)})."""
    lab = np.zeros(mask.shape, dtype=np.int32)
    info = {}
    n = 0
    offs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    if connectivity == 4:
        offs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    ys, xs = np.nonzero(mask)
    h, w = mask.shape
    for y0, x0 in zip(ys, xs):
        if lab[y0, x0]:
            continue
        n += 1
        q = deque([(y0, x0)])
        lab[y0, x0] = n
        mnx = mxx = x0
        mny = mxy = y0
        cnt = 0
        while q:
            y, x = q.popleft()
            cnt += 1
            mnx = min(mnx, x); mxx = max(mxx, x)
            mny = min(mny, y); mxy = max(mxy, y)
            for dy, dx in offs:
                yy, xx = y + dy, x + dx
                if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not lab[yy, xx]:
                    lab[yy, xx] = n
                    q.append((yy, xx))
        info[n] = (mxx - mnx + 1, mxy - mny + 1, cnt)
    return lab, info


def family_split_by_style(ink, mode, dash_max_w=35, bold_min_thick=4,
                          min_run_w=2, glyph_max_w=22, glyph_min_h=9):
    """Split a two-family plot into (granularity_ink, density_ink) BY STROKE STYLE.

    This is the structural half of the cross-family fix. These Kodak sheets
    distinguish the two families by how the line is DRAWN, not by where it sits,
    and 5219 prints a legend that says so in words ("Blue Density ... Blue
    Grain"). Separating on style first means a tracer working inside one mask
    cannot reach the other family's ink at all, crossings included.

    mode -- 'dash' for sheets whose granularity curves are dashed or dotted
        (5207, 5219): the three SOLID density curves come out as connected
        components several hundred px wide, everything else <= 11 px, so a
        width threshold separates them. Measured widths: 438/438/424 px on 5207
        and 442/442/442 on 5219 against a largest dash of 11 px -- a threshold
        of 35 px sits in a wide empty gap, not near either population.
    mode -- 'bold' for sheets whose granularity curves are drawn heavy but solid
        (5203, 5213): granularity strokes run 4-9 px thick, density strokes 1-3,
        so a vertical run-length threshold separates them.

    Text glyphs and the axis tick combs are dropped first, by geometry: a
    component narrower than `glyph_max_w` and at least `glyph_min_h` tall is a
    letter, and one <= 4 px wide and >= 6 tall is a tick. Dash segments are
    short but THIN, which is what keeps them out of that net.

    Returns (gran_ink, dens_ink) as boolean arrays over the same grid. Callers
    still supply the plot's interior bounds -- this function does not know where
    the frame is.
    """
    if mode not in ("dash", "bold"):
        raise ValueError("mode must be 'dash' or 'bold'")

    lab, info = _components(ink)
    keep = np.ones(len(info) + 1, dtype=bool)
    keep[0] = False
    for i, (w, h, _c) in info.items():
        if (w <= glyph_max_w and h >= glyph_min_h) or (w <= 4 and h >= 6):
            keep[i] = False
    clean = keep[lab]

    if mode == "bold":
        thick = np.zeros_like(clean)
        rows, cols = clean.shape
        for x in range(cols):
            col = clean[:, x]
            y = 0
            while y < rows:
                if col[y]:
                    y0 = y
                    while y < rows and col[y]:
                        y += 1
                    if y - y0 >= bold_min_thick:
                        thick[y0:y, x] = True
                else:
                    y += 1
        gran = thick
    else:
        lab2, info2 = _components(clean)
        narrow = np.zeros(len(info2) + 1, dtype=bool)
        for i, (w, _h, _c) in info2.items():
            narrow[i] = w <= dash_max_w
        gran = narrow[lab2]

    # A style test can leave speckles where the OTHER family's antialiasing
    # happens to be locally thick. Real curve fragments are wide; speckles are
    # not, so one width filter on the result removes them.
    lab3, info3 = _components(gran)
    wide = np.zeros(len(info3) + 1, dtype=bool)
    for i, (w, _h, _c) in info3.items():
        wide[i] = w >= min_run_w
    gran = wide[lab3]
    if mode == "bold":
        lab4, info4 = _components(gran)
        long_ = np.zeros(len(info4) + 1, dtype=bool)
        for i, (w, _h, _c) in info4.items():
            long_[i] = w >= 30
        gran = long_[lab4]

    return gran, clean & ~gran


def column_runs_weighted(ink, gray, x, y0, y1):
    """[(centroid, thickness_px)] per ink run in column x, ink-weighted centroid.

    `column_runs` above returns run midpoints. The intensity-weighted centroid
    is the better estimator on an antialiased raster, and the thickness comes
    back with it because thickness is what identifies the family on the bold
    sheets.
    """
    col = ink[:, x]
    groups, cur = [], []
    for y in range(y0 + 3, y1 - 3):
        if col[y]:
            cur.append(y)
        elif cur:
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)
    out = []
    for run in groups:
        ys = np.asarray(run, dtype=float)
        wt = 1.0 - gray[run, x]
        c = float(np.average(ys, weights=wt)) if wt.sum() > 0 else float(ys.mean())
        out.append((c, float(len(run))))
    return out


def trace_predictive(ink, gray, x_range, y0, y1, seed_x, seeds, direction=+1,
                     tol0=3.0, tol_grow=0.7, max_bridge=26, hist=16,
                     slope_cap=2.5, merge_px=0.0):
    """Follow N tracks from `seeds` in ONE direction, slope-predictively.

    Same exclusion guarantee as `trace_tracks` -- one ink run may be claimed by
    at most one track per column -- with two changes that were needed to get
    through the VISION3 crossings:

      * the predicted position for column x is (last REAL point) + slope * dx,
        with the slope from a linear fit over the last `hist` accepted points.
        Predicting from the previous PREDICTION instead compounds the slope and
        kills the track a few columns into any dash gap;
      * the acceptance tolerance widens with the miss count (`tol_grow` px per
        missed column), so a dash gap or a blanked crossing is bridged without
        having to set a single loose tolerance everywhere.

    ONE DIRECTION ONLY, deliberately. See the direction rule in the module
    docstring: on these plots the tangential density/granularity junction is
    decidable rightward from the left plateau and undecidable leftward, so a
    caller that traces "both ways from a mid seed" gets a hybrid curve back and
    no check will notice.

    ⚠ merge_px -- COAST THROUGH A CROSSING INSTEAD OF GUESSING AT IT. Default 0.0
    keeps the original behaviour exactly, so no existing caller changes. Set it
    and, at any column where two live tracks predict positions within merge_px of
    each other, NEITHER is allowed to claim ink: both count a miss and continue on
    their own fitted slope until they separate again.

    This exists because of a measured failure on Gevacolor 682 Fig. 8, where the
    dotted cyan curve descends through the dash-dot magenta curve at about 425 nm.
    For roughly twelve columns the two traces are one ink run. Greedy nearest
    assignment gives that run to whichever track predicts closer -- fine in itself
    -- but the run is then ACCEPTED INTO THAT TRACK'S SLOPE HISTORY, and the
    merged ink is nearly flat, so the descending track's fitted slope collapses
    from +0.75 px/column to +0.3. When the curves separate again the flattened
    prediction lands on the WRONG branch, and both curves swap identity with every
    residual still small and no ordering check able to see it: the traced pair
    still looks like two smooth curves, just not the two that were printed.

    Refusing to decide is the correct answer at a crossing -- the ink genuinely
    does not say which curve it belongs to -- and coasting on the slope measured
    BEFORE the merge is the only information that does. Measured on that figure:
    with merge_px = 0 the two low curves come back swapped (peak assignment 522 nm
    to the track seeded on the 683 nm curve); with merge_px anywhere in 6-12 px
    all three peaks land on the right tracks and agree with the printed values.

    The cost is a gap: the crossing columns carry no sample for either track. On
    682 Fig. 8 that is about 14 columns, 8 nm, less than one grid step of the
    stored 10 nm sampling, and it is filled by interpolation between measured
    points rather than by a guessed assignment.

    seeds -- {name: y} at column `seed_x`.
    Returns {name: {x: y}}.
    """
    st = {k: dict(pts=[(seed_x, float(v))], miss=0, alive=True)
          for k, v in seeds.items()}
    out = {k: {} for k in seeds}
    x = seed_x
    while x_range[0] <= x <= x_range[1]:
        cands = [c for c, _t in column_runs_weighted(ink, gray, x, y0, y1)]
        pred = {}
        for k, s in st.items():
            if not s['alive']:
                continue
            p = s['pts'][-hist:]
            if len(p) >= 3:
                xs = np.array([q[0] for q in p], dtype=float)
                ys = np.array([q[1] for q in p], dtype=float)
                sl = float(np.polyfit(xs, ys, 1)[0])
                sl = max(-slope_cap, min(slope_cap, sl))
            else:
                sl = 0.0
            x_last, y_last = s['pts'][-1]
            pred[k] = y_last + sl * (x - x_last)
        merged = set()
        if merge_px > 0.0:
            live = list(pred)
            for i in range(len(live)):
                for j in range(i + 1, len(live)):
                    if abs(pred[live[i]] - pred[live[j]]) <= merge_px:
                        merged.add(live[i])
                        merged.add(live[j])
        pairs = sorted(((abs(c - pred[k]), k, c)
                        for k in pred if k not in merged for c in cands),
                       key=lambda t: t[0])
        claimed, taken = set(), set()
        for dist, k, c in pairs:
            if k in claimed or c in taken:
                continue
            if dist > tol0 + tol_grow * st[k]['miss']:
                continue
            claimed.add(k)
            taken.add(c)
            st[k]['pts'].append((x, c))
            st[k]['miss'] = 0
            out[k][x] = c
        for k, s in st.items():
            if s['alive'] and k not in claimed:
                s['miss'] += 1
                if s['miss'] > max_bridge:
                    s['alive'] = False
        if not any(s['alive'] for s in st.values()):
            break
        x += direction
    return out
