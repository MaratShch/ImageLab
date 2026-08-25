#!/usr/bin/env python3
"""Generate doc/FilmActiveProfiles.md -- the coverage and traceability report.

Every "+" in the output means: a manufacturer document, an official technical
publication, or a machine-traced plot from one of those backs that property
for that stock. Every "-" means the value is currently an estimate, an
analogy from a related stock, or a reconstruction.

WHY THIS IS GENERATED AND NOT WRITTEN BY HAND
  The evidence already exists in two machine-readable places and one
  human-readable one:
    * structured fields that only ever get populated from a document
      (MTFSpec.resolving_power_*, SpectralSensitivity arrays,
      Provenance.sources / .tier / .fitted_from);
    * the per-field comments inside each FilmProfile literal, which carry the
      citations written while each number was adopted ("SOURCE PDF/...",
      "published diffuse RMS granularity, CONFIRMED", "[T1]", "Fit RMS ...").
  A hand-maintained table would drift from those within a week. This script
  reads both, so regenerating it after any profile edit keeps the report true.

DETECTION RULES, stated so a reader can audit them
  has_doc      the stock's Provenance.sources holds a real citation (not the
               _NO_DATASHEET placeholder) OR its source block cites a document
               ("SOURCE ", "PDF/PROFILES/", "publication", "Technical Data").
  Per property, "+" requires has_doc AND property-specific evidence:
    Spectral Sensitivity  digitised curves present, or a documented
                          sensitisation range / limit in nm.
    H&D Curve             fitted_from == "datasheet_curve" with a real
                          citation, or an explicit trace/fit record
                          ("digitis", "traced", "Fit RMS", "sensitometric").
    Spectral Response     digitised SpectralSensitivity arrays present. This
                          is the stricter form of the first column: curve
                          DATA, not merely a documented range.
    Grain Characteristics documented crystal/clump/structure statement.
    RMS Granularity       an rms/granularity figure marked published,
                          confirmed, printed or sourced.
    MTF / Resolving       MTFSpec.resolving_power_lp_mm_highc > 0 (only ever
                          set from a sheet), or documented lp/mm or MTF.
    Film Base             documented base material or thickness.
    Emulsion Properties   documented layer count/thickness/coating/grain type.
    Processing            documented process or developer (ECN-2, C-41, E-6,
                          D-96, ID-11, ...).
    Exposure Latitude     documented latitude, push range or EI range.
    Dynamic Range         documented density range, Dmax or latitude in logH.
    Colour Characteristics  colour stock with documented dye/mask/balance or
                          densitometry status.
    Additional Physical   documented reciprocity, filter factors, base
                          shrinkage, anti-halation/remjet, DX coding.

  Two deliberate consequences to keep in mind while reading the table:
    * A "-" is never a claim that the property is missing from the model --
      every stock renders. It says the number is not backed by a document.
    * Monochrome stocks show "-" under Colour Characteristics by definition,
      not for lack of research.

Usage:  python3 gen_active_profiles.py [-o doc/FilmActiveProfiles.md]
"""
from __future__ import annotations

import argparse
import re

import film_profiles as fp
from film_profiles import Feature

# The spectral consumers, imported so this report can state what the RENDERER
# actually does with each curve rather than only whether a curve exists. Added
# 2026-08-13 when the curves stopped being inert; see
# CHANGES_2026-08-13_spectral_path.md. Imported lazily-tolerantly because this
# generator must still run if film_sim's optional dependencies are unavailable.
try:
    import film_sim as _fs
except Exception:                                   # pragma: no cover
    _fs = None

PLACEHOLDER = "No official manufacturer datasheet available"

MANUFACTURER = [
    ("KODAK", "Eastman Kodak"), ("EASTMAN", "Eastman Kodak"),
    ("KODACHROME", "Eastman Kodak"), ("TECHNICOLOR", "Technicolor"),
    ("FUJI", "Fujifilm"), ("AGFA", "Agfa-Gevaert"), ("GEVA", "Gevaert"),
    ("ORWO", "ORWO / VEB Filmfabrik Wolfen"), ("SVEMA", "Svema (Shostka)"),
    ("TASMA", "Tasma (Kazan)"), ("SOVCOLOR", "Soviet (Sovcolor)"),
    ("SOVIET", "Soviet"), ("ILFORD", "Ilford / HARMAN"),
    ("KENTMERE", "HARMAN technology"), ("KONICA", "Konica / Konica Minolta"),
    ("ROLLEI", "Maco / Rollei"), ("MACO", "Maco"), ("FOMA", "Foma Bohemia"),
    ("FERRANIA", "Film Ferrania"), ("POLAROID", "Polaroid"),
    ("DUFAY", "Dufay-Chromex"), ("LUMIERE", "Lumiere"),
    ("CINESTILL", "CineStill"), ("GENERIC", "generic amateur stock"),
]


#: Which measurement standard each stored code corresponds to. The profiles
#: carry the codes; this is the mapping to the published standard, so a reader
#: can tell what a density or a speed number actually means. Mixing Status M
#: and Status A, or an ISO speed and a manufacturer EI, silently corrupts any
#: comparison between stocks -- which is exactly why the codes exist.
DENSITY_STD = {
    "status_m": "Status M (ISO 5-3)",
    "status_a": "Status A (ISO 5-3)",
    "visual_iso": "visual diffuse (ISO 5-3)",
}
SPEED_STD = {
    "iso6": "ISO 6",
    "iso2240": "ISO 2240",
    "iso5800": "ISO 5800",
    "manufacturer_ei": "manufacturer EI (no standard)",
}

PROPS = [
    ("Spectral Sensitivity", "spec_any"),
    ("Characteristic (H&D) Curve", "hd"),
    ("Spectral Response Curves", "spec_curve"),
    ("Spectral Curve Consumed By", "spec_used"),
    ("Film Grain Characteristics", "grain"),
    ("RMS Granularity", "rms"),
    ("Grain sigma(D) Shape", "sigma_shape"),
    ("MTF / Resolving Power", "mtf"),
    ("Film Base Properties", "base"),
    ("Emulsion Properties", "emul"),
    ("Processing Characteristics", "proc"),
    ("Exposure Latitude", "lat"),
    ("Dynamic Range", "dr"),
    ("Color Characteristics", "colour"),
    ("Additional Physical Properties", "phys"),
]

RX = {
    "doc": re.compile(r"SOURCE |PDF/PROFILES|publication |Technical Data|"
                      r"datasheet|Data Sheet|technical data|Gurlev|"
                      r"Гурлев|GOST|ГОСТ", re.I),
    "hd": re.compile(r"digitis|digitiz|traced|Fit RMS|sensitometric|"
                     r"characteristic curve|D-logE|gamma_rec|H&D", re.I),
    "spec_any": re.compile(r"sensiti[sz]ation|spectral sensitivity|"
                           r"panchromat|orthochromat|nm\b|sensitisation limit", re.I),
    "grain": re.compile(r"crystal|clump|grain structure|grain size|cubic|"
                        r"tabular|T-grain|flat.grain", re.I),
    "rms": re.compile(r"(rms|granularity)[^.]{0,120}?"
                      r"(publish|confirm|printed|sheet|SOURCE|diffuse|"
                      r"datasheet|measured)", re.I | re.S),
    # Added 2026-08-17 with the VISION3 sigma(D) adoption. Deliberately narrow:
    # it matches the provenance wording used when a triple is TRACED from a
    # vendor plot, so a tier-3 estimated triple does not pick up a "+".
    "sigma_shape": re.compile(r"sigma\(D\)[^.]{0,160}?"
                              r"(TRACED|traced|published plot|vendor plot|"
                              r"Granularity Curves)|"
                              r"sigma_shape[^.]{0,160}?"
                              r"(TRACED|traced|published plot)", re.I | re.S),
    "mtf": re.compile(r"lp/mm|lines/mm|lin/mm|MTF|resolving power", re.I),
    "base": re.compile(r"polyester|\bPET\b|acetate|triacetate|nitrate|"
                       r"Estar|clear base|base thickness|remjet|\bum base\b|"
                       r"grey base|gray base", re.I),
    "emul": re.compile(r"layer thickness|emulsion thickness|multilayer|"
                       r"three-emulsion|supercoat|single thin emulsion|"
                       r"coating|\bum layer|thin emulsion", re.I),
    "proc": re.compile(r"ECN|C-41|CNK|\bE-6\b|CRK|D-96|D-76|ID-11|Microfine|"
                       r"Refinal|Konicadol|developer|process(ed|ing)?\b|"
                       r"Rodinal|Xtol|HC-110|Perceptol|CT-2", re.I),
    "lat": re.compile(r"latitude|push(ed|able|ing)?\b|EI range|"
                      r"overexpos|exposure index range|\bEI \d+ to", re.I),
    "dr": re.compile(r"density range|dynamic range|D-?max|Dmax|"
                     r"latitude[^.]{0,40}log", re.I),
    "colour": re.compile(r"\bdye\b|dyes\b|mask|saturation|colour balance|"
                         r"color balance|Status M|Status A|imbibition|"
                         r"interimage|inter-image", re.I),
    "phys": re.compile(r"reciprocity|Schwarzschild|shrinkage|curl|"
                       r"anti-halation|remjet|\bDX\b|Wratten|filter factor|"
                       r"AURA", re.I),
}

CITE = re.compile(
    r"SOURCE\s+([^\n.]{6,140})"
    r"|((?:PDF/PROFILES/)[\w./+-]+\.pdf)"
    r"|(Kodak publication [\w-]+)"
    r"|(publication ([\w-]+))"
    r"|(H-1-[\w.]+)"
    r"|(TI\d{4})"
    r"|(AF3-[\w]+)"
    r"|(TDS[NB]-\d+)")


def blocks_from_source(path: str = "film_profiles.py") -> dict[str, str]:
    """Per-profile source text, so the inline citations are visible."""
    src = open(path, encoding="utf-8").read()
    out: dict[str, str] = {}
    hits = list(re.finditer(r'name="([A-Z0-9_]+)",', src))
    for i, m in enumerate(hits):
        end = hits[i + 1].start() if i + 1 < len(hits) else len(src)
        blk = src[m.start():end]
        # BOUND THE BLOCK AT THE END OF ITS OWN LITERAL. Fixed 2026-08-13.
        # A profile body is indented; anything starting in column 0 is the next
        # module-level statement and belongs to nobody. Without this cut the
        # textually LAST profile in each tuple swallowed everything up to the
        # next name= match -- which for the last FilmProfile means the whole
        # tail of the module, including _PROVENANCE_SOURCES. The citation
        # scanner then harvested other stocks' document numbers and printed
        # them as this stock's references. Observed concretely:
        # EASTMANCOLOR_5248_1953 was credited with Kodak publications F-4016,
        # F-4043 and F-4001, which are the T-MAX sheets and have nothing to do
        # with a 1953 Eastmancolor negative. The bug predates this fix and
        # simply moved from stock to stock as the last profile changed, so any
        # earlier report may carry the same contamination on whichever stock
        # happened to be last.
        cut = re.search(r"\n(?=[A-Za-z_@#])", blk)
        if cut:
            blk = blk[:cut.start()]
        out[m.group(1)] = blk
    return out


def manufacturer(name: str) -> str:
    for key, who in MANUFACTURER:
        if name.startswith(key) or key in name:
            return who
    return "unknown"


def film_type(p) -> str:
    bits = []
    if p.is_monochrome:
        bits.append("B&W")
    else:
        bits.append("Colour")
    if p.reseau is not None:
        bits.append("additive mosaic")
    if p.name == "TECHNICOLOR_THREE_STRIP":
        bits.append("3-strip separation")
    bits.append("reversal" if p.is_reversal else "negative")
    return ", ".join(bits)


def official_name(p, block: str) -> str:
    """Prefer a real product name from the description, else the key."""
    m = re.search(r'"\[T[123]\]\s*([^".]{4,60})', block)
    return p.name.replace("_", " ")


def citations(p, block: str, extra_sources: tuple = ()) -> str:
    found: list[str] = []
    for s in tuple(p.provenance.sources) + tuple(extra_sources):
        if PLACEHOLDER not in s:
            found.append(s)
    for m in CITE.finditer(block):
        txt = next((g for g in m.groups() if g), "").strip(" ,;:")
        if len(txt) > 4 and txt not in found:
            found.append(txt)
    # de-duplicate while keeping order, and keep the cell readable
    seen: list[str] = []
    for f in found:
        f = re.sub(r"\s+", " ", f).strip()
        if f and not any(f in s for s in seen):
            seen.append(f)
    return "; ".join(seen[:4]) if seen else "-"


#: A property keyword only counts as documented when a document marker sits
#: within this many characters of it. Without the proximity test, a stock that
#: cites a sheet for ONE number scores "+" on every property whose keyword
#: appears anywhere in its prose -- measured on the first run, that inflated
#: grain and MTF coverage to 80%, with KENTMERE PAN 100 credited for
#: granularity and resolving power that its sheet does not print.
PROX_CHARS = 320

#: Profile comments record ABSENCES as carefully as presences -- "the sheet
#: prints no granularity or resolving-power numbers", "RMS not printed on the
#: sheet", "grain SIZE is not measurable from those files". Those sentences
#: contain the property keyword AND sit beside the citation, so a naive
#: proximity test reads them as evidence FOR the property. Measured: it
#: credited KENTMERE PAN 100 with resolving power its sheet explicitly does
#: not print. This window is scanned backwards from each keyword for a
#: negation before the hit is allowed to count.
NEG_WINDOW = 90
RX_NEG = re.compile(
    r"\b(no|not|never|none|without|absent|lack|lacks|unmeasurab\w*|"
    r"cannot|can't|un(known|available)|missing|omits?|omitted|"
    r"rejected|estimate[sd]?|assumed|unverified|only)\b", re.I)


def _documented_near(block: str, key: str) -> bool:
    """True when a property keyword sits close to a document marker AND is not
    inside a sentence that denies the property."""
    doc_at = [m.start() for m in RX["doc"].finditer(block)]
    if not doc_at:
        return False
    for m in RX[key].finditer(block):
        pos = m.start()
        if not any(abs(pos - d) <= PROX_CHARS for d in doc_at):
            continue
        before = block[max(0, pos - NEG_WINDOW):pos]
        # also look a little past the keyword: "RMS not printed on the sheet"
        after = block[m.end():m.end() + 40]
        if RX_NEG.search(before) or RX_NEG.search(after):
            continue
        return True
    return False


def evaluate(p, block: str) -> dict[str, str]:
    real_src = any(PLACEHOLDER not in s for s in p.provenance.sources)
    has_doc = real_src or bool(RX["doc"].search(block))
    sp = p.spectral
    curves = sp.has_data

    def mark(flag: bool) -> str:
        return "+" if flag else "-"

    res = {}
    res["spec_any"] = mark(curves or _documented_near(block, "spec_any"))
    res["hd"] = mark(
        (p.provenance.fitted_from == "datasheet_curve" and real_src)
        or _documented_near(block, "hd")
    )
    res["spec_curve"] = mark(curves)

    # -- what the RENDERER does with the curve, not whether one exists --------
    # Until 2026-08-13 every one of these curves was stored and read by nothing,
    # so "curve present" and "curve used" were the same cell. They are now
    # different questions and this column answers the second one.
    #
    #   balance          the curve drives colour-temperature balance in both the
    #                    Python reference and the C++ engine. True for every
    #                    stock that carries curves: the balance derivation needs
    #                    no primary basis, so nothing can disqualify it.
    #   +mono            the monochrome-weight derivation additionally passes the
    #                    basis-reach guard. Available but OFF by default.
    #   mono refused     the guard refuses it: the emulsion is sensitised beyond
    #                    what three visible primaries can excite, and projecting
    #                    onto them would derive a confidently wrong answer.
    if not curves:
        res["spec_used"] = "-"
    elif _fs is None:
        res["spec_used"] = "balance"
    else:
        peak = _fs.spectral_peak_lambda(p)
        out = _fs.spectral_out_of_reach(p)
        if sp.log_s_pan and not (sp.log_s_r and sp.log_s_g and sp.log_s_b):
            if _fs.spectral_monochrome_weights(p) is not None:
                res["spec_used"] = "balance, +mono"
            else:
                res["spec_used"] = ("balance, mono refused (peak %g nm, %.0f%% "
                                    "beyond reach)" % (peak or 0.0,
                                                       100.0 * (out or 0.0)))
        else:
            res["spec_used"] = "balance"
    res["grain"] = mark(_documented_near(block, "grain"))
    res["rms"] = mark(_documented_near(block, "rms"))
    # A default triple is not a property at all, so it cannot be "documented":
    # report "-" rather than letting a stock with (0,1,0) inherit a "+" from
    # neighbouring granularity prose.
    _shape = (p.grain.sigma_shape_toe, p.grain.sigma_shape_mid,
              p.grain.sigma_shape_dmax)
    res["sigma_shape"] = mark(_shape != (0.0, 1.0, 0.0)
                              and _documented_near(block, "sigma_shape"))
    # ⚠ `mtf_measured` IS EVIDENCE AND USED NOT TO COUNT, 2026-08-23. The rule
    # was keyword-only, so a stock whose curve had actually been traced off its
    # sheet could still be printed in red as "the model's own estimate" -- which
    # is what happened to KODAK_VISION_500T_5279 and to the two Fuji Super-F
    # stocks: their prose says "contrast transfer function" and "Coltman", never
    # the literal token the regex wants. The flag is set only by an audit that
    # re-derives the curve from the sheet on every build, so it is STRONGER
    # evidence of documentation than any keyword match, and it short-circuits.
    res["mtf"] = mark(
        p.mtf.mtf_measured
        or p.mtf.resolving_power_lp_mm_highc > 0.0
        or _documented_near(block, "mtf")
    )
    for key in ("base", "emul", "proc", "lat", "dr", "phys"):
        res[key] = mark(_documented_near(block, key))
    res["colour"] = mark(
        (not p.is_monochrome) and _documented_near(block, "colour")
    )
    return res




# ---------------------------------------------------------------------------
# Stocks whose numbers come from a Soviet TU (технические условия) -- a state
# MANUFACTURING SPECIFICATION. Those figures are ACCEPTANCE LIMITS, not
# measurements of a sample: "RMS <= 22" means no batch was allowed to be
# grainier than 22, and real stock generally sat inside the limit. Presenting
# them as measured values would overstate what is known and would bias the
# whole Soviet section pessimistically -- worst permitted grain, minimum
# permitted sharpness, minimum permitted latitude. They are therefore marked
# as a THIRD class, distinct from both "documented measurement" and "model
# estimate", and the legend says so.
_SPEC_LIMIT_STOCKS = {
    "SVEMA_DS_4", "SVEMA_DS_5M", "SVEMA_LN_8", "SVEMA_LN_9", "SVEMA_LN_9S",
    "SVEMA_CO_32D",
}

def _f(v, nd=2):
    return ("%%.%df" % nd) % v


def _tri(t, nd=3):
    return "/".join(("%%.%df" % nd) % v for v in t)


def numeric_cells(p, ev, blocks_all) -> list[str]:
    """One numeric cell per property column.

    A trailing ``*`` means the number is the model's own estimate rather than a
    documented figure -- the same evidence test that produced the +/- table,
    carried over so no precision is implied where none exists.
    """
    def mk(key, text):
        # THREE classes, not two:
        #   plain            -- documented measurement
        #   blue + dagger    -- a SPECIFICATION LIMIT from a Soviet TU: the
        #                       real film was no worse than this, but the
        #                       measured value is unknown and generally better
        #   red + asterisk   -- the model's own estimate, nothing documented
        # The middle class was added 2026-08-17: without it a TU ceiling reads
        # as a measurement, which silently converts "no batch was allowed to
        # exceed 22" into "this film measures 22".
        if ev[key] == "+":
            if p.name in _SPEC_LIMIT_STOCKS:
                return ('<span style="color:#1560bd" title="specification '
                        'limit, not a measurement">%s\u2020</span>' % text)
            return text
        return '<span style="color:red">%s*</span>' % text

    cv = p.curves
    mono = p.is_monochrome
    g = p.grain
    m = p.mtf
    sp = p.spectral
    ii = p.interimage
    co = p.coating
    rc = p.reciprocity

    # spectral sensitivity: the weights actually used by the renderer
    c_spec = mk("spec_any", _tri(p.spectral_weights))

    # H&D: gamma and base+fog
    if mono:
        c_hd = mk("hd", "g %.3f  Dmin %.3f" % (cv.g.gamma, cv.g.dmin))
    else:
        c_hd = mk("hd", "g %.3f/%.3f/%.3f  Dmin %.2f/%.2f/%.2f" % (
            cv.r.gamma, cv.g.gamma, cv.b.gamma,
            cv.r.dmin, cv.g.dmin, cv.b.dmin))

    # digitised spectral curves: point count and range
    if sp.has_data:
        n_lay = sum(1 for l in (sp.log_s_r, sp.log_s_g, sp.log_s_b, sp.log_s_pan) if l)
        n_pts = max(len(sp.log_s_r), len(sp.log_s_g), len(sp.log_s_b),
                    len(sp.log_s_pan))
        hi = sp.lambda_start_nm + (n_pts - 1) * sp.lambda_step_nm
        c_curves = "%dx%d pts, %.0f-%.0f nm @%.0f" % (
            n_pts, n_lay, sp.lambda_start_nm, hi, sp.lambda_step_nm)
    else:
        c_curves = "none"

    # What the RENDERER does with that curve. Before 2026-08-13 the answer was
    # "nothing" for every stock, so this column did not need to exist; the curve
    # column above answered both questions at once. It now reports the second
    # question separately, which is the distinction this whole report is about:
    # a field existing is not a field being used.
    c_used = ev.get("spec_used", "-")

    c_grain = mk("grain", "clump %.1f/%.1f/%.1f um  gain %.2f  fog %.2f" % (
        g.clump_um_r, g.clump_um_g, g.clump_um_b, g.clump_gain, g.fog_grain))

    rms = g.rms_rgb()
    c_rms = mk("rms", "%.1f" % g.rms_granularity if rms[0] == rms[1] == rms[2]
               else "%.1f (%.1f/%.1f/%.1f)" % ((g.rms_granularity,) + rms))
    # sigma(D) SHAPE, added 2026-08-17. Before that date no stock had a traced
    # triple, so a column reporting it would have been (0,1,0) for all 154 and
    # this report would have been right to omit it. The four VISION3 stocks now
    # carry one read off the vendor plot, and a traceability report that omitted
    # a newly documented [T1] property would be exactly the drift this file
    # exists to prevent. The tier is decided by the same evidence scan as every
    # other cell, so an estimated triple still prints red.
    _shape = (g.sigma_shape_toe, g.sigma_shape_mid, g.sigma_shape_dmax)
    c_shape = ("-" if _shape == (0.0, 1.0, 0.0)
               else mk("sigma_shape", "%.2f/%.2f/%.2f" % _shape))

    rp = ""
    if m.resolving_power_lp_mm_highc > 0:
        if m.resolving_power_lp_mm_lowc > 0:
            rp = "  RP %.0f/%.0f lp/mm" % (m.resolving_power_lp_mm_lowc,
                                           m.resolving_power_lp_mm_highc)
        else:
            # several sheets print only the 1000:1 figure
            rp = "  RP %.0f lp/mm @1000:1" % m.resolving_power_lp_mm_highc
    c_mtf = mk("mtf", "f50 %.0f/%.0f/%.0f c/mm%s" % (m.f50_r, m.f50_g, m.f50_b, rp))

    base = "tint %s" % _tri(p.base_tint)
    if Feature.NITRATE_BASE in p.features:
        base += ", nitrate"
    c_base = mk("base", base)

    if mono:
        emul = "single silver layer"
    elif p.reseau is not None:
        emul = "panchro + %.0f l/mm reseau" % p.reseau.lines_per_mm
    elif p.name == "TECHNICOLOR_THREE_STRIP":
        emul = "3 separate B&W records"
    else:
        emul = "tripack, dye cloud %.1f um" % g.dye_cloud_um
    emul += ", misreg %.1f um" % p.misregistration_um
    c_emul = mk("emul", emul)

    c_proc = mk("proc", "%s; %s; Callier q %.2f" % (
        DENSITY_STD.get(p.density_metric, p.density_metric),
        SPEED_STD.get(p.speed_criterion, p.speed_criterion),
        p.callier_q))
    c_lat = mk("lat", "%.1f stops" % cv.g.latitude_stops)
    c_dr = mk("dr", "Dmax %.2f, range %.2f" % (cv.g.dmax, cv.g.dmax - cv.g.dmin))

    if mono or not ii.active:
        c_col = mk("colour", "silver tone %+.2f" % p.silver_tone if mono else "none")
    else:
        pct = _iie_pct(p)
        c_col = mk("colour", "IIE %.0f/%.0f/%.0f%%  DIR %.2f" % (
            pct[0], pct[1], pct[2], p.couplers.strength))

    phys = "recip p %.3f @%.1fs" % (rc.schwarzschild_p_g, rc.onset_s)
    phys += ", vig %.2f st" % p.default_vignette
    if co.has_coating_field:
        phys += ", coat %.1f%%" % (co.coating_sigma * 100.0)
    if co.has_buckle:
        phys += ", buckle %.0f%%" % (co.buckle_mtf_loss * 100.0)
    if co.has_edge_fog:
        phys += ", edge fog %.3f D" % co.edge_fog_density
    c_phys = mk("phys", phys)

    # Cell order MUST match PROPS, or the per-stock table silently misaligns:
    # the header is built from PROPS and the rows from this list.
    return [c_spec, c_hd, c_curves, c_used, c_grain, c_rms, c_shape, c_mtf,
            c_base, c_emul, c_proc, c_lat, c_dr, c_col, c_phys]


def _iie_pct(p):
    """Model interimage percentage per (blue, green, red) receiver, by the
    US5273870A protocol -- reuses the calibrator in film_profiles."""
    coef = (p.interimage.a_rg, p.interimage.a_gr, p.interimage.a_br)
    r, g, b = fp._iie_measure((p.curves.r, p.curves.g, p.curves.b), list(coef),
                              max(p.interimage.iterations, 1))
    return b, g, r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", default="doc/FilmActiveProfiles.md")
    args = ap.parse_args()

    blocks = blocks_from_source()
    names = {p.name for p in fp.FILM_PROFILES}

    def parent_of(p):
        cands = [n for n in names if n != p.name and p.name.startswith(n + "_")]
        return max(cands, key=len) if cands else None

    def evidence_block(p) -> str:
        """A gauge variant is the SAME EMULSION as its parent -- the coating
        line did not know what width the roll would be slit to -- so it
        inherits the parent's documentation. Without this, the retired SVEMA_FN_64_8MM
        scored "-" on every property while the parent stock scored "+" on most,
        which would misreport identical emulsions as differently evidenced."""
        blk = blocks.get(p.name, "") + " " + p.description
        par = parent_of(p)
        if par:
            blk += " " + blocks.get(par, "")
        return blk

    by_name = {p.name: p for p in fp.FILM_PROFILES}
    rows = []
    for p in fp.FILM_PROFILES:
        blk = evidence_block(p)
        par = parent_of(p)
        extra = by_name[par].provenance.sources if par else ()
        ev = evaluate(p, blk)
        rows.append((p, ev, citations(p, blk, extra)))

    L: list[str] = []
    w = L.append
    w("# FilmActiveProfiles.md — coverage and traceability report")
    w("")
    w(f"Generated by `gen_active_profiles.py` from `film_profiles.py` "
      f"(schema v{fp.SCHEMA_VERSION}). **{len(rows)} film stocks.** "
      f"Regenerate after any profile edit — this file is derived, never "
      f"hand-edited.")
    w("")
    # ⚠ A LIVE COVERAGE SUMMARY, added 2026-08-19 at the owner's request: this file
    # is the thing that should say, at a glance, WHAT STILL NEEDS SEARCHING. Every
    # number below is counted from the live database at generation time, so it
    # cannot go stale the way a hand-written summary does.
    _n = len(fp.FILM_PROFILES)
    _sig = sum(1 for q in fp.FILM_PROFILES if q.grain.sigma_shape_measured)
    _mtfm = sum(1 for q in fp.FILM_PROFILES if q.mtf.mtf_measured)
    _spec = sum(1 for q in fp.FILM_PROFILES
                if getattr(q, "spectral", None) and q.spectral.has_data)
    _dye = sum(1 for q in fp.FILM_PROFILES
               if getattr(q, "dye_density", None) and q.dye_density.has_data)
    _rp = sum(1 for q in fp.FILM_PROFILES
              if q.mtf.resolving_power_lp_mm_lowc > 0
              or q.mtf.resolving_power_lp_mm_highc > 0)
    _stack = sum(1 for q in fp.FILM_PROFILES
                 if getattr(q, "layer_stack", None) and q.layer_stack.has_data)
    _recip = sum(1 for q in fp.FILM_PROFILES
                 if getattr(q, "reciprocity_table", None)
                 and q.reciprocity_table.has_data)
    _PLACE = "No official manufacturer datasheet available"
    _nosrc = sorted(q.name for q in fp.FILM_PROFILES
                    if q.provenance.sources
                    and all(_PLACE in x for x in q.provenance.sources))
    w("## Measured coverage, counted live at generation time")
    w("")
    w("This is the answer to \"what still needs searching\". Each row counts the "
      "stocks whose value for that property comes from a **document or a traced "
      "curve**, against the %d in the database." % _n)
    w("")
    w("| Property | Measured | Remaining | What would close the gap |")
    w("|---|---|---|---|")
    w(f"| Grain sigma(D) shape | **{_sig}** | {_n - _sig} | raster granularity "
      f"pages remain on disk; four were read by vision3_granularity.py and one "
      f"by the 7266 pass (2026-08-25b). Every measured shape is still Kodak, so "
      f"every non-Kodak sigma(D) is unmeasured |")
    w(f"| MTF rolloff shape | **{_mtfm}** | {_n - _mtfm} | 199 vector MTF pages "
      f"inventoried, 26 curves on 12 sheets traced (queue C2b) |")
    w(f"| Resolving power (printed pair) | **{_rp}** | {_n - _rp} | only sheets "
      f"that print a TOC pair; many never did |")
    w(f"| Spectral sensitivity curve | **{_spec}** | {_n - _spec} | mostly a "
      f"tracing job on pages already held |")
    w(f"| Spectral dye density curve | **{_dye}** | {_n - _dye} | 191 vector "
      f"dye-density pages inventoried (queue B1, G7) |")
    w(f"| Layer stack (coating order) | **{_stack}** | {_n - _stack} | printed as "
      f"a diagram on some sheets; Cheltsov & Bongard 1958 Table 24 for the rest |")
    w(f"| Reciprocity table | **{_recip}** | {_n - _recip} | almost never printed "
      f"as a table. ⚠ THE PARENTHETICAL HERE WAS STALE UNTIL 2026-08-25: it read "
      f"\"`ReciprocitySpec` is also still read by no renderer (queue C8)\", but C8 "
      f"closed 2026-08-23 -- reciprocity is wired into both renderers through "
      f"`RenderSettings.exposure_time_s` / `AlgoControls::exposureTimeS` and is "
      f"parity-audited |")
    w("")
    w("**%d stocks carry no source at all** beyond the `_NO_DATASHEET` placeholder: "
      "%s. `GENERIC_BW` and `GENERIC_COLOR` are in that list by design -- they are "
      "generic classes, not gaps. See `NotFound.md` section 1 for a per-stock "
      "acquisition plan." % (len(_nosrc), ", ".join("`%s`" % n for n in _nosrc)))
    w("")
    w("## How to read this")
    w("")
    w("Every cell carries the **actual value the simulator uses**, in one of "
      "**three** classes:")
    w("")
    w("| Marking | Meaning |")
    w("|---|---|")
    w("| plain | a **documented measurement** from a manufacturer datasheet, "
      "standard or measured curve |")
    w("| <span style=\"color:#1560bd\">blue with a dagger \u2020</span> | a "
      "**specification limit** from a Soviet TU (технические условия) -- a state "
      "manufacturing specification. `RMS <= 22` means no batch was permitted to be "
      "grainier than 22; the real film generally sat **inside** the limit and its "
      "measured value is unknown. Reading these as measurements would bias the "
      "affected stocks pessimistically: worst permitted grain, minimum permitted "
      "sharpness, minimum permitted latitude |")
    w("| <span style=\"color:red\">red with an asterisk `*`</span> | the model's "
      "own **estimate** -- nothing documented, no precision implied |")
    w("")
    w("The evidence test behind the second and third classes is the same one that "
      "produced the earlier +/- table; only the presentation distinguishes a "
      "specification ceiling from a measured value.")
    w("")
    w("### Units and meaning, column by column")
    w("")
    w("| Column | Contents |")
    w("|---|---|")
    w("| Spectral Sensitivity | renderer weights R/G/B, sum 1.0 |")
    w("| Characteristic (H&D) Curve | straight-line gamma and base+fog density; "
      "three values = per dye layer |")
    w("| Spectral Response Curves | digitised points x layers, wavelength range, "
      "sampling step |")
    w("| Spectral Curve Consumed By | which render path actually reads the "
      "curve. `balance` = colour-temperature balance, live in both builds. "
      "`+mono` = the monochrome-weight derivation also passes the basis-reach "
      "guard (available, OFF by default). `mono refused` = the guard rejects "
      "it because the emulsion is sensitised beyond the reach of three visible "
      "primaries. `-` = no curve on file |")
    w("| Film Grain Characteristics | grain correlation length (clump) per "
      "layer in um, clump gain, fog grain |")
    w("| RMS Granularity | sigma(D)x1000 through a 48 um aperture at **NET "
      "density 1.0** (i.e. dmin + 1.0), which is the convention Kodak prints "
      "-- '*Read at a net diffuse visual density of 1.0, using a 48-micrometre "
      "aperture' (5248 p1, 5222 p1). Schema v9 pinned this: the renderer "
      "reproduces the stored figure at exactly that density. Brackets = "
      "per-layer |")
    w("| Grain sigma(D) Shape | sigma multipliers at D=dmin / D=1.0 / D=dmax "
      "describing how granularity varies with density, plus the stored INTERIOR "
      "PEAK where one is measured. `-` = the legacy sqrt(D-dmin) law, which is "
      "what the renderer uses for every stock without the `sigma_shape_measured` "
      "flag. **%d stocks carry a traced shape** -- four VISION3 stocks from "
      "raster plots (2026-08-17), seven more from vector plots (2026-08-18, "
      "queue items C1c/E0b), KODAK_EKTACHROME_100D_5285, and "
      "KODAK_TRI_X_REVERSAL_200 (2026-08-25b, queue C29 -- the first B&W stock "
      "and the only one whose sigma RISES toward dmax). The other %d hold "
      "either the default or a "
      "heuristic triple that is INERT because both branches of that heuristic "
      "are wrong in sign. Schema v8 added `sigma_shape_peak`, so an interior "
      "maximum IS now representable and is stored where measured (1.23-3.13x the "
      "D=1.0 value, at D 0.65-3.34) |"
      % (sum(1 for q in fp.FILM_PROFILES if q.grain.sigma_shape_measured),
         sum(1 for q in fp.FILM_PROFILES
             if not q.grain.sigma_shape_measured)))
    w("| MTF / Resolving Power | f50 in cycles/mm per layer; RP = resolving "
      "power at 1.6:1 / 1000:1 contrast. Since schema v10 the ROLLOFF SHAPE is "
      "also stored and read, as 1/(1+(f/f50)^q) behind an `mtf_measured` flag "
      "-- **%d stock(s) measured**, the rest keep the legacy Gaussian "
      "bit-for-bit. Both laws pass through 0.5 at f50 exactly, so a measured "
      "rolloff changes shape and never level |"
      % sum(1 for q in fp.FILM_PROFILES if q.mtf.mtf_measured))
    w("| Film Base Properties | base transmittance tint R/G/B, base material "
      "when notable |")
    w("| Emulsion Properties | layer architecture, dye cloud size, channel "
      "misregistration |")
    w("| Processing Characteristics | densitometry status and speed criterion |")
    w("| Exposure Latitude | toe-to-shoulder span of the green record, stops |")
    w("| Dynamic Range | asymptotic Dmax and Dmax-Dmin of the green record |")
    w("| Color Characteristics | interimage effect per receiver B/G/R as the "
      "percentage gamma steepening of US5273870A, and DIR coupler strength; "
      "monochrome stocks show silver image tone instead |")
    w("| Additional Physical Properties | Schwarzschild exponent and its onset, "
      "lens vignette in stops, coating sigma, gate buckling, edge fog |")
    w("")
    w("Two things a `-` does **not** mean:")
    w("")
    w("* It is not a gap in the model. Every stock renders every property — "
      "the simulator has no missing inputs. A `-` is a statement about "
      "*evidence*, not about capability.")
    w("* Monochrome stocks carry `-` under Colour Characteristics by "
      "definition, not for want of research.")
    w("")
    w("Detection is mechanical and auditable: structured fields that only ever "
      "get filled from a document (resolving power, digitised spectral "
      "arrays, `Provenance`), plus the per-field citations written into each "
      "profile literal as its numbers were adopted. The exact rules are in "
      "the `gen_active_profiles.py` docstring.")
    w("")

    tier = {1: 0, 2: 0, 3: 0}
    for p, _, _ in rows:
        tier[p.provenance.tier] = tier.get(p.provenance.tier, 0) + 1
    # spec_used carries free text, not a +/- mark, so it is excluded from the
    # coverage tally -- counting it would make the percentages meaningless.
    tot = {k: 0 for _, k in PROPS if k != "spec_used"}
    for _, ev, _ in rows:
        for _, k in PROPS:
            if ev[k] == "+":
                tot[k] += 1
    w("## Measurement standards")
    w("")
    w("Every number above is only meaningful against the conditions it was "
      "measured under, so the profiles store the densitometry and speed "
      "criterion per stock rather than assuming one. Mixing Status M with "
      "Status A, or an ISO speed with a manufacturer EI, silently corrupts "
      "any comparison between stocks.")
    w("")
    w("| Standard | Governs | Where it appears here |")
    w("|---|---|---|")
    w("| **ISO 5-3** | spectral conditions for density: Status M, Status A, "
      "visual diffuse | Processing column; sets what every Dmin/Dmax/gamma "
      "figure means |")
    # ⚠ THESE FOUR COUNTS WERE HARDCODED AND ALL FOUR WERE WRONG BY 2026-08-25
    # (27/34/13/15 against a live 51/58/17/34). Derived from the database now, for
    # the same reason the placeholder count was: a hand-typed census goes stale
    # the next time a stock is added and nothing notices.
    _spd = {}
    for _q in fp.FILM_PROFILES:
        _spd[_q.speed_criterion] = _spd.get(_q.speed_criterion, 0) + 1
    w("| **ISO 6** | black-and-white negative film speed | Processing "
      "column, %d stocks |" % _spd.get("iso6", 0))
    w("| **ISO 5800** | colour negative film speed | Processing column, "
      "%d stocks |" % _spd.get("iso5800", 0))
    w("| **ISO 2240** | colour reversal film speed | Processing column, "
      "%d stocks |" % _spd.get("iso2240", 0))
    w("| **ISO 6328** | photographic resolving power | the RP figures in the "
      "MTF column. Kodak TI0835 states its 50/100 lp/mm for 5247 were "
      "measured by the *ISO 6328-1982* method |")
    w("| **ANSI PH2.39** | MTF measurement | Kodak TI0835 cites it (60% AIM) "
      "for the 5247 MTF curves from which f50 values were read |")
    w("| 48 um aperture convention | diffuse RMS granularity | the RMS "
      "column. Manufacturer practice rather than an ISO number: Kodak sheets "
      "specify a 48 um aperture at D=1.0 above Dmin, Konica sheets state "
      "\"12X, 0.048 mm, Dmin+1.0\" |")
    w("| **US5273870A** definition | interimage effect as percentage gamma "
      "steepening, separation vs white-light exposure | the IIE figures in "
      "the Colour column; the patent cites T. H. James, *The Theory of the "
      "Photographic Process*, 4th ed. (1977), pp. 574 and 614 |")
    w("")
    w("Two honest caveats on the standards themselves:")
    w("")
    w("* **ISO 5-3, ISO 6, ISO 5800 and ISO 2240 are cited as the framework "
      "the stored codes denote, not as documents in this library.** They were "
      "not purchased or read; the codes were assigned from what each "
      "datasheet states about its own measurement conditions. ISO 6328 and "
      "ANSI PH2.39 are different -- those two are cited *by Kodak TI0835 "
      "itself*, so their attribution is documented rather than inferred.")
    w("* `manufacturer EI` on %d stocks means no standard applies: mostly "
      % _spd.get("manufacturer_ei", 0) + 
      "pre-1960 stocks rated before the modern speed standards existed, plus "
      "the generic amateur-gauge entries. Their ISO/ASA column is a "
      "manufacturer or historical figure, not a standardised speed.")
    w("")
    w("Callier q in the Processing column is the specular-to-diffuse density "
      "ratio (1.0 for dye images, higher for silver). Mees 1942 p.235 gives "
      "the grain-size relation behind it: Callier showed q *\"is closely "
      "related to grain size and increases with it\"*, and Eggert & Kuster "
      "measured d = 6.8 log q.")
    w("")
    w("## Coverage summary")
    w("")
    w("| Property | Documented values | of | % |")
    w("|---|---:|---:|---:|")
    for label, k in PROPS:
        if k == "spec_used":
            # Free-text column: reported separately below, not as a percentage.
            continue
        w(f"| {label} | {tot[k]} | {len(rows)} | {100.0*tot[k]/len(rows):.0f}% |")
    w("")
    # Consumption tally. This is the number the project actually cares about:
    # not how many curves EXIST but how many the renderer READS. Before
    # 2026-08-13 the second number was zero for every stock.
    _bal = sum(1 for p_, ev_, _ in rows if ev_.get("spec_used", "-") != "-")
    _mono = sum(1 for p_, ev_, _ in rows
                if "+mono" in ev_.get("spec_used", ""))
    _ref = sum(1 for p_, ev_, _ in rows
               if "refused" in ev_.get("spec_used", ""))
    w(f"**Spectral curve consumption.** {_bal} of {len(rows)} stocks carry "
      f"digitised sensitivity, and all {_bal} now drive colour-temperature "
      f"balance in both the Python reference and the C++ engine — a path that "
      f"projects onto no primary basis and so cannot be disqualified. Of the "
      f"monochrome stocks among them, **{_mono}** additionally pass the "
      f"basis-reach guard for the monochrome-weight derivation (available, OFF "
      f"by default) and **{_ref}** are refused by it, because the emulsion is "
      f"sensitised beyond what three visible primaries can excite. Until "
      f"2026-08-13 every one of these curves was read by nothing. See "
      f"`CHANGES_2026-08-13_spectral_path.md`.")
    w("")
    w(f"Confidence tiers as recorded in each profile: "
      f"**tier 1** {tier.get(1,0)}, **tier 2** {tier.get(2,0)}, "
      f"**tier 3** {tier.get(3,0)}.")
    w("")
    docd = sum(1 for _, _, c in rows if c != "-")
    w(f"Stocks citing at least one document: **{docd} of {len(rows)}**. "
      f"The remaining {len(rows)-docd} are reconstructions from historical "
      f"and secondary sources — mostly pre-1960 stocks and the generic "
      f"amateur-gauge entries, for which no manufacturer sheet is known to "
      f"survive.")
    w("")
    w("## Per-stock detail")
    w("")
    hdr = (["Film Name", "Manufacturer", "Production Years", "Film Type",
            "ISO / ASA"] + [lbl for lbl, _ in PROPS] + ["Reference Documents"])
    w("| " + " | ".join(hdr) + " |")
    w("|" + "|".join(["---"] * 5 + [":-:"] * len(PROPS) + ["---"]) + "|")
    for p, ev, cite in rows:
        cells = [
            official_name(p, ""), manufacturer(p.name), p.era, film_type(p),
            str(p.exposure_index),
        ] + numeric_cells(p, ev, blocks) + [cite]
        w("| " + " | ".join(c.replace("|", "/") for c in cells) + " |")
    w("")
    w("## Known limitations of this report")
    w("")
    w("This table is derived by pattern-matching the profile source, so it "
      "carries a deliberate bias and two known error classes. Both are stated "
      "here rather than hidden, because a traceability report that overstates "
      "its own evidence is worse than none.")
    w("")
    w("**Biased toward `-` on purpose.** Evidence must sit within "
      f"{PROX_CHARS} characters of a citation and must not be inside a "
      "negation. Profile comments record absences as carefully as presences "
      '("the sheet prints no granularity or resolving-power numbers"), so '
      "without the negation guard those sentences read as proof of the very "
      "property they deny. An earlier run of this generator, before the "
      "guard, reported 80% coverage for grain and MTF and credited "
      "KENTMERE PAN 100 with resolving power its sheet explicitly does not "
      "print. Under-crediting is the safer direction for this document.")
    w("")
    w("**Known false negatives.** ROLLEI INFRARED 400 shows `-` for RMS "
      "Granularity although its sheet prints 11.0; KENTMERE PAN 100 shows "
      "`-` for Film Base although its sheet states 0.125 mm acetate. Both are "
      "the negation guard being too eager on nearby words.")
    w("")
    # 2026-08-18: this count was hardcoded at 27 and went stale the moment the
    # placeholder closure registered six citations. It is now MEASURED from the
    # same two facts the cell rendering already uses -- placeholder-only
    # `Provenance.sources`, and a citation this generator can recover from the
    # block -- so the sentence cannot drift from the table above it again.
    _gap = sorted(p.name for p, _ev, cite in rows
                  if all(PLACEHOLDER in s for s in p.provenance.sources)
                  and cite != "-")
    # Split by tier, because the two halves mean different things: a tier<=2
    # profile with no registered source contradicts its own tier claim (that
    # class was closed on 2026-08-18 and verify.py now guards it), whereas a
    # tier 3 profile is only mis-described by the placeholder's wording.
    _gap_lo = [n for n in _gap if by_name[n].provenance.tier <= 2]
    # Also measured: profiles where the placeholder is the TRUE answer. Stating
    # this number beside the gap stops the closed state from reading as though
    # every profile now has a citation -- 13 legitimately do not.
    _true_ph = sorted(p.name for p in fp.FILM_PROFILES
                      if all(PLACEHOLDER in s for s in p.provenance.sources)
                      and p.name not in _gap)
    if not _gap:
        w(f"**The upstream registry gap is CLOSED (measured, not hardcoded).** "
          f"Zero stocks now cite a document in their profile comments while "
          f"still carrying the `_NO_DATASHEET` placeholder in "
          f"`Provenance.sources`. That gap stood at 27 by the previous "
          f"hardcoded count and was closed on 2026-08-18 by lifting seven "
          f"citations out of profile prose into `_PROVENANCE_SOURCES`; "
          f"`verify.py` now fails if a tier-1 profile, or an undocumented "
          f"tier-2 profile outside a named allowlist, carries the placeholder "
          f"again. **The count above is now derived from the same two facts "
          f"this table's cells use**, so it cannot go stale the way the "
          f"hardcoded 27 did. {len(_true_ph)} stocks still carry the "
          f"placeholder and SHOULD: no document for them exists in the corpus "
          f"or anywhere this project has looked. ⚠ THIS SENTENCE USED TO NAME "
          f"`FUJI_F125_8530` and `FUJI_F125_8630` as the only profiles whose "
          f"tier claim had nothing behind it. Both halves went stale: 8530 "
          f"carries three citations as of 2026-08-24 (Honjo 1989 plus two "
          f"issues of «Техника кино и телевидения»), and 8630 no longer "
          f"exists -- it was a gauge clone and was removed. See "
          f"`NotFound.md` §1.5.")
    else:
        w(f"**A registry gap upstream, {len(_gap)} "
          f"stock{'' if len(_gap) == 1 else 's'} (measured, not hardcoded).** "
          f"{'This stock cites a document' if len(_gap) == 1 else 'These cite documents'} "
          f"in {'its' if len(_gap) == 1 else 'their'} profile comments but "
          f"still carr{'ies' if len(_gap) == 1 else 'y'} the `_NO_DATASHEET` "
          f"placeholder in `Provenance.sources`, having been added without "
          f"being registered in `_PROVENANCE_SOURCES`. This generator recovers "
          f"the citation by parsing the source block, so the Reference "
          f"Documents column is correct -- but the structured field is not, "
          f"and any other consumer of `Provenance` sees the placeholder. "
          + ", ".join(f"`{n}`" for n in _gap) + ". "
          + (f"⚠ **{len(_gap_lo)} of these claim tier <= 2**, which the "
             f"placeholder directly contradicts: "
             + ", ".join(f"`{n}`" for n in _gap_lo) + ". "
             if _gap_lo else "")
          + "Registering them is bookkeeping with no research in it; see "
            "`DIGITIZATION_QUEUE.md`.")
    w("")
    w("**The definitive fix** is per-field provenance in the schema -- a "
      "small `srcs` marker beside each adopted number -- which would replace "
      "all of this inference with a lookup. That is a schema change and needs "
      "approval before it is made.")
    w("")
    w("## What this report says about the database")
    w("")
    w("* Tone reproduction and spectral curves are the best-evidenced "
      "properties, because manufacturers print those plots and they can be "
      "machine-traced.")
    w("* Grain **size** (clump dimensions) is the weakest across the whole "
      "library: sheets print RMS granularity but never clump geometry, and it "
      "cannot be recovered from a scan below the scanner's own resolution.")
    # ⚠ THIS LINE SAID "tier 3 for every stock without exception" until
    # 2026-08-20, contradicting InterimageSpec's own docstring, which has said
    # "Tier 2, upgraded from tier 3 on 2026-08-03" since that upgrade. The
    # coefficients are solved per stock against published patent measurements
    # (US5273870A's IIE percentages, converted with each stock's own gamma), so
    # tier 2 is right and this generator was stale.
    w("* Interimage effects are **tier 2**: no manufacturer DATASHEET publishes "
      "them -- the whole of `PDF/PROFILES` was searched (measured at 448 PDFs / "
      "559 files on 2026-08-18; the earlier \"395 documents\" here was a "
      "stale hand count, corrected 2026-08-25), and the "
      "omission is systematic because camera negative is characterised with a "
      "single white-light exposure series -- but the PATENT literature does, and "
      "the stored coefficients are solved per stock against US5273870A's "
      "published IIE percentages using each stock's own curve gamma. The one "
      "estimated input in that chain is the red white-light gamma (0.55), which "
      "is why it is tier 2 and not tier 1. `density_weighting` for reversal "
      "stocks remains tier 3 -- the mechanism split is documented (US4729943A), "
      "the 0.65 magnitude is not. See "
      "`doc/CHANGES_2026-08-03_v5_interimage.md`.")
    w("* Pre-1960 and Soviet stocks depend most on reconstruction. Where the "
      "owner supplied real scan batches (Svema, Tasma, ORWO) the profiles "
      "carry measured grain and tone instead — see "
      "`doc/REPORT_FN64_355.md` and `doc/SOVIET_EXTRACTION_2026-08-02.md`.")
    w("")
    open(args.output, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print(f"[INFO] wrote {args.output}: {len(rows)} stocks, "
          f"{docd} citing documents")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
