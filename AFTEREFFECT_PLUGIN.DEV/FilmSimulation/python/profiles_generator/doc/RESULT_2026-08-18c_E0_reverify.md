# RESULT 2026-08-18c — E0: eleven profiles re-verified against sheets that were declared absent

## Why this run existed

`film_profiles.py` carried a LOCAL-ARCHIVE CAVEAT, written 2026-07-31, naming twelve documents as
"no copy on file" and putting the corpus at 270 PDFs. On 2026-08-18 a file-by-file recount found
**448 PDFs and eleven of the twelve documents present**. Two of the refutations came from inside
this project's own work: `Ektachrome_100d.pdf` is the file `dye_density.py` already opens, and
`KODAK/5247.pdf` **is** TI0835, whose printed EI 125 the 5247 split had turned on.

Finding the file is not reading it. **No profile had ever been checked against its own sheet.**
This run did that for all eleven, plus settled a citation-year conflict.

Method: text layers extracted with `pdftotext -layout`, then five independent readers, each given
the stored values and told to quote the printed line for every number and to write NOT PRINTED
where a value is absent. **Every load-bearing quote was then re-checked by hand against the text
layer before any value was changed** — which mattered, because two of the readers' headline
findings did not survive that check (see "Findings that were rejected").

---

## 1. Values changed — five

| Stock | Field | Was | Now | Printed evidence |
|---|---|---|---|---|
| `EASTMAN_5247_1983` | `rms_granularity` | 13.0 | **5.0** | "rms Granularity: less than 5 / Read with a microdensitometer, (red, green, blue) using a 48-micrometre aperture" |
| `EASTMAN_5247_1983` | resolving power | (0, 0) | **(50, 100)** | "ISO RPL 50 lines/mm (TOC 1.6:1)" / "ISO RP 100 lines/mm (TOC 1000:1)" |
| `EASTMAN_5247_1983` | reciprocity | 1.00/1.00/1.00 @ 1.0 s | **0.85/0.85/0.88 @ 0.1 s** | "no filter corrections … from 1/10,000 to 1/10 second. At an exposure time of 1 second, use a KODAK Color Compensating Filter CC10Y and increase exposure by 1/2 stop" |
| `EASTMAN_PLUS_X_5231` | `dmin` | 0.120 | **0.210** | "Base Density =0.19" + "Fog 0.02" |
| `EASTMAN_PLUS_X_5231` | reciprocity | 0.95/0.95/0.95 @ 1.0 s | **0.85/0.85/0.85 @ 0.1 s** | "no filter corrections … from 1/10,000 to 1/10 second. At an exposure time of 1 second, increase exposure by 1⁄2 stop" |

### 1.1 The grain change is the big one, and it reverses an earlier decision

`EASTMAN_5247_1983` carried `rms 13.0` with an explicit note that Chibisov 1988's printed RMS 5
was **recorded but not adopted**, because "cross-era metric equivalence of the printed sigma_D
figure is unverified". That was sound reasoning about a Soviet secondary source whose granularity
metric could not be shown to be Kodak's.

**TI0835 removes the objection**, because it is Kodak's own sheet for this exact coating and it
states the figure on the 48 µm / net-density-1.0 convention this database already uses for the
VISION3 stocks. Two independent sources now agree on ~5, and 13.0 sat **2.6× above Kodak's
printed upper bound**.

What is stored is the **bound**, not a measurement — the sheet says "less than 5". For a bound of
this kind that is the conservative direction: the true value lies at or below it. The per-channel
triple is the old 18.5/20.0/24.0 rescaled by 5/13, so the **layer ratio is inherited and remains
unmeasured** — only the level is grounded.

**Not applied to `EASTMAN_5247_1974`.** That profile reconstructs the original EI 100 coating;
TI0835 documents the improved EI 125 one. Keeping values from crossing between the two
generations is the entire point of the split.

### 1.2 A leftover from the 5247 split, found by re-reading

`_RESOLVING_POWER` held `"EASTMAN_5247_1974": (50.0, 100.0)` under a comment saying the pair came
from **TI0835** — the sheet for the *other* generation. The split moved the spectral plate to
`_1983` and missed this dict, leaving the `[T3]` reconstruction claiming a resolving power sourced
from a document about a different emulsion. **Moved to `_1983`; `_1974` now correctly has none.**
A `verify.py` guard now asserts that, and would have caught the original error.

### 1.3 Two reciprocity entries built by the file's own documented method

The convention was already established in `_RECIPROCITY_OVERRIDES` (from the VISION2 5205 entry):
with a printed "no correction to 1/10 s" and a single anchor at 1 s, `onset_s = 0.1` and
`1 − p = stops × log₁₀2`.

* **PLUS-X 5231** — +½ stop at 1 s → `1 − p = 0.1505`, **p = 0.85**, achromatic (B&W, no filter
  prescribed). Replaces the generic monochrome heuristic, which was both too shallow and started
  a decade too late.
* **5247 1983** — same construction, but the sheet prescribes **CC10Y**, so the failure is
  **chromatic**. The file's own rule reads the direction off the filter colour: a CC10Y transmits
  red and green and attenuates blue, so it boosts red and green relative to blue, meaning **blue
  lost the least** → `p_b` sits 0.03 *above* `p_r`/`p_g`. The mirror image of 5205's CC10R case.
  It previously claimed *no failure at all*, which was wrong in kind, not just in magnitude.

---

## 2. Findings that were **rejected** after checking

Both were plausible, both came back from readers as headline corrections, and both would have
introduced errors.

**"Ten missing tungsten exposure indices."** Every Kodak cine sheet prints a second exposure
index, and the stored profiles carry only one. But for the **colour** stocks that second figure is
**filter-derived** — 5203 "Tungsten (3200K): 12 (with 80A filter)", 5285 "25 (with 80A)", 7239
"40/17 *with 85B filter*", 5213/5219/5247 daylight indices with a No. 85. Those are filter
factors, not film speeds, and this database already refuses to store them (the precedent is
recorded on `KONICA_VX_100`). The three **genuine** dual ratings — DOUBLE-X 200, PLUS-X 64,
TRI-X 160 — were **already stored**, from the Photo-Lab-Index. What actually improved is their
provenance: all three are now confirmed by the manufacturer rather than by a compendium.

**"NEOPAN 1600's base tint conflicts with the sheet."** The sheet prints "Gray-tinted …
Triacetate 0.122 mm" while `base_tint` is identity. Not a conflict: `base_tint` models a **colour**
cast, and a grey tint is by definition **neutral**. It belongs in the density — and it is already
there, which is why this stock's `dmin` is 0.211, high for a B&W negative. Setting some
`(k, k, k)` would require a tint density the sheet never prints.

A third reader claim — that TRI-X 7266's reciprocity **conflicts** with the sheet — was
downgraded rather than rejected. The sheet's "no correction … 1/1,000 to 1 second" *confirms*
`onset_s = 1.0`; its other statement ("+½ stop at 1/10,000 second") is **high-intensity
short-exposure** failure, the opposite end of the curve, which `ReciprocitySpec` cannot express at
all. Recorded as a schema limitation.

---

## 3. The structural finding: these sheets mostly don't print numbers

| Sheet family | Prints numeric rms / TOC pair? | Plots |
|---|---|---|
| VISION3 5203 / 5207 / 5213 / 5219 | **No** — method only ("48-micrometre aperture", "multiply by 1000") | **raster**, p3–p4 |
| DOUBLE-X 5222 | **Yes** — 14, and 32/100 | raster |
| PLUS-X 5231 | **Yes** — 10, and 32/100 | **vector**, p3 |
| TRI-X 7266 | **No** — graphical read-off only, no TOC block at all | raster |
| EKTACHROME 7239 | **Yes** — 14, and 40/100 | **vector**, p3 (4 plots) |
| EKTACHROME 100D 5285 | **No** — curves + "multiply by 1000" | **vector**, p3–p4 |
| 5247 / TI0835 | **Yes** — "less than 5", and 50/100 | raster, pp6–9 |
| NEOPAN 1600 | **No** | raster |

So for the four VISION3 stocks, **re-verification is re-tracing, not re-reading** — their stored
granularity, gamma and dmin came off raster curves and cannot be confirmed from printed text in
either direction. That is worth knowing before anyone budgets another verification pass.

**Three digit-for-digit agreements** came out of this run (5222 14/32/100, 5231 10/32/100, 7239
14/40/100). Those are now the best-evidenced numbers in the granularity and resolving-power
fields, and `verify.py` pins them — an exact agreement is evidence, and silent drift away from it
would destroy that evidence.

---

## 4. The Sehlin/Kennel year is settled: **1985**

The paper had no text layer, so page 1 was rendered and read as an image.

> **Sehlin, R. C., Kennel, G. L., Ortman, E. F., Reinking, F. R., "Choosing Eastman Color
> Negative Film 5247 or Eastman Color High-Speed Negative Film 5294", *SMPTE Journal*, July 1985,
> pp. 724–734.**

Every page footer prints "SMPTE Journal, July 1985"; the first-page footnote prints "Copyright ©
1985 by the Society of Motion Picture and Television Engineers" **and explains the filename**:
"Presented at the 125th SMPTE Technical Conference in Los Angeles (paper No. 125-40) on November
2, 1983 … This article was received in final form on September 14, 1984."

**1983 is the conference date. The file `Sehlin_Kennel_etal_1983_…pdf` is misnamed, and this
database's existing 1985 citation was right all along.** No volume or issue number is printed
anywhere in the scan. Granularity-versus-exposure data is in Figs 7, 8, 9 (and a referenced
Fig. 11), which is what queue item E5 wants.

---

## 5. Hazards recorded so they cannot recur

* **`H-1-5294` is not `H-1-5285`.** Two sheets in the corpus are both called "Ektachrome 100D";
  one documents 5285/7285 (held) and the other 5294/7294 (not held). Both answer a search.
* **Searching the corpus for "H-1-5247" hits the 7239 file.** Page 1 of
  `Kodak Eastman EKTACHROME Film (Daylight) 7239.pdf` carries stray template text reading
  "H-1-5247" and "August 1996". Every genuine colophon in that document reads H-1-5239, and the
  file is a single 4-page document with no 5247 content.
* **5213 has no rem-jet; 5219 does.** Printed on both sheets ("An Anti-halation undercoat replaces
  the traditional remjet backing layer" vs "acetate safety base with rem-jet backing"). If any
  backing-dependent behaviour is ever keyed by stock, it must honour that split.
* **TI0835's plates are ten years older than its text.** Body stamped "Revised 6-93", but all four
  graph plates read TI0835A/B/C/D **6-83**. The plotted data is 1983 data in a 1993 wrapper —
  which is the evidence behind the `_1983` name.
* **The 5219 sheet on file is the 2022 revision** while its three siblings are 2026 revisions.

---

## 6. `ReciprocitySpec` is the third carrier with no consumer

Nothing in `film_sim.py` reads reciprocity — `grep` finds zero references. It is emitted to C++
and read by no renderer, exactly like `sigma_shape_*` and `mtf_tail_*`. So the two reciprocity
corrections above are **documentation-grade, not rendering-grade**: they change what the database
*says*, not what any frame looks like, until a consumer exists.

That makes three unread carrier families. Worth deciding as one piece of work rather than three.

---

## 7. New extraction targets this run surfaced

* **`EASTMAN_EKTACHROME_7239` p3 is VECTOR and carries a spectral-dye-density plot.** 7239 is one
  of the five sheets that **failed** B1's dye extraction — the source was never the problem.
* **`EASTMAN_PLUS_X_5231` p3 is VECTOR**: MTF, sensitometric and spectral curves, none with printed
  numbers. Its stored f50 60.0 is an estimate that a trace could ground.
* **`KODAK_EKTACHROME_100D_5285` pp3–4 are VECTOR** and include the diffuse-rms-granularity curves
  — the σ(D) shape for a colour reversal stock, which the database has for no reversal film.

---

## Files touched

`film_profiles.py` (5 values, 10 provenance entries rewritten, 1 dict entry moved),
`verify.py` (+13 checks), `doc/NotFound.md`, `doc/DIGITIZATION_QUEUE.md`,
`PDF/PROFILES/Index.md` (on the owner's disk).
