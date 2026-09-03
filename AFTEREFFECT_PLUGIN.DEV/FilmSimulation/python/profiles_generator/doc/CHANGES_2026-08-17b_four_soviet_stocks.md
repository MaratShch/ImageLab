# Four Soviet stocks added from their TU specifications — 2026-08-17

`SVEMA_LN_8`, `SVEMA_LN_9`, `SVEMA_LN_9S`, `SVEMA_CO_32D`. **144 → 148 stocks.**

## The presentation change that had to come first

A TU (технические условия) is a **state manufacturing specification**. Its numbers are
**acceptance limits**, not measurements of a sample. "RMS ≤ 19" means no batch was permitted
to be grainier than 19; real stock generally sat inside the limit and its measured value is
unknown. Building profiles straight from limits produces, by construction, a film at its
worst permitted grain, minimum permitted sharpness and minimum permitted latitude — and the
bias is uniformly pessimistic, so it does not average out across several stocks.

`FilmActiveProfiles.md` therefore now has **three** marking classes instead of two:

| Marking | Meaning |
|---|---|
| plain | documented **measurement** |
| **blue with a dagger †** | **specification limit** from a TU — the film was no worse than this; the measured value is unknown and generally better |
| red with an asterisk `*` | the model's own **estimate** |

Without the middle class a TU ceiling reads as a measurement, which silently converts "no
batch was allowed to exceed 22" into "this film measures 22". The class applies to
`SVEMA_DS_4`, `SVEMA_DS_5M` and the four new stocks.

## Why these four, and not the other four

Recommended and accepted: **ЛН-8**, **ЛН-9 + ЛН-9С**. Argued against but added at the
owner's request: **ЦО-32Д** (his own childhood slide film — a legitimate reason in a
simulator, and the coarseness is disclosed in the profile rather than smoothed over).
Left out: **ЦО-90Д and ЦО-90Л**, whose two documents carry near-identical norms and would
produce two stocks that render the same, distinguished by nothing the data contains.

## The visual-verification gate paid for itself

The OCR of these typewritten scans repeatedly detaches values from their row labels. Every
number was read from the page image, and that caught real errors:

- **ЛН-8's minimum single-layer sensitivity: OCR said 60, the page says 80.**
- ЛН-8's deformation temperature is **40 °C**, not the 33 °C of the ДС-5М family.
- ЛН-9's sensitivity balance read as garbage in OCR; the page says **1.5**.
- The ЛН-9 table's items 5–10 are printed in **one shared column for both marks** — OCR had
  interleaved them as if each mark had its own values.

## What each stock got

### `SVEMA_LN_8` — ТУ 6-17-1109-88, masked colour negative, professional cine
S ≥ 100; balance ≤ 2.0; mean gradient **0.60 / 0.54 / 0.50** (b/g/r, +0.06 −0.04); Dmin
0.70–1.05 / 0.25–0.60 / ≤ 0.25; latitude ≥ 1.5; RMS **≤ 19 green, ≤ 21 red**; MTF ≥ 0.30
green, ≥ 0.15 red; filter-layer efficiency ≥ 1.0; **red-layer sensitisation limit ≤ 690 nm**;
seven dye-impurity ratios including a **negative** term (K^п_с2 = minus 0.05–0.10, the only
one in the batch); colour separation 75/40/20, 10/80/30, −5/10/135; deformation ≥ 40 °C.

⚠ **The spatial frequency for the MTF row is not printed.** 30 mm⁻¹ is assumed from the
family convention (ДС-5М, ЛН-9 and ЦНД-64 all state it explicitly) and the assumption is
recorded in the profile, not hidden.

### `SVEMA_LN_9` and `SVEMA_LN_9S` — ТУ 6-17-1443-88, one emulsion, two constructions
Shared: S ≥ 100; balance ≤ 1.5; gradients 0.60 / 0.54 / 0.50; latitude ≥ 1.5; filter-layer
efficiency ≥ 1.0; **MTF at an explicitly stated 30 mm⁻¹ of 0.40 ± 0.05 green and 0.22 ± 0.03
red — given as tolerances, not minima, uniquely in this batch**; **RMS ≤ 11 in both green
and red**, the finest-grained Soviet colour stock in the database; six dye-impurity ratios;
colour separation 85/35/20, 15/80/25, 0/15/115 all ±5; any single layer ≥ 80.

They differ in **antihalation construction only** — ЛН-9 a colloidal-silver undercoat,
ЛН-9С a carbon-black lacquer counter-layer on the back — and the specification quantifies
the consequence: **ЛН-9С's whole Dmin ladder sits 0.05–0.10 D lower** (1.00/0.55/0.25 versus
1.10/0.60/0.30). A rear carbon layer suppresses halation without adding density on the
image-bearing side. That makes the pair a **controlled A/B on antihalation construction from
a single document**, which is why they were worth adding as two stocks rather than one.

### `SVEMA_CO_32D` — ТУ 6-17-912-87, amateur colour reversal
Nominal S 32; general sensitivity **by the reversed image 32–63**; balance 1.3–1.8; contrast
**2.2–2.6 upper layer, 1.8–2.2 middle and lower**; contrast balance lower-to-middle ≤ 0.3;
Dmax ≥ 2.2 and Dmin ≤ 0.25 per layer; **useful exposure interval ≥ 1.2 measured between
densities 0.3 and 2.1**; resolving power ≥ 68 lin/mm; the colloidal-silver antihalation
layer is specified to **decolourise** during processing.

⚠ **Its contrast figures are ranges 0.4 wide and the stored gammas are midpoints** (b 2.4,
g = r 2.0). That is a much weaker commitment than the ±0.04 tolerances the negative TUs
give, and it was the basis for arguing against this stock. It is stated in the profile.
The TU also specifies **no granularity and no MTF at all** for this film, unlike every
negative TU in the batch — those remain [T3] class estimates.

## What none of these documents provides

No spectral sensitivity curves, no characteristic-curve shapes (so every `toe_*`/`shoulder_*`
remains [T3]), no reciprocity data, and no Dmax for the negative stocks.

## Verification

`verify.py` **134 PASS / 2 FAIL** — the same two long-standing failures, and their output is
**byte-identical** to before this change, which answers the question I raised before adding:
three new colour negatives with estimated dye matrices neither deepened nor masked the
saturation-hierarchy and neighbour-pair ordering checks.

Three new guards assert what the specifications actually establish: ЛН-9С's Dmin ladder must
sit below ЛН-9's; ЛН-9 must remain finer-grained and sharper than ЛН-8; ЦО-32Д must be
reversal with a σ(D) that turns over past mid-scale. The reversal-stock count moves 33 → 34.
C++ regenerated and compiles clean; 148 names in `film_names.txt`.
