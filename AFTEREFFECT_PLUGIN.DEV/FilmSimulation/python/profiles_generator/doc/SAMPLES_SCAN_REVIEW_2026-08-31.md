# The owner's SAMPLES scans — inventory and what they can and cannot support

**Reviewed 2026-08-31**, 854 files, ~1.1 GB, read in place on the owner's machine. Nothing was
copied into the corpus. This document exists so that **the conclusions survive the files**.

✅ **Disposition, 2026-08-31: the owner moved the scans to separate storage.** They are archived, not
lost, so nothing below depends on them remaining on the working machine. Everything measurable was
measured before they moved; this document is the record.

---

## 1. Two provenance classes, and the split governs everything

| folder | files | genuine scanner output | web-sized re-encodes |
|---|---|---|---|
| `SVEMA-FOTO-64` | 509 | **509** | 0 |
| `ORWO-UT18` | 78 | **37** | 41 |
| `ORWO-NC21` | 109 | 0 | 109 |
| `TASMA-FN64` | 132 | 0 | 132 |
| `SVEMA-FOTO-250` | 26 | 0 | 26 |
| **total** | **854** | **546** | **308** |

**Scanner output** is 4416 × 2944 with EXIF `Make GCMC / Model Scanner / Software UF15 16/08/20
v0.69` — 36.0 mm across 4416 px = **122.7 px/mm**, the pitch the 2026-08-18 Svema work established.

⚠ **The other 308 are ≤ 1216 px wide, carry no EXIF at all, and come in 78 different sizes.** That is
not this scanner, which emits one size with metadata every time. They have been resized and
re-encoded by something, and their provenance is unknown.

**67 files are exact byte-duplicates** of another file in the same folder (the `(1)`, `(2)` copies).
No cross-folder duplicates: the `PICT0179`–`PICT0215` range appears in both `SVEMA-FOTO-64` and
`ORWO-UT18` as **different images**, so the two folders are separate scanning sessions with
independently-restarted numbering.

## 2. ⚠ Most of the monochrome material has had its colour destroyed

Frames in which R == G == B at **every pixel**:

| folder | neutral | of | share |
|---|---|---|---|
| `SVEMA-FOTO-250` | 26 | 26 | **100 %** |
| `SVEMA-FOTO-64` | 411 | 509 | 81 % |
| `TASMA-FN64` | 104 | 132 | 79 % |
| `ORWO-UT18` | 0 | 78 | 0 % |
| `ORWO-NC21` | 0 | 109 | 0 % |

A bit-exactly neutral frame **cannot evidence a colour cast**: it returns zero by construction. Every
`silver_tone` and `base_tint` measurement ever attempted on these Soviet batches was, to that extent,
measuring an artefact of a greyscale conversion.

## 3. What was measured

### D3 — answered, and the stored value did not survive

See the closed D3 row and the note at `TASMA_FN_64.silver_tone`. In brief: 104/132 neutral; the 28
colour-bearing frames give a midtone cast of **R−G = +7.72 ± 10.72**, scatter larger than the mean;
the historical "+8.6 and +15.6" pair are two draws from that same distribution and the larger was
chosen. **`TASMA_FN_64.silver_tone` reverted +0.30 → 0.00.**

⚠ The statistic the queue named was also wrong for the job. `max |R−G|` over a pictorial frame
measures the most saturated **object in the scene**, not the emulsion; over `ORWO-NC21` it returns
137, correctly reporting that a colour negative is a colour photograph.

### The film base is present, unclipped, and merely un-referenced

**50 of the SVEMA-FOTO-64 scanner frames** carry a uniform rebate strip at the right edge, **226 px
wide = 1.85 mm** at this pitch, identical in width frame to frame:

| | |
|---|---|
| strip level | **250.24 ± 2.91** (8-bit) |
| pixels clipped at ≥ 254 | **0.2 %** — so it is *not* clipped |
| density vs scanner white 255 | **0.0082 D**, sd **0.0051 D**, range 0.0035–0.0340 D |

⚠ This **reproduces the documented "0.008–0.028 D relative to scanner white" from the pixels**, on 50
frames, and adds a number the project did not have: **the scanner's per-frame auto-exposure
contributes ±0.005 D of base-level scatter.** That is the noise floor beneath every density reading
from this rig, and it is why the σ(D) estimator was found uninterpretable.

⚠ **The consequence for D1, and a correction to the first version of this note.** It said the base
was "already measured; only the white reference is missing", and that one empty-gate scan would make
all 50 strips absolute **retroactively**. ⚠ **The measurement above refutes that.** The strip of ONE
physical film base ranges **235.8 to 252.9** across the 50 frames, and two frames of the same base
read 241 and 250. A single piece of base does not vary by 9 levels — so the UF15 is re-exposing per
frame, this batch has no single white point, and no later gate frame can calibrate it. The gate frame
must be taken **in the same session as the scans it is meant to calibrate**, and its first job is to
settle whether this scanner has a fixed-exposure mode at all.

### A consistent cast on the Svema scanner frames, and why it cannot be adopted

The 96 colour-bearing SVEMA-FOTO-64 scanner frames give midtone **R−G = −5.03 ± 5.01** and **G−B =
+10.46 ± 1.83**. ⚠ The G−B term is stable to ±1.8 across 96 frames from one device, which is what a
real emulsion tone would look like — **and equally what the scanner's own white balance would look
like.** Without an empty-gate reference the two are not separable, which is exactly why
`SVEMA_FOTO_65`'s tone was reverted to identity on 2026-08-18. Not adopted.

### ⚠ No calibration target exists anywhere in the 854 files

No empty gate, no grey card, no step wedge. Searched every file for a uniform frame and for uniform
extreme edge bands; the lowest-detail frame in the set has std 3.7 at mean 162, which is a
low-contrast photograph. **D1 and D2 remain blocked, and no re-analysis of this material can unblock
them.**

## 4. Retention

| material | verdict |
|---|---|
| **546 scanner frames** | ⚠ **Downgraded the same day, from "keep" to "keep only if the film is gone."** The retroactive-absolute argument for keeping them does not survive the correction above. Every conclusion they support is recorded in this document; the anisotropy question they might serve is blocked on D2, which they do not contain. **If the physical film still exists, a fresh session — gate frame included, fixed exposure if the scanner offers one, no greyscale conversion — is worth more than these files and they may go.** |
| **308 web-sized files** | **Keep, lower priority.** Useless for measurement, but the only ORWO NC21 and UT18 material the project has, and useful as visual reference for grain character and dye fade. |
| **67 byte-duplicates** | **Safe to delete.** |

## 5. What went into the database

- `TASMA_FN_64.silver_tone` **+0.30 → 0.00**, with the measurement recorded at the value.
- `TASMA_OCH_45.silver_tone` **+0.15 flagged, not changed** — now the last nonzero silver tone in the
  database, carrying no source. There are no OCh-45 scans, and refuting a value by analogy is what
  the 2026-08-18 pass refused to do for `SVEMA_FOTO_32` and `SVEMA_FOTO_130`.
- Queue **D3 closed**; **D1 rewritten** with the measured base level, the ±0.005 D noise floor, and
  the fact that `--empty-gate` was never an implemented flag.
