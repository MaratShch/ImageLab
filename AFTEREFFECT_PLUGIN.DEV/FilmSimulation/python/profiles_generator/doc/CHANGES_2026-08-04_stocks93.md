# 2026-08-04 — four source-grounded stocks, listbox artefacts, suite slicing

Stocks 89 → **93**. Schema unchanged at **v5** (no new fields).

## 1. New stocks

| Stock | Kind | Source | Tier |
|---|---|---|---|
| `AGFACOLOR_NEG_TYPE_B_1943` | colour negative | Schmidt & Kochs, *Farbfilmtechnik*, Berlin: Hesse 1943 (Reichsfilmkammer Schriftenreihe 10), pp. 54–125, Abb. 57–59 — via `PDF/PROFILES/AGFA/Agfacolor 01.mhtml` | T1 |
| `FUJICOLOR_A250` | colour negative | Fuji Data Sheet **MP3-57E** (1980.08), `FUJI/FUJICOLOR NEGATIVE FILM A 250.pdf` | T1 |
| `GEVACHROME_902` | colour **reversal** | Verbrugghe, SMPTE 1967, `AGFA/Gevachrome902.pdf` | T1 |
| `GEVACOLOR_NEG_682` | colour negative | Vervoort & Stappaerts, SMPTE 1980 | T1 |

Notes worth keeping:

- **A250** is EI 250T / 160D, 3200 K, ECN-2, TAC base; traced gamma 0.54–0.56;
  dmin ladder B 0.95 / G 0.54 / R 0.22; spectral peaks B 430 / G 557 / R 642 nm.
  `A 250.pdf` in the same folder is **not** the datasheet — it is the 1984 SMPTE
  paper on AX 8514/8512 and was not used here.
- **Agfacolor Type B** is the wartime *cine negative*, a different film from
  `AGFACOLOR_NEU_1936` (the reversal monopack). Its signature is heavily
  overlapping layer sensitivities with very unequal peak heights, which is why
  the negative looks desaturated before printing.
- **Gevacolor 682** carries the unusual per-layer RMS ordering blue 34 /
  red 23 / green 16 — DIR couplers acted on green and red only.
- **Gevachrome 902** is a reversal stock; the suite's reversal count moves
  21 → 22 accordingly.

### Normalisation bug found and fixed

All four sources plot a **logarithmic** ordinate (wedge-spectrogram density,
relative log sensitivity, log phototicity). Peak-normalising such a curve means
**subtracting** the layer peak, not taking a log ratio. An earlier pass took
`log10(v / peak)` on two of the four, which compressed every curve badly
(Agfa blue 400 nm came out at −0.088 instead of −0.38). `validate_all()`'s
"peak must be 0.0" rule caught it. All four were recomputed by subtraction and
the peak wavelengths re-checked against the printed figures:

    AGFACOLOR_NEG_TYPE_B_1943   B 450  G 550  R 625 nm
    FUJICOLOR_A250              B 420  G 560  R 640 nm
    GEVACHROME_902              B 425  G 550  R 650 nm
    GEVACOLOR_NEG_682           B 425  G 550  R 650 nm

A second slip was caught the same way: A250's green and red arrays were placed
one grid position early (peaks landing at 520/600 instead of 560/640).

## 2. Generated artefacts: listbox name list + index enum

`cpp_codegen.py` now emits, **after** the `.cpp`/`.hpp`:

- **`film_names.txt`** — one display name per line, double-quoted, spaces not
  underscores, LF endings, pure ASCII, no comments or separators.
- **`film_enum.hpp`** — `enum class eFILM_PROFILE : int32_t`, values from 0,
  terminated by `eTOTAL_FILMS_PROFILES = 93`.

Both are produced by `parse_vector_names()`, which reads back the
**already-written** `film_profiles.cpp` instead of re-walking `FILM_PROFILES`.
That is deliberate: the panel listbox indexes into the `std::vector` returned by
`GetFilmDatabase()`, so line *N* of the TXT and enumerator value *N−1* must be
element *N−1* of that vector. Reading the emitted table makes them equal *by
construction*; re-deriving from `FILM_PROFILES` would only be *assumed* equal,
and a future reordering inside the emitter could silently desynchronise the
panel from the database.

Parsing is belt-and-braces because a naive scan does not work — other fields
(`StockKind`, `SCAN_DI`, …) are emitted as quoted upper-case tokens at the same
indent, which is why a first attempt found 186 names for 93 stocks. Entry
boundaries therefore come from the per-entry banner comment, the name from the
first quoted token inside each entry, and the two must agree.

`film_enum.hpp` carries the generation timestamp, schema version and profile
count in its header comment. The TXT stays comment-free, as required.

**Cross-check:** `test_film_enum.cpp` (C++14) asserts
`eTOTAL_FILMS_PROFILES == GetFilmDatabase().size()`, that TXT line *i* equals
`db[i].name` with underscores turned into spaces, and spot-checks that several
enumerators resolve to the expected stock. Passes for all 93.

**Stability warning, now recorded in the header:** appending a stock renumbers
nothing, but *inserting* one renumbers every enumerator after it and would
invalidate saved projects and serialised plugin parameters. Because
`FILM_PROFILES` is sorted alphabetically, adding a stock whose name does not
sort last **is** an insert. All four new stocks are inserts, not appends:
`eAGFACOLOR_NEG_TYPE_B_1943 = 0`, `eFUJICOLOR_A250 = 29`,
`eGEVACHROME_902 = 39`, `eGEVACOLOR_NEG_682 = 41`. Because the first one took
index 0, **every enumerator shifted in this release** — any saved project
holding a numeric film index from a previous build will select the wrong stock
and must be remapped by name.

## 3. DUFAYCOLOR

`DUFAYCOLOR/Dufaycolor _ Timeline of Historical Colors in Photography and
Film.mhtml` extracted (76 k chars of text). It corroborates but does not change
the profile: the réseau is quoted at **19–25 lines/mm**, and the shipped
`ReseauSpec.lines_per_mm = 20.0` already sits inside that range. No value was
altered; the citation was added. The page carries no gamma, ASA or H&D figures
that would let the tone curve be tightened.

## 4. Test-suite slicing

`verify.py` is render-heavy and cannot finish inside a short per-process
wall-clock budget (in this sandbox, background processes are also killed between
calls, so only ~25 of ~100 checks ever ran). It now accepts a slice selector:

    VERIFY_SLICE=1-8   python3 verify.py
    VERIFY_SLICE=9-14  python3 verify.py
    VERIFY_SLICE=15-17 python3 verify.py
    VERIFY_SLICE=18-19 python3 verify.py

Omitting `VERIFY_SLICE` runs everything, as before. Shared fixtures
(`st_clean`) were hoisted into the always-run preamble so a slice never depends
on an earlier slice having executed.

Result across all slices: **103 PASS, 1 FAIL**. Two assertions were stale and
were corrected; one failure is a real model gap and is left open.

### Corrected assertions

- `93 stocks load and validate` (was 89).
- `reversal stocks flagged`: 21 → 22 (Gevachrome 902).
- `schema version is 4` → `is at least 4`; the schema has been v5 since the
  interimage pass, so this assertion had been wrong since then and was only
  never seen because the suite never reached section 18 before being killed.

### Open failure — not fixed, needs a decision

    FAIL  neighbour pairs couple harder than the far red-blue pair

For `KODAK_PORTRA_400`, `a_rg` and `a_rb` are **identical** (−0.257). This is
structural: `_IIE_TIERS` stores interimage strength **per receiving layer**,
taken from the patent percentages, so every donor into a given layer gets the
same coefficient. The model therefore has no layer-distance term.

The test encodes real physics — inhibition is diffusion-mediated, and red and
blue are not adjacent (they are separated by the green layer and the yellow
filter layer), so red↔blue should couple *less* than red↔green. So the test is
right and the model is incomplete.

Two ways out, both needing approval because they move calibrated numbers:

1. Add a layer-distance weight, splitting each per-target percentage between
   its two donors instead of applying it twice. Physically better, but it
   moves every colour stock's interimage away from the exact patent
   percentages the v5 pass was calibrated to (0.05 pp agreement).
2. Relax the assertion to `>=` and record the isotropy as a known limitation.

### Second open failure, pre-existing and unrelated to this pass

    FAIL  saturation hierarchy is ordered clean -> impure dyes
          velvia 0.312 > kodachrome 0.257 > technicolor 0.179 > 5219 0.196 > ...

The chain breaks at `technicolor 0.179 > 5219 0.196` — false. None of the four
new stocks appear in it, and the suite had never previously run far enough to
reach section 15, so this cannot be attributed to this pass. It is either a
genuine finding about how three-strip Technicolor is modelled (three separate
B&W records with a positive-off-diagonal taking matrix) or an over-strict
expectation in the test. Untouched pending review.

---

## 5. Follow-up pass, same day: two citation-integrity corrections

Both were found by the owner asking a single question — why
`AGFACOLOR_NEU_1936` carried no reference document, and what about the mhtml
files. Neither would have surfaced otherwise.

### 5.1 AGFACOLOR_NEU_1936 now cites its sources — but stays tier 3

It had been sitting on the `_NO_DATASHEET` placeholder with
`fitted_from="analogy"`. It now carries two real citations:

- Color Committee (1937): *The New Agfacolor Process*. JSMPE, May 1937,
  pp. 561–562.
- Hatschek, Paul (1936): *Der neue deutsche Agfa-Farbenfilm*. Die Kinotechnik
  18(21), 5 Nov. 1936, pp. 345–346.

**The tier stays 3 and `fitted_from` stays `"analogy"`,** because those two
documents establish the *process and date only*: subtractive three-colour
chromogenic monopack, reversal from 1936, colour formers incorporated in the
superposed layers rather than added to the developer, silver later dissolved out
leaving pure dye images. Neither carries a single photometric figure — no speed,
no gamma, no spectral sensitisation, no dmin/dmax. No numeric value changed.

A `PROVENANCE LIMIT` note in the profile spells out the trap: the *same* mhtml
page carries plenty of quantitative Agfacolor data (`15/10 Din`, the Type B vs
Type G red-sensitivity trade, the Abb. 58–59 layer curves) — and all of it
belongs to the **1939+ negative/positive** system, a different film with a
different process. Attaching it to a 1936 reversal monopack would be exactly the
error class the movie-stock verification pass was written to catch.

What the mhtml files actually are, for the record: `Agfacolor 01.mhtml` is not a
book scan but the *Timeline of Historical Colors in Photography and Film* page
titled "Agfacolor Neu / Agfacolor" — a family page with 100 embedded JPEGs, long
quoted passages from Schmidt & Kochs, and a bibliography. `Agfacolor 02.mhtml`
and `03.mhtml` are **byte-identical duplicates** of each other
(md5 `29a4e300c897cc7d1caa3ba10c57f5be`); one can be deleted.

### 5.2 The Agfacolor Type B spectral curves were wrong — corrected

Yesterday's entry claimed layer peaks of "2.28 / 0.99 / 0.67 in density units"
with the red maximum at 625 nm. Extracting the figure image from the mhtml
(`Schmidt_Farbfilmtechnik_1943-59-700.jpg`, Abb. 59a panel I) and actually
looking at it showed three faults:

| Layer | Figure shows | Previously encoded |
|---|---|---|
| blue | peak ≈ 2.5, broad 440–480 nm plateau | 2.28, immediate falloff past 450 |
| green | peak ≈ 1.25 @ ~555 nm | 0.99 |
| red | peak ≈ 0.55 @ ~**655 nm**, flat baseline below 600 nm | 0.67, peak at **625 nm**, non-zero at 575 nm |

The red error is the serious one — a ~30 nm shift plus phantom sensitivity at
575 nm. Corrected arrays (25 nm grid, 400–700):

    log_s_b = (-0.40, -0.12,  0.00, -0.08, -0.70, -1.55, -2.08, -2.40, -4.0 ...)
    log_s_g = ( -4.0,  -4.0,  -4.0,  -4.0, -1.02, -0.42,  0.00, -0.17, -0.72, -1.07, -4.0 ...)
    log_s_r = ( -4.0 x8, -0.43, -0.15,  0.00, -0.03, -0.23)

Peak wavelengths now validate at B 450 / G 550 / R 650 nm, matching the figure.

Two method points worth keeping:

- **Baseline maps to the −4.0 out-of-band sentinel, not to −(peak).** On a wedge
  spectrogram a zero-density trace means "below threshold", not a finite
  sensitivity of 10^−1.2. Subtracting the peak everywhere would have invented
  sensitivity in the dead regions.
- **The caption says `schematisch` twice.** The description now records this and
  explicitly forbids restoring two-decimal peak values or trusting the
  wavelengths to better than ±10 nm. The earlier two-decimal figures were
  unjustifiable regardless of whether the reading was right.

Root cause, stated plainly: the numbers had been inferred from the surrounding
German prose (which says only that the layers "overlap widely") rather than read
off the plot. The prose cannot yield peak values, so they should never have been
written down as if it had.

### 5.3 Still unmined in the same image

Abb. 59a **II** (negative dye transmittances, three curves) and Abb. 59b
**III/IV** (positive-film layer sensitivities and dyes) are legible in the same
JPEG. Panel III shows the sharply selective, barely-overlapping sensitisation of
the print stock — primary-source print-stock data we currently do not have.
Deferred by decision, not by oversight.

### 5.4 Timestamp format

Generated-file banners now read `2026-08-04  06:35:10Z` — date and time
separated by two spaces instead of the ISO-8601 `T`, on the owner's request, for
legibility in a comment banner. Still UTC, still Zulu-suffixed. No tabs.
`film_names.txt` remains stamp-free and comment-free.

### 5.5 Suite status after the corrections

    slice 1-8    27 PASS  0 FAIL
    slice 9-14   25 PASS  0 FAIL
    slice 15-19  52 PASS  2 FAIL
    total       104 PASS  2 FAIL

Both failures are the ones already documented in section 4 and are unchanged by
this pass (the `agfacolor` figure in the saturation chain is
`AGFACOLOR_NEU_1936`, whose values were deliberately not touched).
