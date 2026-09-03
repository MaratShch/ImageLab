# CHANGES 2026-08-13 — renames, gauge-variant retirement, natural ordering

Owner-requested (2026-08-12 evening, executed 2026-08-13 morning). Database
100 → 98 stocks. All three changes are breaking for saved projects: every
`eFILM_PROFILE` value after index 26 moved (renames shift alphabetical
position AND the ordering rule changed). The planned enum-ID freeze — stable
per-stock integers with display order decoupled — remains queued and would
have prevented this class of break.

## 1. SVEMA_FN_64 → SVEMA_FOTO_65; gauge variants retired

* Basis: owner statement, supported by Gurlev 1986 p296 already cited in the
  profile — the USSR standard defines Foto-65 (С 65 GOST) as the same
  emulsion as the FN-64 cine designation. FN-64 kept in `aliases`
  ("fn64", "fn-64", "svema fn64"), so presets and searches still resolve.
* `SVEMA_FN_64_16MM` and `SVEMA_FN_64_8MM` DELETED. They existed only
  because TemporalSpec lives on FilmProfile and the schema has no per-gauge
  slot; the emulsion data was identical by design. Gauge is now solely the
  format control's job, per the owner's design decision of 2026-08-12
  ("one representation in the database; final film properties computed in
  correspondence to the width parameter").
* Their 7 genuinely gauge-dependent TRANSPORT values (weave x/y/corner,
  flicker pct/hz, dirt rate, native fps) are PRESERVED, not lost:
  `_GAUGE_TRANSPORT_PRESERVED` in film_profiles.py, keyed by FORMAT name,
  unconsumed, awaiting the transport-on-format schema change.
* Side effect, deliberate: the f50 drift between the three copies
  (35.0 corrected on the 35 mm entry per GOST 24876-81, 34.0 stale on the
  variants) and the resolving-power drift (110 vs sentinel 0) are gone with
  the copies. The surviving entry carries the corrected values.

## 2. SVEMA_TSNL_32 / _65 → SVEMA_CNL_32 / _65

* Owner-requested transliteration of ЦНЛ. Note this is a Latin-lookalike
  transliteration, not phonetic (ISO 9 / BGN give "TS"); chosen by the
  owner with that stated. All "tsnl" spellings kept in `aliases`; the
  Cyrillic mark stays in the gen_film_names.py comment.

## 3. EIGHT_MM_BW / EIGHT_MM_COLOR → GENERIC_BW / GENERIC_COLOR

* The gauge is not a property of the emulsion; these are generic amateur
  reversal stocks and the gauge comes from the format control. Their grain
  and MTF figures were verified to be physical (reversal-chain reasoning),
  not perceptual gauge tuning, before the rename. `default_format` stays
  "8mm" (their historically dominant gauge) as the default only. All old
  aliases kept.

## 4. Ordering: alphabetical → natural (numeric-aware)

* `FILM_PROFILES` is now sorted with digit runs compared as numbers:
  SVEMA_FOTO_32 < FOTO_65 < FOTO_130 < FOTO_250 (owner-requested order) and
  AGFA_APX_25 < APX_100 < APX_400. Plain alphabetical had put FOTO_130
  first. This ordering defines the enum, so it is part of the break above.

## Known limitation exposed by the collapse (pre-existing, now visible)

`_coating_for` derives edge fog and gate-buckle gauge factors from
`default_format` — the profile's DEFAULT, not the RENDERED format. The
retired 8 mm variant got edge fog because its default was "8mm";
SVEMA_FOTO_65 (default super35) rendered at 8 mm now gets none. This is
exactly the Appendix B.4.10 defect: gauge-derived properties must follow
the width control. Fix deferred to the transport-on-format design change
(owner discussion pending). verify.py's narrow-gauge edge-fog test now uses
GENERIC_BW (default 8mm, reversal — expectation inverted: fog DARKENS a
positive) and passes.

## Verification

* verify.py: 104 PASS / 2 FAIL — both failures pre-existing, byte-identical
  against the pristine backup (saturation hierarchy; red-blue interimage
  pair). Zero regressions from this change. Count check updated 100 → 98.
* Generated film_profiles.hpp/cpp, film_enum.hpp, film_names.txt refreshed
  and copied to the C++ root; film_profiles.cpp compiles; AlgorithmMain.cpp
  passes -fsyntax-only against the new tables.
* Stale name references swept: zero in .py (outside preserved-history
  comments), zero in engine code (three comments updated with the rename
  noted); historical CHANGES_*/REPORT_* docs left as written — they record
  events under the names of their time.
* doc/FilmActiveProfiles.md and doc/FilmCurves.md regenerated (98 stocks).
