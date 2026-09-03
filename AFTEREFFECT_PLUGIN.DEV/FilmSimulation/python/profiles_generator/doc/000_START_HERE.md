# Documentation snapshot — 2026-08-31

**Read in this order. The first three are current state; everything else is either
per-stock reference or a dated audit trail.**

| # | File | What it answers |
|---|---|---|
| 1 | **`PROGRESS.md`** | *Where are we?* One screen at the top: build state, database counts, live coverage table, engine status, queue census. Everything below that heading is dated history. |
| 2 | **`DIGITIZATION_QUEUE.md`** | *What is left, and what is in the way of each item?* §0 is the snapshot, §3 is the census and the only authority for the row counts. |
| 3 | **`NotFound.md`** | *What is missing, and what would close it?* A research checklist: one screen of open gaps, then a per-stock acquisition plan. |
| 4 | `FilmActiveProfiles.md` | *Per-stock coverage.* **Generated on every build** — every cell is the value the simulator actually uses, marked measured / specification-limit / estimate. Never hand-edited. |
| 5 | `FilmCurves.md` | Generated. Every stored characteristic curve, per stock. |
| 6 | `README.md` | The model itself: the 25-stage pipeline, **Implementation status** (Python / scalar C++ / AVX2), the file map, the verification gates, known limits. Its dated `Status <date>` entries are history. |
| 7 | `AVX2_OPTIMISATION_RULES.md`, `SINGLE_THREAD_OPTIMISATION.md`, `MEMORY_OPTIMISATION.md`, `STAGE_FUSION_PROPOSAL.md`, `D1_TYPE_ALIGNMENT_2026-08-11.md` | The engine-side playbooks. All single-thread. |
| 8 | `EMULSION_KNOWLEDGE_BASE.md`, `DEFECT_LAYER.md` | Domain reference behind the model. |
| 9 | `RESULT_*.md`, `CHANGES_*.md`, `REVIEW_*.md`, `ASSESSMENT_*.md`, `REPORT_*.md` | Dated batch records. **Snapshots of their day, kept verbatim including wrong turns, because the reasoning is the audit trail.** Never current state. |
| 10 | `DIGITIZATION_QUEUE_history.md`, `NotFound_history.md` | Superseded revisions of #2 and #3. Provenance of a fix, never state. |

## The state, in five lines

- **Build green.** `verify.py` 461 PASS / 1 FAIL (a deliberate baseline). 28 audits, 23 run
  here, all green. `g++ -Wall -Wextra` clean on 18 TUs with zero bytes of output.
- **Database:** 165 film stocks, 11 print stocks, 14 gauges, schema **v22**.
- **Engine:** 25 stage entry points, and **all 25 exist in both the scalar reference and the
  AVX2 production path**.
- **Queue:** 102 rows, 78 closed, **24 live** — 9 wait on the owner, 6 on a document proved
  absent, 5 are ordinary work, 4 on a model decision.
- ⚠ **The largest gap is not a document.** Nothing checks a render against a photograph.

## What changed in this pass, and why some numbers moved

`DIGITIZATION_QUEUE.md` §0/§3 and `NotFound.md`'s front matter were **rewritten from the
parse, not appended to**; the archaeology they carried went to the two `*_history.md` files.
Three corrections came out of doing it:

1. ⚠ **`SCHEMA_VERSION` read 18 and the database was at v22.** Four additive versions landed
   on 2026-08-30/31 with their fields commented and the constant never bumped, so every
   document repeating "schema v18" was wrong. The version is now registered in
   `doc_consistency.py` for three documents, so a repeat fails the build.
2. ⚠ **`README.md`'s pipeline table said 15 steps and the pipeline has 25.** Re-derived from
   the `AlgoStageNN*` symbols in the tree.
3. ⚠ **Four stocks were counted as having no source while holding traced curves.** The PORTRA
   160NC/160VC/400NC/400VC quartet was created from E-190 pp 9-12 and no provenance entry was
   written, so it fell through to the `_NO_DATASHEET` placeholder. The no-source count went
   13 -> 9.
