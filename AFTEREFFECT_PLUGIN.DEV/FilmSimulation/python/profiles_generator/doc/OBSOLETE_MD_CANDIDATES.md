# Candidates for deletion — for your approval. Nothing has been deleted.

You asked for a list of obsolete MD/TXT files. I ran a reference scan over the whole
project rather than judging by filename or date: for every doc file, how many other `.md`,
`.MD`, `.py` or `.txt` files mention it by name, plus a byte-identical duplicate check.

**The honest headline: the doc set is in better shape than "many obsolete files" would
suggest.** Only three files are safe to delete, and all three are exact duplicates. Four
files have *zero inbound references* and I am recommending you **keep** them anyway —
reasons below, because "nothing links to it" turned out not to mean "nothing is in it".

---

## A. Safe to delete — verified byte-identical duplicates (≈1.13 MB)

`FilmDatabase_Charecteristics.MD` exists in **three** places with the same MD5
(`1429122f…`), and the Russian mirror in **two** (`26de0124…`):

| Path | Size | Action |
|---|---|---|
| `PYTHON/profile_generator/doc/FilmDatabase_Charecteristics.MD` | 302,822 | **KEEP** — this is the one the generators and the CHANGES files cite |
| `doc/FilmDatabase_Charecteristics.MD` | 302,822 | delete (duplicate) |
| `FilmDatabase_Charecteristics.MD` (project root) | 302,822 | delete (duplicate) |
| `PYTHON/profile_generator/doc/FilmDatabase_Charecteristics_Rus.MD` | 550,155 | **KEEP** |
| `doc/FilmDatabase_Charecteristics_Rus.MD` | 550,155 | delete (duplicate) |

All four files that cite `FilmDatabase_Charecteristics` by name
(`CHANGES_2026-08-13_cheltsov1958.md`, `CHANGES_2026-08-13_spectral_path.md`,
`CHANGES_2026-08-14b_fuji_kodak_websites.md`, `CHANGES_2026-08-15_neopan1600.md`) live in
`PYTHON/profile_generator/doc/`, so keeping that copy keeps every citation resolvable.

⚠ One thing to decide, not for me to decide: with **no version control**, three copies are
also three chances of divergence. They are identical *today*. If any workflow of yours
writes to the root or `doc/` copy, deleting it will silently redirect that workflow. If you
are unsure, the zero-risk version of this is to delete the root copy only and keep `doc/`.

## B. Zero inbound references, but I recommend KEEPING — unique content

| File | Refs | Why not to delete |
|---|---|---|
| `CHANGES_2026-08-16b_queue_p1.md` | 0 | I had this on the list, then read it. It is the **origin document for method rule 9** (`toe_k ≤ shoulder_k ≤ 2·toe_k`), and it carries two things the condensed rule does not: why `verify.py`'s monotonicity tolerance is divided by each curve's own gamma (a flat Dmax shelf multiplies ulp noise by gamma, so a fixed allowance is either too loose for gamma-1 or too tight for gamma-11), and 2383's per-layer **absolute** LAD log H values 1.097 / 0.754 / 0.445. Delete it and the next person to touch that tolerance rediscovers it the hard way |
| `CHANGES_2026-08-16e_svema_gost_norms.md` | 0 | Records a **conflict** — Zhurba 1990 Table 2 against the stored SVEMA Foto values — and the reasoning for not adopting it. Your own method rules say a conflict is *recorded, never averaged*; deleting the record defeats the rule. **The real defect here is the missing citation, not the file.** Suggest adding a pointer from `DIGITIZATION_QUEUE.md` §2 or the SVEMA provenance instead |
| `CHANGES_59stocks.txt` | 0 | Opens with its own status note: *"historical changelog, kept verbatim"*, and warns that the `SOVCOLOR_DS_4` in it never entered the database. That warning is the value — it stops someone re-adopting a superseded [T3] reconstruction |
| `SVEMA64_MY.txt` | 0 | Same pattern: *"kept as a measurement source record… Do not adopt values from this file directly."* It is the raw 290-frame analyzer output behind `SVEMA_FN_64`, superseded by the 355/509-frame batches. Deleting it deletes the provenance of a stock that is still in the database |

## C. Everything else — not candidates

The remaining 42 files in `PYTHON/profile_generator/doc/` each have **at least one live
inbound citation** (from 1 to 22). Deleting any of them breaks a reference that some other
document makes by name. The most-cited are `DIGITIZATION_QUEUE.md` (22),
`Found.md` (19), `NotFound.md` (18), `FilmActiveProfiles.md` (15), `FilmCurves.md` (13),
`next_week_task.md` (13).

The seven engine docs under `doc/` (`AVX2_OPTIMISATION_RULES.md`,
`AVX2_OPTIMISATION_2026-08-11.md`, `D1_TYPE_ALIGNMENT_2026-08-11.md`, `DEFECT_LAYER.md`,
`MEMORY_OPTIMISATION.md`, `SINGLE_THREAD_OPTIMISATION.md`, `STAGE_FUSION_PROPOSAL.md`) are
C++/AVX2 records, untouched by this change and outside its scope.

---

## D. A staleness problem worth more than the deletions

Not a deletion candidate — a correctness one, found while packaging. The **project-root
copies of the generated C++ were a generation behind** the generator's:

| File | root | `PYTHON/profile_generator/` |
|---|---|---|
| `film_profiles.cpp` | 625,831 bytes | 675,798 bytes |
| `film_names.txt` | MD5 `d20cdfea…` | MD5 `4fc19216…` |
| `film_enum.hpp` | MD5 `b09e5470…` | MD5 `f966555f…` |

So whatever last built from the root directory built an **older database** than the one
`verify.py` was validating. `CHANGES_2026-08-16b` says "both copies synced", so the drift is
more recent than that. The regenerated set in this archive's `CPP/scalar/` and `CPP/AVX2/`
is identical in both folders (verified by diff), which fixes it for this delivery — but the
underlying trap remains: two copies, no version control, and nothing that checks they agree.
**Suggest a `verify.py` check comparing the two copies' hashes**, so the next divergence
fails loudly instead of shipping quietly. Your call — I have not added it.
