# Official-source web hunt — 2026-08-16

The owner asked whether the red (estimated) values in `FilmActiveProfiles.md` could be
collected through official channels. The answer is measured rather than asserted: the hunt
started with the gap class most likely to be published — whole stocks carrying no
documentation at all.

## Retrieved and entered

| Stock | Official source | What it gave |
|---|---|---|
| `KENTMERE_PAN_100` | HARMAN technology Ltd, "KENTMERE PAN 100 — Technical Information", 4 pp, July 2022, fetched from **ilfordphoto.com** | ISO 100/21° to daylight (ID-11, 20 °C, intermittent agitation, spiral tank); base **0.125 mm / 5-mil acetate**; reciprocity **Ta = Tm^1.26** beyond 1 s, none 1 s…1/10 000 s; 11-developer × 3-EI development matrix; handle in total darkness |
| `KENTMERE_PAN_400` | same series, PAN 400 sheet | ISO 400/27°; same base; **Ta = Tm^1.30**; 11-developer × EI 320/400/800 matrix |
| `KONICA_VX_100` | Konica, "Konica Color VX 100 Film (IMPROVED) — Technical Data Sheet", via the 125px datasheet mirror | **Diffuse RMS granularity 4** (48 µm aperture, 12×, D-min+1.0, Status M); **resolving power 63 lines/mm at 1.6:1 and 125 at 1000:1**; achromatic reciprocity (+1 stop at 10 s, no CC filter); full nine-layer structure; CNK-4 / C-41 |
| `KONICA_CENTURIA_SUPER_400` | own sheet located (csuper400.pdf) + adjacent VX SUPER 400 sheet | Front matter only — ISO 400/27°, triacetate, DX 26-5, emulsion #400–#499, MCC/UCC crystal technology, CNK-4. **The data-table page did not survive text extraction**, so RMS and resolving stay [C3]. VX SUPER 400's own numbers (RMS 4, resolving 50/100) are recorded in provenance and explicitly **not** back-applied — VX SUPER and CENTURIA SUPER are different Konica families |

`ProcessingSpec` entered for both Kentmere stocks (ILFORD ID-11 stock, 20 °C, the sheets'
own agitation regime). `contrast_index` deliberately left at 0.0: neither sheet prints a
contrast-index or gamma figure anywhere.

**"No documentation of any kind" list: 23 → 21 → 18 stocks** across today's two passes.

## What the hunt says about the red cells

Three of the four retrieved sheets **agreed exactly** with values the database already
carried — the Schwarzschild exponents 0.794 = 1/1.26 and 0.769 = 1/1.30, the RMS 4, the
resolving 63/125. That is the useful result: those estimates were sound, and they are now
citations rather than estimates. Nothing was altered to match a source, and nothing was
invented where a source was silent — both Kentmere sheets print no granularity, no
resolving power, no MTF, no characteristic curve and no spectral data at all, and that
absence is recorded against the stocks rather than filled in.

## Still open, with the route for each

- **ORWO NC24 and the UT family** — archive.org does host VEB Filmfabrik Wolfen material
  (an ORWOCOLOR NC 19 booklet and an ORWO NP 22 datasheet are confirmed present). A
  targeted archive.org fulltext sweep for the ORWO Handbuch / Farbenfibel is the next step.
- **Konica reversal stocks** (Chrome Centuria 100, Chrome R100) — the 125px mirror carries
  the negative line only; try Wayback snapshots of Konica Japan product pages.
- **Never-published parameter classes** — dye impurity matrices, DIR/interimage
  coefficients, Callier coefficients, layer thicknesses. These were never in datasheets in
  any era, so the route is patents (Google Patents, class **G03C**, assignee Eastman Kodak
  or Fuji Photo Film — worked emulsion examples disclose layer structure and coupler
  chemistry) and the photographic-science literature (*Photographic Science and
  Engineering*, *Journal of Photographic Science*, Mees & James 4th ed. 1977). Queued in
  `next_week_task.md`.

## Incident recorded during this session

`NotFound.md` and the two dated CHANGES files written earlier in the session disappeared
from the working folder. Cause identified: files **newly created through the shell** in the
mounted folder are not retained, while edits to existing files are. `NotFound.md` was
restored intact from the 09:22 ZIP and re-edited; the CHANGES files were rewritten through
the file tool. Rule for future sessions: create new files with the file tool, not from the
shell.
