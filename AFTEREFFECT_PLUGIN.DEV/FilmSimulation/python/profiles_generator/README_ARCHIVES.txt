Three archives, 2026-08-29 (second session) -- the spectral_weights work.

  1_python.zip        generator, database definition, audits and guards
                      (34 Python + 2 readme txt + the 25 generated C++/txt
                      artefacts, so the two parity audits run out of the box)
  2_generated_db.zip  the generated database in the FilmProfile/ layout
  3_docs.zip          all Markdown, reviewed against the live database

NO SCALAR OR AVX2 ARCHIVE IN THIS DELIVERY, AND THAT IS DELIBERATE
  No algorithm source was edited. The change is in the Python renderer's
  default, in the provenance records, and in the report generator. The C++
  algorithm already did the right thing -- see below, which is the main finding.

WHAT CHANGED, IN ONE PARAGRAPH
  The owner asked why AGFA_APX_100 and APX_400 showed a red, estimate-marked
  "Spectral Sensitivity" cell when Agfa's sheets carry those curves. They were
  never ignored: both were vector-traced on 2026-08-17 to 0.50 nm and 0.0034
  log. That column prints spectral_weights -- the RGB collapse triple -- not the
  curve, which is four columns to its right. Chasing it turned up three things:
  a Python/C++ divergence that had been shipping, 48 provenance records
  asserting a derivation nobody had run, and a guard measuring itself on a grid
  that had already discarded the evidence it was testing for.

  Full account: doc/RESULT_2026-08-29c_spectral_weights.md

THE ONE THAT MATTERS
  Algo_07_Sim.cpp calls AlgoSpectralMonoWeights() UNCONDITIONALLY and always
  has. film_sim gated the identical derivation behind RenderSettings.
  spectral_mono, which defaulted to False. For the 24 monochrome stocks that
  carry a traced pan curve the plugin derived and the reference renderer did
  not -- two engines, two different black-and-white images, both plausible.
  Worst case KODAK_PLUS_X_125, blue weight 0.110 stored against 0.502 derived.
  spectral_mono now defaults to True, and spectral_mono_parity.py compiles the
  plugin's own translation unit and compares all 68 monochrome stocks on every
  build: 67 agree exactly.

  NOTHING IN THE C++ NEEDS REBUILDING FOR THIS. The plugin was already
  rendering the derived weights. What changed is that the Python reference now
  agrees with it.

NO spectral_weights VALUE CHANGED
  Not one literal in the database moved. What changed is which value the
  renderer reads (the curve, not the literal, on 24 stocks), and what the
  provenance records say about it.

CONSISTENCY CHECK PERFORMED BEFORE ARCHIVING
  * film ordering        IDENTICAL across all four representations, verified
                         programmatically: FILM_PROFILES (161), film_names.txt
                         (161), film_enum.hpp (161 + TOTAL_FILMS_PROFILES), and
                         the 161 profile blocks emitted across the 16
                         film_profiles_data_*.cpp. Nothing was re-sorted.
  * film_names.txt       md5 41e0bc5d2c7db82324529e773f2fd5ee -- UNCHANGED from
                         the file the owner supplied. No ListBox index moves.
  * verify.py            422 PASS / 1 FAIL (the FAIL is the saturation
                         hierarchy the owner said to leave alone -- baseline)
  * build.py             20 audits registered, 14 pass, 5 skip for sources not
                         staged in this checkout, 1 fails (see below)
  * generated files      all 25 content-identical between the generator
                         directory and the project root, differing only in the
                         generation timestamp line
  * generated database   18 TUs compile clean under g++ -std=c++14 -Wall
                         -Wextra with ZERO bytes of output
  * doc_consistency      every registered documentation count matches the live
                         database, re-run after the documentation edits
  * no stale artefacts   cpp_codegen.py re-run into a temp dir and compared:
                         no generated file differs except by its timestamp

WHAT IS ACTUALLY DIFFERENT INSIDE 2_generated_db.zip
  Only the 16 film_profiles_data_*.cpp changed in content, and only in their
  ParamSource arrays -- 118 spectral_weights records rewritten by rule. Every
  header, film_names.txt, film_enum.hpp, film_display_order.txt, film_ids.lock,
  LoadFilmDataBase.* and film_profiles.cpp are byte-identical to the 2026-08-29
  morning delivery once the generation timestamp line is ignored. The complete
  FilmProfile/ tree is shipped anyway so the drop-in cannot be half-applied.

ONE KNOWN FAILURE, PRE-EXISTING, NOT FROM THIS WORK
  cpp_parity.py fails on the grain stage: rendered amplitude differs from the
  Python reference by 1.83e-01 against a 2e-05 tolerance, and Algo_11_Sim.cpp
  no longer carries the 'ampScale' marker. AlgoAddGrain computes
  sqrt(max(D - dmin, 0) + fog) with no net-1.0 normalisation, so the figure is
  exactly sqrt(1 + fog_grain) - 1 and rms_granularity does not mean the printed
  number in the C++ render. Queue C30/C33, the recorded FilmGrainSigma bypass.
  Re-confirmed today by running the pristine 1_python.zip copy against the same
  algorithm tree: identical failures. Nothing in this work touches GrainSpec,
  dmax or the grain path.

  ⚠ AND A CORRECTION TO THE MORNING'S DELIVERY NOTE. It reported "0 failures /
  0 warnings -- the first fully green build". That run used a build root with
  no algorithm sources, where cpp_parity SKIPS five of its probes and exits 0.
  Verified against an empty root: five [SKIP] lines, exit 0. The claim is
  withdrawn. Nothing regressed; the instrument had simply not been looking.

TWO NEW QUEUE ROWS, NEITHER FIXED HERE
  C39  ROLLEI_INFRARED_400 stores the UNFILTERED sensitisation (peak 410 nm,
       0.028 of its energy past 700 nm), so no honest gamut guard refuses it and
       both engines now derive a near-flat (0.349, 0.315, 0.336). Its authored
       red-dominant (0.52, 0.20, 0.28) encodes an assumed IR taking filter that
       NO FIELD IN THE PROFILE RECORDS. Needs a taking_filter carrier, not a
       tuned threshold -- 0.028 is below every ordinary panchromatic stock's own
       out-of-reach share, so any threshold catching this one starts refusing
       Tri-X.
  C40  ⚠ A LIVE RENDERING DEFECT IN THE SHIPPED C++, ON ONE STOCK. The
       gamut-reach guard exists only in Python. AlgoSpectralMonoWeights() has no
       peak test and no out-of-reach test, so KONICA_INFRARED_750 renders in the
       plugin at (0.1611, 0.1931, 0.6458) -- BLUE-dominant -- against the
       authored and correct red-dominant (0.55, 0.15, 0.30). The fix is the two
       tests Python already has, ported: ~20 lines, one translation unit, no
       signature change, no AVX2 involvement (the derivation is setup-domain and
       scalar in both builds). NOT DONE HERE because the algorithm sources were
       out of scope; doing it means re-issuing 3_scalar.zip and 4_avx2.zip.
       spectral_mono_parity.py accepts it under --allow-guard-gap AND names it
       in its own [OK] line on every single build, so accepting it cannot
       quietly become forgetting it.
