PYTHON SOURCE  --  2026-09-01e
===============================
53 modules.

  film_profiles.py    THE DATABASE.  Hand-maintained; everything else reads it.
  film_sim.py         the reference renderer -- one definition of every law
  cpp_codegen.py      emits the generated C++ database archive
  verify.py           493 assertions over the database and the laws
  cpp_parity.py       drives the real C++ stages against film_sim
  interimage_parity.py
  build.py            verify -> codegen -> sync -> audit -> docs -> compile
  gen_active_profiles.py, gen_film_curves_md.py    documentation generators

  ~35 source readers (agfa_*.py, kodak_*.py, bbc_t101_2.py, mees_callier_q.py,
  trumpy_callier_q.py, jp_jps_1965_269.py, flueckiger_2018.py, ...) --
  each re-derives its adopted numbers from the original PDF on every build.

build.py takes ~8 minutes.  The PDF corpus is NOT included (it is your own
document collection); readers whose source is absent report [SKIP] and the
build stays green.
