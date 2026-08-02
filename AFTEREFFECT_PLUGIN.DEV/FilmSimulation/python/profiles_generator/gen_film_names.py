"""
Official film-name list generator.

Writes ``film_names.txt``: one official film name per line, each enclosed
in double quotation marks -- no commas, no comments, no extra text. Line
order is EXACTLY the order of the ``std::vector<FilmProfile>`` emitted by
``cpp_codegen.py`` (both iterate the same ``FILM_PROFILES`` tuple, which is
sorted by profile name at module load).

Run this only AFTER ``cpp_codegen.generate()`` has produced and verified
the C++ source; ``run.cmd`` / the release checklist enforce that order.

Official names use spaces as printed by the manufacturer; identifiers'
underscores become spaces, and the Soviet / ORWO stocks whose official
designations are hyphenated (e.g. «Фото-130» -> FOTO-130) are spelled via
the explicit override table below. Transliteration follows the project
convention: З->Z, Л->L, Ц->TS, Ч->CH (СВЕМА ФОТО-32 -> SVEMA FOTO-32,
ТАСМА МЗ-3Л -> TASMA MZ-3L).
"""

from pathlib import Path

from film_profiles import FILM_PROFILES

__all__ = ["generate_names", "OFFICIAL_NAME_OVERRIDES"]

#: identifier -> official printed name. Anything absent falls back to
#: ``name.replace("_", " ")``.
OFFICIAL_NAME_OVERRIDES: dict[str, str] = {
    # Svema (Свема), Shostka. Official designations are hyphenated:
    # «Фото-32/65/130/250», «ДС-4», «ЦНЛ-32/65», «ФН-64».
    "SVEMA_FOTO_32": "SVEMA FOTO-32",
    "SVEMA_FOTO_130": "SVEMA FOTO-130",
    "SVEMA_FOTO_250": "SVEMA FOTO-250",
    "SVEMA_FN_64": "SVEMA FN-64",
    "SVEMA_FN_64_16MM": "SVEMA FN-64 16MM",
    "SVEMA_FN_64_8MM": "SVEMA FN-64 8MM",
    "SVEMA_DS_4": "SVEMA DS-4",
    "SVEMA_TSNL_32": "SVEMA TSNL-32",
    "SVEMA_TSNL_65": "SVEMA TSNL-65",
    # Tasma (Тасма), Kazan: «ФН-64», «ОЧ-45».
    "TASMA_FN_64": "TASMA FN-64",
    "TASMA_OCH_45": "TASMA OCH-45",
    # ORWO, Wolfen: leaflet W 746 prints "ORWO CHROM-FILM UT 18".
    "ORWO_CHROM_UT18": "ORWO CHROM UT18",
}


def official_name(identifier: str) -> str:
    return OFFICIAL_NAME_OVERRIDES.get(identifier, identifier.replace("_", " "))


def generate_names(outdir: Path | str = ".") -> Path:
    out = Path(outdir) / "film_names.txt"
    lines = [f'"{official_name(p.name)}"' for p in FILM_PROFILES]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return out


if __name__ == "__main__":
    path = generate_names()
    print(f"wrote {path} ({len(FILM_PROFILES)} names)")
