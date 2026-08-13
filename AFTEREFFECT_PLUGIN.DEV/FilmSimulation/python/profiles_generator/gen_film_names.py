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
    # 2026-08-13: FN-64 renamed FOTO-65 (same film per the USSR standard;
    # the FN-64 cine designation lives in the aliases), gauge variants
    # retired, and TSNL transliterated CNL (owner request; Cyrillic mark
    # remains in the comment above).
    "SVEMA_FOTO_65": "SVEMA FOTO-65",
    "SVEMA_DS_4": "SVEMA DS-4",
    "SVEMA_CNL_32": "SVEMA CNL-32",
    "SVEMA_CNL_65": "SVEMA CNL-65",
    # Tasma (Тасма), Kazan: «ФН-64», «ОЧ-45».
    "TASMA_FN_64": "TASMA FN-64",
    "TASMA_OCH_45": "TASMA OCH-45",
    # ORWO, Wolfen: leaflet W 746 prints "ORWO CHROM-FILM UT 18".
    "ORWO_CHROM_UT18": "ORWO CHROM UT18",
    # Kodak Data Book 1952: official spellings are hyphenated
    # (Panatomic-X, Tri-X, Ortho-X). Added 2026-08-11.
    "KODAK_VERICHROME_1952": "KODAK VERICHROME (1952)",
    "KODAK_PANATOMIC_X_SHEET_1952": "KODAK PANATOMIC-X SHEET (1952)",
    "KODAK_TRI_X_SHEET_1952": "KODAK TRI-X SHEET (1952)",
    "KODAK_ORTHO_X_SHEET_1952": "KODAK ORTHO-X SHEET (1952)",
    # 2026-08-13 batch: official spellings are hyphenated / spaced.
    "KODAK_TMAX_100": "KODAK T-MAX 100",
    "KODAK_TMAX_400": "KODAK T-MAX 400",
    "KODAK_TMAX_P3200": "KODAK T-MAX P3200",
    "KODAK_TRI_X_400TX": "KODAK TRI-X 400",
    "KODAK_TRI_X_320TXP": "KODAK TRI-X 320",
    "KODAK_PLUS_X_125": "KODAK PLUS-X PAN 125",
    "KODAK_VERICOLOR_III_160": "KODAK VERICOLOR III 160",
    "KODAK_ULTRA_COLOR_100UC": "KODAK ULTRA COLOR 100UC",
    "KODAK_ULTRA_COLOR_400UC": "KODAK ULTRA COLOR 400UC",
    "AGFA_SCALA_200X": "AGFA SCALA 200x",
}


def official_name(identifier: str) -> str:
    return OFFICIAL_NAME_OVERRIDES.get(identifier, identifier.replace("_", " "))


def generate_names(outdir: Path | str = ".", separator: bool = True) -> Path:
    """Write film_names.txt in FILM_PROFILES (database) order.

    Separator (2026-08-13, owner request): by default every line except the
    LAST ends with "|" inside the closing quote -- "NAME|" -- so that the
    lines concatenate (as adjacent C++ string literals) into the single
    pipe-separated string an After Effects popup expects. ``separator=False``
    (--no_separator) restores the previous pipe-free format.
    """
    out = Path(outdir) / "film_names.txt"
    last = len(FILM_PROFILES) - 1
    lines = [
        f'"{official_name(p.name)}{"|" if (separator and i < last) else ""}"'
        for i, p in enumerate(FILM_PROFILES)
    ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Emit the film-name list.")
    ap.add_argument("--no_separator", action="store_true",
                    help="emit WITHOUT the trailing '|' separators "
                         "(the pre-2026-08-13 format)")
    ns = ap.parse_args()
    path = generate_names(separator=not ns.no_separator)
    print(f"wrote {path} ({len(FILM_PROFILES)} names)")
