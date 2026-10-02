"""«Современные фотоматериалы и их обработка» -- the development-time tables,
read by COLUMN and by VESSEL (queue P54, 2026-09-18b).

⚠⚠ THE ROW'S DIAGNOSIS WAS WRONG AND THE WRONG DIAGNOSIS IS WHY 705 VALUES SAT
HELD FOR THREE DAYS. It said: "a table covering TWO films is split between them
by dividing the temperature list in half, and where the two films do not have
the same number of temperature columns that split lands wrong, which shows up
as one temperature appearing twice in a row".

The temperature DOES appear twice. It is not two films. Table 3.241 on p.392 is
one film -- Kodak Professional TRI-X 400 / 400TX -- and the header reads

    Проявитель │      Малый бак        │      Большой бак
               │ 18 20 21 22 24 °С     │ 18 20 21 22 24 °С

so the repeat is the VESSEL: a small tank and a large tank, each with its own
five temperatures. The previous reader saw 18 twice in one row, concluded the
row was structurally misread, and refused the whole table under the
whole-table-or-nothing rule -- which was the right rule applied to a
misunderstanding of what the second block was.

⚠ AND THE FIELD WAS ALREADY THERE. `DevelopmentPoint.vessel` has existed since
the table was first parsed and is `""` on all 861 stored points, because the
reader had nowhere to get it from. Splitting the header at the restart both
recovers the held values AND fills the field, so the same fix that unblocks the
count also makes every point it recovers more specific than the ones already in.

THE SPLIT IS SELF-CHECKING, which is what licenses it:

  * the temperature header must split into blocks that are each STRICTLY
    INCREASING -- a restart is the only place a block may end;
  * the number of blocks must equal the number of vessel captions found above
    the header, or the table is refused;
  * within each (developer, vessel) block the time must FALL as the temperature
    rises, which is the same physical check as before, now applied to the right
    grouping;
  * a table is adopted WHOLE or not at all, unchanged.
"""
import re
import sys
from pathlib import Path

import pymupdf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sovremennye_2004 as S                                  # noqa: E402

#: The vessel captions the book uses, in the order a header may present them,
#: mapped to the short English token stored in `DevelopmentPoint.vessel`.
VESSELS: tuple[tuple[str, str], ...] = (
    ("Малый бак", "small tank"),
    ("Большой бак", "large tank"),
    ("Малые баки", "small tank"),
    ("Большие баки", "large tank"),
    ("Кювета", "tray"),
    ("Бачок", "tank"),
)

#: The vulgar fractions the book sets, as a suffix on the whole number.
FRACTIONS: dict[str, float] = {
    "1/2": 0.5, "1/4": 0.25, "3/4": 0.75, "1/3": 1.0 / 3.0,
    "2/3": 2.0 / 3.0, "1/8": 0.125, "3/8": 0.375, "5/8": 0.625,
    "7/8": 0.875,
}

TEMP = re.compile(r"(\d{2})\s*°?\s*[СC]")
TABLE = re.compile(r"Табл\.\s*(\d+)\.(\d+)")
#: A body cell: an integer with an optional vulgar-fraction suffix.
CELL = re.compile(r"^(\d{1,3})(1/2|1/4|3/4|1/3|2/3|1/8|3/8|5/8|7/8)?$")
#: Words that mean "this cell is empty", not "this cell is zero".
REFUSALS = ("рекомендуется", "рекомен", "дуется", "—", "-")


#: Caption boilerplate, removed before the film name is read.
BOILER = re.compile(
    r"(Режимы\s+проявления|рулонной|листовой|фотопленки|фотоплёнки|"
    r"кинопленки|фотобумаги|пленки|плёнки|Время\s+проявления)", re.I)
#: The trailing condition qualifiers, and what each one is in schema terms.
QUALIFIERS: tuple[tuple[str, str, str, str], ...] = (
    # ⚠ «роторно-барабанном» MAPS ONTO THE EXISTING 'drum' AND NOT ONTO A NEW
    # SPELLING. `DevelopmentPoint.vessel` already carries 'drum' for the same
    # physical process -- a rotating-drum processor. Emitting 'rotary drum'
    # beside it would split one process across two enum values, which is what
    # that field's validator refuses and what it caught on 2026-09-18.
    ("роторно-барабанном", "vessel", "drum", ""),
    ("в кювете", "vessel", "tray", ""),
    ("в кюветах", "vessel", "tray", ""),
    ("в баке", "vessel", "tank", ""),
    ("135 и 120", "format", "", "135+120"),
    ("135 формата", "format", "", "135"),
    ("120 формата", "format", "", "120"),
    ("по Push", "push", "", ""),
    ("Push", "push", "", ""),
)


#: Everything from here on in a caption is the column header running into it.
TAIL = re.compile(r"(,\s*мин|при\s+температуре|Время|Коэфф|Проявитель)")


def _split_caption(text):
    """(film names, qualifier) from a table caption.

    ⚠ A CAPTION MAY NAME TWO FILMS, and that IS the two-film case queue P54 was
    reaching for -- but it is a UNION, not a column split: «Kodak Plus-X Pan
    Professional / PXE и Kodak Plus-X Pan Professional / PXT» is one set of
    times that applies to both stocks. Splitting on « и » gives both names and
    the same points go to each; nothing is divided between them.
    """
    s = TAIL.split(" ".join(BOILER.sub(" ", text).split()))[0]
    qual = ""
    for needle, _kind, _v, _f in QUALIFIERS:
        pos = s.find(needle)
        if pos >= 0:
            qual = s[pos:].strip()
            s = s[:pos].strip()
            break
    s = s.strip(" .,;*")
    parts = [_tidy(q) for q in re.split(r"\s+и\s+", s)]
    return [q for q in parts if len(q) > 3], qual


#: A trailing Russian prepositional phrase is a CONDITION, never part of the
#: name: «... в малом баке», «... в поддонах», «... в роторно-барабанных
#: процессорах». Cut at the first standalone «в».
_PREP = re.compile(r"\s+в\s+.*$|\s+в$")
_LEAD = re.compile(r"^(и\s+|листового\s+формата\s+|рулонных\s+|листовых\s+|"
                   r"Обработка\s+|формата\s+)+", re.I)


def _tidy(name):
    s = _PREP.sub("", _LEAD.sub("", name)).strip(" .,;*")
    s = re.sub(r"\s*\d{3}(\s*\+\s*\d{3})?\s*формата.*$", "", s).strip()
    return s


def stock_of(name):
    """The database stock a caption name means, or None.

    ⚠ THE FIGURE CAPTIONS AND THE TABLE CAPTIONS DO NOT SPELL THE SAME FILM THE
    SAME WAY. `sovremennye_2004.STOCK_MAP` was built from figure captions,
    which drop the house prefix -- «TRI-X 400 / 400TX» -- while a table caption
    carries it in full: «Kodak Professional TRI-X 400 / 400TX». Trying the name
    and then the name with each prefix removed costs nothing and is the
    difference between 3 mapped stocks and all of them.
    """
    cands = [name]
    for pre in ("Kodak Professional ", "Kodak ", "Fuji ", "Fujifilm ",
                "Agfa ", "Ilford ", "Konica "):
        if name.startswith(pre):
            cands.append(name[len(pre):])
    for c in cands:
        st = S.STOCK_MAP.get(c)
        if st:
            return st
    return None


def parse_cell(text):
    """'63/4' -> 6.75. Returns None for anything that is not a time."""
    m = CELL.match(text.strip())
    if not m:
        return None
    whole = float(m.group(1))
    frac = FRACTIONS.get(m.group(2) or "", 0.0)
    # ⚠ A THREE-DIGIT WHOLE NUMBER WITH A FRACTION IS A MISREAD, not a
    # two-hour development: the book sets 10 3/4 as "103/4" and 100 minutes
    # never appears in these tables. Anything over 99 with a fraction is
    # refused rather than guessed at.
    if whole > 99 and frac:
        return None
    return whole + frac


def visual_rows(page, tol=3.5):
    """Words clustered into visual rows, each row sorted left to right."""
    rows = {}
    for x0, y0, x1, y1, txt, *_ in page.get_text("words"):
        if not txt.strip():
            continue
        key = None
        for k in rows:
            if abs(k - y0) <= tol:
                key = k
                break
        if key is None:
            key = y0
            rows[key] = []
        rows[key].append((x0, x1, txt))
    return [(y, sorted(v)) for y, v in sorted(rows.items())]


def split_blocks(temps):
    """Split a temperature header into strictly increasing blocks."""
    blocks, cur = [], []
    for t in temps:
        if cur and t[0] <= cur[-1][0]:
            blocks.append(cur)
            cur = []
        cur.append(t)
    if cur:
        blocks.append(cur)
    return blocks


def read_page(page):
    """Every development table on one page, as dicts, or [] when none."""
    rows = visual_rows(page)
    out = []
    for i, (y, words) in enumerate(rows):
        line = " ".join(w[2] for w in words)
        temps = [(int(m.group(1)), None) for m in TEMP.finditer(line)]
        if len(temps) < 3:
            continue
        # anchor each temperature to the x centre of the word that carries it
        anchors = []
        for x0, x1, txt in words:
            m = TEMP.search(txt)
            if m:
                anchors.append((int(m.group(1)), 0.5 * (x0 + x1)))
        if len(anchors) != len(temps):
            continue
        blocks = split_blocks(anchors)
        # the vessel captions, taken from the two lines above the header
        above = " ".join(w[2] for j in (i - 1, i - 2) if j >= 0
                         for w in rows[j][1])
        vessels = []
        for ru, en in VESSELS:
            pos = above.find(ru.split()[0])
            if pos >= 0 and en not in [v[1] for v in vessels]:
                vessels.append((pos, en))
        vessels = [v[1] for v in sorted(vessels)]
        if len(blocks) != len(vessels):
            # one block and no caption is the ordinary single-vessel table
            if len(blocks) == 1 and not vessels:
                vessels = [""]
            else:
                continue
        # the film name: everything between the nearest «Табл.» caption above
        # and this header, joined, with the caption boilerplate and the
        # trailing CONDITION qualifier taken off. ⚠ The qualifier is not noise:
        # «в кювете» is a tray, «в роторно-барабанном процессоре» a rotary-drum
        # processor, «135 и 120 формата» a film format -- all of them fields
        # this schema already has, and all of them lost by the old reader.
        film, qual = [], ""
        for j in range(i - 1, max(-1, i - 9), -1):
            cap = " ".join(w[2] for w in rows[j][1])
            if not TABLE.search(cap):
                continue
            joined = " ".join(" ".join(w[2] for w in rows[k][1])
                              for k in range(j, i))
            joined = TABLE.sub(" ", joined)
            film, qual = _split_caption(joined)
            break
        body = []
        for y2, w2 in rows[i + 1:]:
            txt = " ".join(w[2] for w in w2)
            if TABLE.search(txt) or TEMP.search(txt):
                break
            cells = [(0.5 * (a + b), t) for a, b, t in w2 if parse_cell(t)]
            if len(cells) < 2:
                continue
            name = " ".join(t for a, b, t in w2 if not parse_cell(t)).strip()
            if not name or any(r in name for r in REFUSALS[:3]):
                name = body[-1][0] if body else ""
            if not name:
                continue
            body.append((name, cells))
        if body:
            out.append(dict(film=film, qual=qual, blocks=blocks,
                            vessels=vessels, body=body))
    return out


def table_points(tab):
    """One table -> points, or None when any physical check fails."""
    flat = [(t, x, v) for v, blk in zip(tab["vessels"], tab["blocks"])
            for t, x in blk]
    pts = []
    for name, cells in tab["body"]:
        got = {}
        for cx, txt in cells:
            best = min(flat, key=lambda q: abs(q[1] - cx))
            if abs(best[1] - cx) > 34:
                return None
            key = (best[2], best[0])
            if key in got:
                return None
            got[key] = parse_cell(txt)
        for vessel in set(k[0] for k in got):
            series = sorted((k[1], v) for k, v in got.items() if k[0] == vessel)
            if len(series) < 2:
                continue
            for (t0, m0), (t1, m1) in zip(series, series[1:]):
                if m1 > m0 + 1e-9:
                    return None
            for t, m in series:
                pts.append((name, vessel, float(m), float(t)))
    return pts or None


def main(argv=()):
    src = S.PDF if S.PDF.is_file() else S.ALT_PDF
    doc = pymupdf.open(src)
    tables = clean = 0
    points = []
    for i in range(doc.page_count):
        for tab in read_page(doc[i]):
            tables += 1
            got = table_points(tab)
            if got is None:
                continue
            clean += 1
            for name, vessel, minutes, celsius in got:
                points.append((i + 1, tab["film"], name, vessel, minutes,
                               celsius))
    vessels = sum(1 for p in points if p[3])
    print("[OK] sovremennye_dev_tables.py -- %d tables read, %d clean, "
          "%d points, %d of them with a vessel"
          % (tables, clean, len(points), vessels))
    if "--dump" in argv:
        for p in points:
            print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
