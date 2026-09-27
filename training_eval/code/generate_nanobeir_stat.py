#!/usr/bin/env python3
import argparse
import csv
import re
import zipfile
from datetime import datetime, timezone
from html import escape
from pathlib import Path


TASK_LINE = re.compile(r"^(Nano\S+Retrieval)\s+([-+]?\d+(?:\.\d+)?)\s*$")


def size_key(name: str):
    match = re.search(r"(\d+(?:\.\d+)?)([kKmM]?)", name)
    if not match:
        return (float("inf"), name)
    value = float(match.group(1))
    unit = match.group(2).lower()
    if unit == "k":
        value *= 1_000
    elif unit == "m":
        value *= 1_000_000
    return (value, name)


def parse_summary(path: Path) -> dict[str, float]:
    scores = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = TASK_LINE.match(line.strip())
        if match:
            scores[match.group(1)] = float(match.group(2))
    if not scores:
        raise ValueError(f"No Nano*Retrieval task scores found in {path}")
    scores["mean"] = sum(scores.values()) / len(scores)
    return scores


def find_summaries(root: Path, run_dir: str) -> list[Path]:
    return sorted(root.glob(f"*/{run_dir}/summary.md"), key=lambda p: size_key(p.parents[1].name))


def cell_ref(row_index: int, col_index: int) -> str:
    name = ""
    while col_index:
        col_index, remainder = divmod(col_index - 1, 26)
        name = chr(65 + remainder) + name
    return f"{name}{row_index}"


def write_csv(path: Path, table: list[list[object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows(table)


def write_xlsx(path: Path, table: list[list[object]], sheet_name: str = "NanoBEIR") -> None:
    # Minimal dependency-free XLSX writer for a single worksheet.
    strings = []
    string_to_index = {}

    def shared_string_index(value: object) -> int:
        text = str(value)
        if text not in string_to_index:
            string_to_index[text] = len(strings)
            strings.append(text)
        return string_to_index[text]

    def sheet_xml() -> str:
        rows_xml = []
        for row_idx, row in enumerate(table, start=1):
            cells_xml = []
            for col_idx, value in enumerate(row, start=1):
                if value == "" or value is None:
                    continue
                ref = cell_ref(row_idx, col_idx)
                if isinstance(value, (int, float)):
                    cells_xml.append(f'<c r="{ref}"><v>{value}</v></c>')
                else:
                    cells_xml.append(f'<c r="{ref}" t="s"><v>{shared_string_index(value)}</v></c>')
            rows_xml.append(f'<row r="{row_idx}">{"".join(cells_xml)}</row>')
        col_count = max((len(row) for row in table), default=1)
        cols_xml = "".join(
            f'<col min="{idx}" max="{idx}" width="{"28" if idx == 1 else "16"}" customWidth="1"/>'
            for idx in range(1, col_count + 1)
        )
        return (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            f"<cols>{cols_xml}</cols><sheetData>{''.join(rows_xml)}</sheetData>"
            "</worksheet>"
        )

    created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    worksheet_xml = sheet_xml()
    shared_strings_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        f'count="{len(strings)}" uniqueCount="{len(strings)}">'
        + "".join(f"<si><t>{escape(text)}</t></si>" for text in strings)
        + "</sst>"
    )

    files = {
        "[Content_Types].xml": (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>'
            '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
            '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            '<Override PartName="/xl/sharedStrings.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml"/>'
            '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
            "</Types>"
        ),
        "_rels/.rels": (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
            '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>'
            "</Relationships>"
        ),
        "docProps/core.xml": (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
            'xmlns:dc="http://purl.org/dc/elements/1.1/" '
            'xmlns:dcterms="http://purl.org/dc/terms/" '
            'xmlns:dcmitype="http://purl.org/dc/dcmitype/" '
            'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
            "<dc:creator>generate_nanobeir_stat.py</dc:creator>"
            f'<dcterms:created xsi:type="dcterms:W3CDTF">{created}</dcterms:created>'
            f'<dcterms:modified xsi:type="dcterms:W3CDTF">{created}</dcterms:modified>'
            "</cp:coreProperties>"
        ),
        "xl/workbook.xml": (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            f'<sheets><sheet name="{escape(sheet_name)}" sheetId="1" r:id="rId1"/></sheets>'
            "</workbook>"
        ),
        "xl/_rels/workbook.xml.rels": (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
            '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
            '<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/sharedStrings" Target="sharedStrings.xml"/>'
            "</Relationships>"
        ),
        "xl/worksheets/sheet1.xml": worksheet_xml,
        "xl/sharedStrings.xml": shared_strings_xml,
        "xl/styles.xml": (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            '<fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>'
            '<fills count="1"><fill><patternFill patternType="none"/></fill></fills>'
            '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
            '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
            '<cellXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/></cellXfs>'
            "</styleSheet>"
        ),
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for filename, content in files.items():
            zf.writestr(filename, content)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a nanobeir-stat.csv-style table from multiple summary.md files."
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=".",
        help="Directory containing folders like msmarco-10k/train-qwen3_8b-merged/summary.md",
    )
    parser.add_argument(
        "-r",
        "--run-dir",
        default="train-qwen3_8b-merged",
        help="Run subdirectory name that contains summary.md",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="nanobeir-stat.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--xlsx-output",
        default=None,
        help="Output XLSX path. Defaults to the CSV output path with .xlsx suffix.",
    )
    parser.add_argument(
        "--no-xlsx",
        action="store_true",
        help="Only write CSV, without the XLSX copy.",
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    summaries = find_summaries(root, args.run_dir)
    if not summaries:
        raise SystemExit(f"No summary.md found under {root}/*/{args.run_dir}/")

    columns = [path.parents[1].name for path in summaries]
    parsed = {path.parents[1].name: parse_summary(path) for path in summaries}

    task_names = sorted(
        {task for scores in parsed.values() for task in scores if task != "mean"}
    )
    rows = task_names + ["mean"]

    output = Path(args.output).expanduser()
    if not output.is_absolute():
        output = root / output

    table = [["field", *columns]]
    for row_name in rows:
        table.append(
            [
                row_name,
                *[
                    "" if row_name not in parsed[column] else parsed[column][row_name]
                    for column in columns
                ],
            ]
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(output, table)
    print(f"Wrote {output}")

    if not args.no_xlsx:
        xlsx_output = (
            Path(args.xlsx_output).expanduser()
            if args.xlsx_output
            else output.with_suffix(".xlsx")
        )
        if not xlsx_output.is_absolute():
            xlsx_output = root / xlsx_output
        write_xlsx(xlsx_output, table)
        print(f"Wrote {xlsx_output}")


if __name__ == "__main__":
    main()
