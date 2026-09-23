"""Small helpers for extracting LSOA rows from Excel workbooks.

The dashboard's runtime requirements include pandas but not an Excel engine.
These helpers therefore read the small subset of XLSX XML needed by the
source-data extracts using only Python's standard library.
"""

from __future__ import annotations

import csv
import json
import posixpath
import re
from pathlib import Path
from typing import Iterable, Iterator
from xml.etree import ElementTree as ET
from zipfile import ZipFile


MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PACKAGE_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"


def _tag(name: str) -> str:
    return f"{{{MAIN_NS}}}{name}"


def _column_number(column: str) -> int:
    """Return a zero-based column number for an Excel column label."""

    number = 0
    for character in column:
        number = number * 26 + ord(character) - ord("A") + 1
    return number - 1


def _column_from_reference(reference: str) -> str:
    match = re.match(r"([A-Z]+)", reference)
    if match is None:
        raise ValueError(f"Invalid Excel cell reference: {reference}")
    return match.group(1)


def _shared_strings(workbook: ZipFile) -> list[str]:
    filename = "xl/sharedStrings.xml"
    if filename not in workbook.namelist():
        return []

    root = ET.fromstring(workbook.read(filename))
    return [
        "".join(text.text or "" for text in shared_string.iter(_tag("t")))
        for shared_string in root.findall(_tag("si"))
    ]


def _sheet_paths(workbook: ZipFile) -> dict[str, str]:
    workbook_root = ET.fromstring(workbook.read("xl/workbook.xml"))
    relationships_root = ET.fromstring(workbook.read("xl/_rels/workbook.xml.rels"))

    relationships = {
        relationship.attrib["Id"]: relationship.attrib["Target"]
        for relationship in relationships_root.findall(
            f"{{{PACKAGE_REL_NS}}}Relationship"
        )
    }

    paths = {}
    for sheet in workbook_root.find(_tag("sheets")):
        relationship_id = sheet.attrib[f"{{{REL_NS}}}id"]
        target = relationships[relationship_id].lstrip("/")
        if not target.startswith("xl/"):
            target = posixpath.join("xl", target)
        paths[sheet.attrib["name"]] = target
    return paths


def _cell_value(cell: ET.Element, shared_strings: list[str]) -> str:
    value = cell.find(_tag("v"))
    cell_type = cell.attrib.get("t")

    if cell_type == "s":
        return shared_strings[int(value.text)] if value is not None else ""
    if cell_type == "inlineStr":
        return "".join(text.text or "" for text in cell.iter(_tag("t")))
    return value.text if value is not None else ""


def iter_worksheet_rows(
    workbook_path: Path, sheet_name: str
) -> Iterator[tuple[int, dict[str, str]]]:
    """Yield row number and a column/value mapping for an XLSX worksheet."""

    with ZipFile(workbook_path) as workbook:
        try:
            worksheet_path = _sheet_paths(workbook)[sheet_name]
        except KeyError as error:
            raise ValueError(f"Worksheet not found: {sheet_name}") from error

        shared_strings = _shared_strings(workbook)
        with workbook.open(worksheet_path) as worksheet:
            for _, row in ET.iterparse(worksheet, events=("end",)):
                if row.tag != _tag("row"):
                    continue

                values = {
                    _column_from_reference(cell.attrib["r"]): _cell_value(
                        cell, shared_strings
                    )
                    for cell in row.findall(_tag("c"))
                }
                yield int(row.attrib["r"]), values
                row.clear()


def lincolnshire_lsoa_codes(geojson_path: Path) -> set[str]:
    """Read the LSOA codes represented by the Lincolnshire GeoJSON."""

    with geojson_path.open(encoding="utf-8") as geojson_file:
        geojson = json.load(geojson_file)

    codes = set()
    for feature in geojson.get("features", []):
        properties = feature.get("properties", {})
        code = (
            properties.get("CODE")
            or properties.get("LSOA_CODE")
            or properties.get("LSOA21CD")
        )
        if code:
            codes.add(str(code).strip())

    if not codes:
        raise ValueError(f"No LSOA codes found in {geojson_path}")
    return codes


def extract_lsoa_rows(
    workbook_path: Path,
    sheet_name: str,
    header_row_number: int,
    lsoa_column: str,
    lsoa_codes: set[str],
) -> tuple[list[str], list[list[str]], set[str]]:
    """Extract rows whose LSOA column is represented in lsoa_codes."""

    headers: list[str] | None = None
    header_columns: list[str] = []
    rows: list[list[str]] = []
    matched_codes: set[str] = set()

    for row_number, values in iter_worksheet_rows(workbook_path, sheet_name):
        if row_number == header_row_number:
            header_columns = sorted(values, key=_column_number)
            headers = [values[column] for column in header_columns]
            if not all(headers):
                raise ValueError(
                    f"Blank header found in {workbook_path.name}, {sheet_name}, "
                    f"row {header_row_number}"
                )
            continue

        if row_number <= header_row_number:
            continue
        if headers is None:
            raise ValueError(
                f"Header row {header_row_number} was not found in {sheet_name}"
            )

        code = values.get(lsoa_column, "").strip()
        if code in lsoa_codes:
            rows.append([values.get(column, "") for column in header_columns])
            matched_codes.add(code)

    if headers is None:
        raise ValueError(
            f"Header row {header_row_number} was not found in {sheet_name}"
        )

    return headers, rows, matched_codes


def write_csv(path: Path, headers: Iterable[str], rows: Iterable[Iterable[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(headers)
        writer.writerows(rows)
