"""Export Intergraph Smart 3D MDB2 label data without starting Access."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

try:
    from access_parser import AccessParser
except ImportError as error:
    raise SystemExit(
        "Missing dependency: install it with "
        "python -m pip install -r requirements-vue-mdb.txt"
    ) from error


LINKAGE_COLUMNS = [
    "linkage_index",
    "DMRSLinkage",
    "link_key_1",
    "link_key_2",
    "link_key_3",
    "link_key_4",
    "label_file_index",
    "file_name",
    "moniker",
]

LONG_LABEL_COLUMNS = [
    "linkage_index",
    "moniker",
    "file_name",
    "label_line_number",
    "label_name_index",
    "label_name",
    "label_value_index",
    "label_value",
    "label_value_numeric",
    "extended_label",
]


def table_rows(table: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Convert access-parser's column-oriented result into row dictionaries."""
    columns = list(table)
    if not columns:
        return []
    row_count = len(table[columns[0]])
    return [
        {column: table[column][row_index] for column in columns}
        for row_index in range(row_count)
    ]


def text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.hex()
    return str(value)


def sort_number(value: Any) -> tuple[int, str]:
    try:
        return int(value), ""
    except (TypeError, ValueError):
        return sys.maxsize, text(value)


def write_csv(path: Path, columns: Iterable[str], rows: Iterable[dict[str, Any]]) -> None:
    fieldnames = list(columns)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: text(row.get(column, "")) for column in fieldnames})


def unique_index(
    rows: list[dict[str, Any]], column: str
) -> tuple[dict[Any, dict[str, Any]], int]:
    result: dict[Any, dict[str, Any]] = {}
    duplicates = 0
    for row in rows:
        key = row.get(column)
        if key in result:
            duplicates += 1
        result[key] = row
    return result, duplicates


def _prepare_label_join(
    linkage_rows: list[dict[str, Any]],
    label_rows: list[dict[str, Any]],
    label_name_rows: list[dict[str, Any]],
    label_value_rows: list[dict[str, Any]],
):
    linkage_by_id, duplicate_linkage_indexes = unique_index(linkage_rows, "linkage_index")
    names_by_id, duplicate_name_indexes = unique_index(label_name_rows, "label_name_index")
    values_by_id, duplicate_value_indexes = unique_index(label_value_rows, "label_value_index")

    labels_by_linkage: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for label in label_rows:
        labels_by_linkage[label.get("linkage_index")].append(label)

    label_header_by_id: dict[Any, str] = {}
    used_header_keys: set[str] = set()
    for label_name in sorted(
        label_name_rows,
        key=lambda row: sort_number(row.get("label_name_index")),
    ):
        label_name_index = label_name.get("label_name_index")
        base_header = text(label_name.get("label_name")).strip() or f"label_{label_name_index}"
        header = base_header
        suffix = 2
        while header.casefold() in used_header_keys:
            header = f"{base_header} [{label_name_index}]"
            if header.casefold() in used_header_keys:
                header = f"{base_header} [{label_name_index}]__{suffix}"
                suffix += 1
        used_header_keys.add(header.casefold())
        label_header_by_id[label_name_index] = header

    maximum_occurrences: dict[Any, int] = {}
    for labels in labels_by_linkage.values():
        occurrences = Counter(label.get("label_name_index") for label in labels)
        for label_name_index, occurrence_count in occurrences.items():
            maximum_occurrences[label_name_index] = max(
                maximum_occurrences.get(label_name_index, 0), occurrence_count
            )

    property_columns: list[str] = []
    for label_name in sorted(
        label_name_rows,
        key=lambda row: sort_number(row.get("label_name_index")),
    ):
        label_name_index = label_name.get("label_name_index")
        base_header = label_header_by_id[label_name_index]
        occurrence_count = maximum_occurrences.get(label_name_index, 0)
        property_columns.extend(
            base_header if occurrence == 1 else f"{base_header}__{occurrence}"
            for occurrence in range(1, occurrence_count + 1)
        )

    missing_linkage = sum(
        label.get("linkage_index") not in linkage_by_id for label in label_rows
    )
    missing_name = sum(label.get("label_name_index") not in names_by_id for label in label_rows)
    missing_value = sum(label.get("label_value_index") not in values_by_id for label in label_rows)
    line_equals_name = sum(
        label.get("label_line_number") == label.get("label_name_index") for label in label_rows
    )
    dmrs_mismatch = sum(
        text(linkage.get("DMRSLinkage"))
        != " ".join(text(linkage.get(f"link_key_{index}")) for index in range(1, 5))
        for linkage in linkage_rows
    )
    summary = {
        "linkage_rows": len(linkage_rows),
        "label_rows": len(label_rows),
        "label_name_rows": len(label_name_rows),
        "label_value_rows": len(label_value_rows),
        "linkages_with_labels": len(labels_by_linkage),
        "linkages_without_labels": len(linkage_rows) - len(labels_by_linkage),
        "labels_per_linkage": dict(
            sorted(
                Counter(len(labels) for labels in labels_by_linkage.values()).items(),
                key=lambda item: item[0],
            )
        ),
        "missing_label_linkage_foreign_keys": missing_linkage,
        "missing_label_name_foreign_keys": missing_name,
        "missing_label_value_foreign_keys": missing_value,
        "duplicate_linkage_indexes": duplicate_linkage_indexes,
        "duplicate_label_name_indexes": duplicate_name_indexes,
        "duplicate_label_value_indexes": duplicate_value_indexes,
        "label_line_number_equals_label_name_index": line_equals_name,
        "dmrs_linkage_text_mismatches": dmrs_mismatch,
        "file_name_counts": dict(Counter(text(row.get("file_name")) for row in linkage_rows)),
    }
    return labels_by_linkage, names_by_id, values_by_id, label_header_by_id, property_columns, summary


def _iter_joined_rows(linkage_rows, labels_by_linkage, names_by_id, values_by_id,
                      label_header_by_id, compact=False):
    """Join one linkage at a time; compact mode omits ordinary label copies."""
    for linkage in sorted(
        linkage_rows,
        key=lambda row: sort_number(row.get("linkage_index")),
    ):
        linkage_index = linkage.get("linkage_index")
        labels = sorted(
            labels_by_linkage.get(linkage_index, []),
            key=lambda row: (
                sort_number(row.get("label_line_number")),
                sort_number(row.get("label_name_index")),
            ),
        )
        wide_row = {column: linkage.get(column, "") for column in LINKAGE_COLUMNS}
        long_rows = []
        occurrence_by_name: Counter[Any] = Counter()
        for label in labels:
            label_name_index = label.get("label_name_index")
            label_value_index = label.get("label_value_index")
            label_name = names_by_id.get(label_name_index, {}).get("label_name", "")
            label_value = values_by_id.get(label_value_index, {}).get("label_value", "")
            occurrence_by_name[label_name_index] += 1
            occurrence = occurrence_by_name[label_name_index]
            base_header = label_header_by_id.get(
                label_name_index,
                f"label_{label_name_index}",
            )
            wide_row[base_header if occurrence == 1 else f"{base_header}__{occurrence}"] = label_value
            # The business path needs exceptional numeric/extended labels only.
            numeric = values_by_id.get(label_value_index, {}).get("label_value_numeric", "")
            has_value = label_value is not None and (
                not isinstance(label_value, str) or bool(label_value.strip()))
            needs_detail = bool(label.get("extended_label")) or (
                not has_value and numeric not in (None, 0, 0.0))
            if compact and not needs_detail:
                continue
            long_rows.append(
                {
                    "linkage_index": linkage_index,
                    "moniker": linkage.get("moniker", ""),
                    "file_name": linkage.get("file_name", ""),
                    "label_line_number": label.get("label_line_number", ""),
                    "label_name_index": label_name_index,
                    "label_name": label_name,
                    "label_value_index": label_value_index,
                    "label_value": label_value,
                    "label_value_numeric": numeric,
                    "extended_label": label.get("extended_label", ""),
                }
            )
        yield wide_row, long_rows



def build_export_rows(linkage_rows, label_rows, label_name_rows, label_value_rows):
    labels, names, values, headers, columns, summary = _prepare_label_join(
        linkage_rows, label_rows, label_name_rows, label_value_rows)
    wide_rows, long_rows = [], []
    for row, details in _iter_joined_rows(linkage_rows, labels, names, values, headers):
        wide_rows.append(row)
        long_rows.extend(details)
    return wide_rows, long_rows, columns, summary


def read_database_tables(database_path: Path):
    parser = AccessParser(str(database_path))
    tables = {
        name: table_rows(parser.parse_table(name))
        for name in ("linkage", "labels", "label_names", "label_values")
    }
    return tables, dict(access_parser_version=parser.version, page_size=parser.page_size)


def read_business_database(database_path: Path):
    """Stream linked properties without materializing CSV wide/long tables."""
    tables, metadata = read_database_tables(database_path)
    labels, names, values, headers, columns, summary = _prepare_label_join(
        tables['linkage'], tables['labels'], tables['label_names'], tables['label_values'])
    summary.update(metadata)
    rows = _iter_joined_rows(tables['linkage'], labels, names, values, headers, compact=True)
    return rows, columns, summary


def read_database(database_path: Path):
    tables, metadata = read_database_tables(database_path)
    wide_rows, long_rows, property_columns, summary = build_export_rows(
        tables["linkage"], tables["labels"], tables["label_names"], tables["label_values"])
    summary.update(metadata)
    return tables, wide_rows, long_rows, property_columns, summary


def export_database(database_path: Path, output_dir: Path) -> dict[str, Any]:
    tables, wide_rows, long_rows, property_columns, summary = read_database(database_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "elements.csv", LINKAGE_COLUMNS + property_columns, wide_rows)
    write_csv(output_dir / "labels_long.csv", LONG_LABEL_COLUMNS, long_rows)
    label_name_columns = list(tables["label_names"][0]) if tables["label_names"] else []
    label_value_columns = list(tables["label_values"][0]) if tables["label_values"] else []
    write_csv(output_dir / "label_names.csv", label_name_columns, tables["label_names"])
    write_csv(output_dir / "label_values.csv", label_value_columns, tables["label_values"])
    summary.update(
        {
            "database": str(database_path.resolve()),
            "output_dir": str(output_dir.resolve()),
        }
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    argument_parser = argparse.ArgumentParser(
        description="Export Smart 3D MDB2 linkage and label data without launching Access."
    )
    argument_parser.add_argument("database", type=Path, help="Path to a .mdb2/.mdb file")
    argument_parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        help="Output directory; defaults to <database>_mdb_export",
    )
    return argument_parser.parse_args()


def main() -> int:
    arguments = parse_args()
    database_path = arguments.database.resolve()
    if not database_path.is_file():
        print(f"Database not found: {database_path}", file=sys.stderr)
        return 2
    output_dir = arguments.output_dir or database_path.with_name(
        f"{database_path.stem}_mdb_export"
    )
    summary = export_database(database_path, output_dir.resolve())
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
