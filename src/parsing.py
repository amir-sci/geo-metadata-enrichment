from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Any

import pandas as pd
from lxml import etree
from shapely.geometry import box

from .config import (
    INPUT_XML_DIR,
    PARSED_DIR,
    RAW_DIR,
    PARSING_ERRORS_FILE,
    RAW_CANONICAL_FILE,
    SUMMARY_FILE,
    PREFERRED_ABSTRACT_TAGS,
    PREFERRED_KEYWORD_TAGS,
    PREFERRED_PLACE_TAGS,
    TARGET_FIELDS,
)
from .utils import ensure_dir, normalize_text, save_dataframe


def _local_name(el) -> str:
    return etree.QName(el).localname.lower()


def _xpath_local(root, names: tuple[str, ...] | list[str]):
    checks = {n.lower() for n in names}
    return [el for el in root.iter() if _local_name(el) in checks]


def _first_text(root, names: tuple[str, ...] | list[str]) -> str | None:
    for el in _xpath_local(root, names):
        text = normalize_text(el.text)
        if text:
            return text
    return None


def _all_texts(root, names: tuple[str, ...] | list[str]) -> list[str]:
    values = []
    seen = set()
    for el in _xpath_local(root, names):
        text = normalize_text(el.text)
        if text and text.lower() not in seen:
            seen.add(text.lower())
            values.append(text)
    return values


def _bbox_from_root(root):
    west = _first_text(root, ["westbc", "west"])
    east = _first_text(root, ["eastbc", "east"])
    north = _first_text(root, ["northbc", "north"])
    south = _first_text(root, ["southbc", "south"])
    if all([west, east, north, south]):
        try:
            west_f, east_f, north_f, south_f = map(float, [west, east, north, south])
            geom = box(west_f, south_f, east_f, north_f)
            centroid = geom.centroid
            return [west_f, south_f, east_f, north_f], float(centroid.y), float(centroid.x)
        except Exception:
            return None, None, None
    return None, None, None


def _is_nonempty(value):
    if isinstance(value, list):
        return len([x for x in value if normalize_text(x)]) > 0
    if value is None:
        return False
    if isinstance(value, str):
        s = value.strip()
        return s not in {"", "[]", "None", "nan", "NaN"}
    return bool(value)


def parse_xml_file(xml_path: str | Path) -> dict[str, Any] | None:
    xml_path = Path(xml_path)
    parser = etree.XMLParser(recover=False, huge_tree=True, resolve_entities=False, remove_blank_text=True)
    try:
        root = etree.parse(str(xml_path), parser).getroot()
    except Exception as exc:
        return {
            "filename": xml_path.name,
            "source_path": str(xml_path),
            "valid": False,
            "parse_error": str(exc),
        }

    title = _first_text(root, ["title"])
    abstract = _first_text(root, list(PREFERRED_ABSTRACT_TAGS))
    # Use thematic keywords for keyword completion, plus preserve raw place keywords separately.
    thematic_keywords = _all_texts(root, list(PREFERRED_KEYWORD_TAGS))
    place_names = _all_texts(root, list(PREFERRED_PLACE_TAGS))
    bbox, centroid_lat, centroid_lon = _bbox_from_root(root)

    return {
        "filename": xml_path.name,
        "source_path": str(xml_path),
        "valid": True,
        "parse_error": None,
        "title": title,
        "abstract": abstract,
        "purpose": _first_text(root, ["purpose"]),
        "keywords": thematic_keywords,
        "place_names": place_names,
        "bbox": bbox,
        "centroid_lat": centroid_lat,
        "centroid_lon": centroid_lon,
        "origin": _all_texts(root, ["origin"]),
        "datacred": _first_text(root, ["datacred"]),
        "topic_category": _all_texts(root, ["topiccategory", "theme", "themekey"]),
        "language": _first_text(root, ["language"]),
        "lineage": _first_text(root, ["lineage", "procdesc"]),
        "contact": _all_texts(root, ["cntorg", "cntper", "cntpos", "cntvoice", "cntemail"]),
        "metadata_date": _first_text(root, ["metd", "caldate", "pubdate", "procdate"]),
    }


def parse_all_xml(input_dir: str | Path | None = None, output_dir: str | Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    input_dir = Path(input_dir or INPUT_XML_DIR)
    output_dir = Path(output_dir or PARSED_DIR)
    ensure_dir(output_dir)
    ensure_dir(RAW_DIR)

    xml_files = sorted(glob.glob(str(input_dir / "*.xml")))
    rows = []
    invalid = []
    for file_path in xml_files:
        parsed = parse_xml_file(file_path)
        if parsed and parsed.get("valid"):
            rows.append(parsed)
        else:
            invalid.append(parsed or {"filename": Path(file_path).name, "source_path": file_path, "valid": False, "parse_error": "unknown"})

    valid_df = pd.DataFrame(rows)
    invalid_df = pd.DataFrame(invalid)

    # Serialize list-like columns to JSON strings for CSV compatibility.
    for col in ["keywords", "place_names", "origin", "topic_category", "contact"]:
        if col in valid_df.columns:
            valid_df[col] = valid_df[col].apply(lambda x: x if isinstance(x, str) else pd.Series([x]).iloc[0])
    # Save outputs
    save_dataframe(valid_df, output_dir / "canonical_raw.parquet")
    save_dataframe(valid_df, output_dir / RAW_CANONICAL_FILE)
    save_dataframe(invalid_df, output_dir / PARSING_ERRORS_FILE)

    summary = pd.DataFrame([
        {
            "total_xml_files": len(xml_files),
            "valid_xml_files": int(len(valid_df)),
            "invalid_xml_files": int(len(invalid_df)),
            "with_title": int(valid_df["title"].notna().sum()) if "title" in valid_df else 0,
            "with_abstract": int(valid_df["abstract"].notna().sum()) if "abstract" in valid_df else 0,
            "with_keywords": int(valid_df["keywords"].apply(_is_nonempty).sum()) if "keywords" in valid_df else 0,
            "with_place_names": int(valid_df["place_names"].apply(_is_nonempty).sum()) if "place_names" in valid_df else 0,
            "with_bbox": int(valid_df["bbox"].notna().sum()) if "bbox" in valid_df else 0,
        }
    ])
    save_dataframe(summary, output_dir / SUMMARY_FILE)
    return valid_df, invalid_df
