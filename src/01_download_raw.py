from __future__ import annotations

import argparse
import math
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

from utils import ensure_dir, safe_request, save_metadata, write_outputs

# -----------------------------
# Configuration / constants
# -----------------------------

SSB_API_BASE = "https://data.ssb.no/api/pxwebapi/v2"
SSB_LANG = "en"

SSB_HPI_TABLE_ID = "07221"
SSB_CPI_TABLE_ID = "08981"

SSB_HPI_META_URL = f"{SSB_API_BASE}/tables/{SSB_HPI_TABLE_ID}/metadata"
SSB_HPI_DATA_URL = f"{SSB_API_BASE}/tables/{SSB_HPI_TABLE_ID}/data"

SSB_CPI_META_URL = f"{SSB_API_BASE}/tables/{SSB_CPI_TABLE_ID}/metadata"
SSB_CPI_DATA_URL = f"{SSB_API_BASE}/tables/{SSB_CPI_TABLE_ID}/data"

# Norges Bank: Official policy rate daily download/API (CSV) linked from the "Policy rate" stats page
# Series bundle: OL (overnight lending), SD (policy rate / sight deposit), RR (reserve rate)
NORGESBANK_POLICY_RATE_API_URL = "https://data.norges-bank.no/api/data/IR/B.KPRA.OL%2BSD%2BRR.R"

# Practical guardrails
SSB_MAX_CELLS_SAFETY = 750_000
SSB_REGION_TARGET_MAX = 50

TIME_TOKENS = [
    "tid", "time",
    "year", "aar", "år",
    "month", "maaned", "måned",
    "quarter", "kvartal",
    "week", "uke",
    "day", "dato", "date",
]

DEFAULT_HEADERS = {
    "User-Agent": "norway-housing-pressure-tracker/1.0 (portfolio project)",
    "Accept": "*/*",
}


# -----------------------------
# Path + timestamp helpers
# -----------------------------

def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def now_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M")


# -----------------------------
# SSB json-stat2 helpers
# -----------------------------

def _ordered_codes(category: Dict[str, Any]) -> List[str]:
    idx = category.get("index")
    if isinstance(idx, list):
        return [str(x) for x in idx]
    if isinstance(idx, dict):
        return [str(code) for code, pos in sorted(idx.items(), key=lambda kv: kv[1])]
    raise ValueError(f"Unsupported category.index type: {type(idx)}")


def _labels_map(category: Dict[str, Any]) -> Dict[str, str]:
    lab = category.get("label", {}) or {}
    return {str(k): str(v) for k, v in lab.items()}


def ssb_get_metadata(session: requests.Session, meta_url: str) -> Dict[str, Any]:
    resp = safe_request(session, meta_url, method="GET", params={"lang": SSB_LANG})
    return resp.json()


def ssb_dimension_ids(meta: Dict[str, Any]) -> List[str]:
    # SSB PxWebApi v2 metadata is json-stat2; dimension order is typically in top-level `id`.
    top_ids = meta.get("id")
    if isinstance(top_ids, list):
        return [str(x) for x in top_ids]

    dim = meta.get("dimension", {})
    ids = dim.get("id")
    if isinstance(ids, list):
        return [str(x) for x in ids]

    raise ValueError("Could not read dimension ids from SSB metadata response.")


def ssb_codes_labels(meta: Dict[str, Any], var_id: str) -> Tuple[List[str], Dict[str, str]]:
    dim = meta.get("dimension", {})
    if var_id not in dim:
        raise ValueError(f"Dimension '{var_id}' not found in metadata.dimension")
    cat = dim[var_id]["category"]
    codes = _ordered_codes(cat)
    labels = _labels_map(cat)
    return codes, labels


def ssb_find_var_id(ids: List[str], prefer: List[str], contains: List[str]) -> Optional[str]:
    for p in prefer:
        if p in ids:
            return p
    for i in ids:
        il = i.lower()
        if any(tok in il for tok in contains):
            return i
    return None


def is_time_like_dim(dim_id: str) -> bool:
    d = dim_id.lower()
    return any(tok in d for tok in TIME_TOKENS)


def pick_total_like_code(codes: List[str], labels: Dict[str, str]) -> str:
    patterns = ["total", "all", "overall", "whole country", "norway", "entire"]
    for c in codes:
        if any(p in labels.get(c, "").lower() for p in patterns):
            return c
    for c in ["00", "0", "TOT", "TOTAL"]:
        if c in codes:
            return c
    return codes[0]


def pick_region_subset(codes: List[str], labels: Dict[str, str], max_n: int) -> List[str]:
    preferred_terms = ["norway", "whole country", "oslo", "bergen", "trondheim", "stavanger", "tromsø"]
    preferred: List[str] = []
    for term in preferred_terms:
        for c in codes:
            if term in labels.get(c, "").lower() and c not in preferred:
                preferred.append(c)
    remaining = [c for c in codes if c not in preferred]
    return (preferred + remaining)[:max_n]


def pick_from_1992_or_all(codes: List[str]) -> str:
    for c in codes:
        if str(c).startswith("1992"):
            return f"from({c})"
    return "*"


def selection_to_codes(all_codes: List[str], selection: str | List[str]) -> List[str]:
    if isinstance(selection, list):
        return selection
    if selection == "*":
        return all_codes
    if selection.startswith("from(") and selection.endswith(")"):
        start = selection[len("from(") : -1]
        return all_codes[all_codes.index(start) :] if start in all_codes else all_codes
    return [selection]


def build_ssb_query(meta: Dict[str, Any], table_kind: str) -> Tuple[Dict[str, Any], List[str], Optional[str], Optional[str]]:
    ids = ssb_dimension_ids(meta)

    content_id = ssb_find_var_id(ids, prefer=["ContentsCode"], contains=["contents"])
    region_id = ssb_find_var_id(ids, prefer=["Region"], contains=["region"])
    if not content_id:
        raise ValueError("Could not identify ContentsCode in SSB metadata.")

    # Time dims: prefer Tid if present (single combined time dimension)
    if "Tid" in ids:
        time_ids = ["Tid"]
    else:
        time_ids = [d for d in ids if is_time_like_dim(d)]
    if not time_ids:
        raise ValueError("Could not identify any time-like dimension in SSB metadata.")

    selections: Dict[str, str | List[str]] = {}
    selections[content_id] = "*"

    for tid in time_ids:
        t_codes, _ = ssb_codes_labels(meta, tid)
        selections[tid] = pick_from_1992_or_all(t_codes)

    if table_kind == "cpi":
        kons_id = ssb_find_var_id(ids, prefer=["Konsumgrp"], contains=["konsum", "consum"])
        if kons_id:
            k_codes, k_labels = ssb_codes_labels(meta, kons_id)
            selections[kons_id] = pick_total_like_code(k_codes, k_labels)

    core = {content_id, *time_ids}
    if region_id:
        core.add(region_id)

    for dim_id in ids:
        if dim_id in core:
            continue
        d_codes, d_labels = ssb_codes_labels(meta, dim_id)
        selections[dim_id] = pick_total_like_code(d_codes, d_labels)

    warn: Optional[str] = None

    if region_id:
        r_codes, r_labels = ssb_codes_labels(meta, region_id)
        if len(r_codes) > SSB_REGION_TARGET_MAX:
            selections[region_id] = pick_region_subset(r_codes, r_labels, SSB_REGION_TARGET_MAX)
            warn = f"Region dimension truncated to {SSB_REGION_TARGET_MAX} values for manageability."

    # Cell-cap guard
    counts: Dict[str, int] = {}
    explicit_selected: Dict[str, List[str]] = {}
    for dim_id in ids:
        d_codes, _ = ssb_codes_labels(meta, dim_id)
        sel = selections.get(dim_id, "*")
        sel_codes = selection_to_codes(d_codes, sel)
        explicit_selected[dim_id] = sel_codes
        counts[dim_id] = len(sel_codes)

    est_cells = 1
    for dim_id in ids:
        est_cells *= max(1, counts[dim_id])

    if est_cells > SSB_MAX_CELLS_SAFETY:
        trim_order = ([region_id] if region_id else []) + sorted(ids, key=lambda x: counts.get(x, 1), reverse=True)
        for dim_id in trim_order:
            if not dim_id or counts.get(dim_id, 1) <= 1:
                continue
            other_prod = int(est_cells / counts[dim_id])
            max_allowed = max(1, math.floor(SSB_MAX_CELLS_SAFETY / max(1, other_prod)))
            if max_allowed < counts[dim_id]:
                selections[dim_id] = explicit_selected[dim_id][:max_allowed]
                warn2 = (
                    f"Cell-cap guard: trimmed {dim_id} from {counts[dim_id]} to {max_allowed} selections "
                    f"(estimated cells {est_cells} > {SSB_MAX_CELLS_SAFETY})."
                )
                warn = (warn + " " + warn2) if warn else warn2
                break

    params: Dict[str, Any] = {
        "lang": SSB_LANG,
        "outputformat": "csv",
        "outputformatparams": "separatorsemicolon,usecodes",
    }
    for dim_id, sel in selections.items():
        params[f"valueCodes[{dim_id}]"] = ",".join(sel) if isinstance(sel, list) else sel

    return params, time_ids, region_id, warn


def read_ssb_csv(session: requests.Session, data_url: str, params: Dict[str, Any]) -> pd.DataFrame:
    resp = safe_request(session, data_url, method="GET", params=params)
    text = resp.content.decode("utf-8-sig")
    return pd.read_csv(StringIO(text), sep=";")


def verify_df(df: pd.DataFrame, time_cols: List[str], region_col: Optional[str]) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "row_count": int(df.shape[0]),
        "column_count": int(df.shape[1]),
        "column_names": list(df.columns.astype(str)),
    }
    for tcol in time_cols:
        if tcol in df.columns:
            s = df[tcol].astype(str)
            info["time_col"] = tcol
            info["time_min_code"] = str(s.min()) if len(s) else None
            info["time_max_code"] = str(s.max()) if len(s) else None
            break
    if region_col and region_col in df.columns:
        info["distinct_regions"] = int(df[region_col].nunique(dropna=True))
    return info


# -----------------------------
# Norges Bank (policy rate daily) download
# -----------------------------

def _sniff_sep(sample: str) -> str:
    # Very small heuristic: choose the delimiter that appears more in header line.
    header = sample.splitlines()[0] if sample.splitlines() else sample
    return ";" if header.count(";") > header.count(",") else ","


def download_norgesbank_policy_rate_daily(session: requests.Session) -> pd.DataFrame:
    """
    Uses Norges Bank open data API (CSV) linked from the official policy rate statistics page.
    """
    # Prefer full history via startPeriod; fallback to lastNObservations if startPeriod is rejected.
    params_primary = {"bom": "include", "format": "csv", "startPeriod": "1986-01-01", "locale": "en"}
    params_fallback = {"bom": "include", "format": "csv", "lastNObservations": "20000", "locale": "en"}

    try:
        resp = safe_request(session, NORGESBANK_POLICY_RATE_API_URL, method="GET", params=params_primary, headers=DEFAULT_HEADERS)
    except Exception:
        resp = safe_request(session, NORGESBANK_POLICY_RATE_API_URL, method="GET", params=params_fallback, headers=DEFAULT_HEADERS)

    text = resp.content.decode("utf-8-sig")
    sep = _sniff_sep(text[:2000])
    df = pd.read_csv(StringIO(text), sep=sep)

    return df


# -----------------------------
# Main
# -----------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Part 1: Download raw datasets and store with metadata.")
    parser.add_argument("--dry-run", action="store_true", help="Create folders + metadata skeleton only (no downloads).")
    args = parser.parse_args()

    root = project_root()
    raw_dir = root / "data" / "raw"
    docs_dir = root / "docs"
    ensure_dir(raw_dir)
    ensure_dir(docs_dir)

    ts = now_timestamp()

    metadata: Dict[str, Any] = {
        "project": "Norway Housing Pressure Tracker (1992–present)",
        "part": "Part 1 — Download raw",
        "retrieved_at_local": datetime.now().isoformat(timespec="seconds"),
        "timestamp_tag": ts,
        "datasets": [],
        "run_warnings": [],
    }

    if args.dry_run:
        metadata["run_warnings"].append("Dry run: no downloads executed.")
        meta_path = docs_dir / "metadata_part1.json"
        save_metadata(metadata, meta_path)
        print("DRY RUN OK")
        print(f"- Created/verified: {raw_dir}")
        print(f"- Wrote metadata:   {meta_path}")
        return 0

    session = requests.Session()
    session.headers.update(DEFAULT_HEADERS)

    # --- Dataset 1: SSB HPI (07221)
    ds1_name = "ssb_hpi_07221"
    ds1: Dict[str, Any] = {
        "dataset_name": ds1_name,
        "endpoint": SSB_HPI_DATA_URL,
        "parameters": None,
        "output_files": [],
        "row_count": None,
        "column_count": None,
        "column_names": None,
        "warnings": [],
    }
    try:
        meta = ssb_get_metadata(session, SSB_HPI_META_URL)
        params, time_ids, region_id, warn = build_ssb_query(meta, table_kind="hpi")
        ds1["parameters"] = params
        if warn:
            ds1["warnings"].append(warn)

        df_hpi = read_ssb_csv(session, SSB_HPI_DATA_URL, params=params)
        out = write_outputs(df_hpi, raw_dir / f"{ds1_name}_{ts}", write_parquet=True)
        ds1["output_files"] = out["files"]
        ds1["warnings"].extend(out["warnings"])
        ds1.update(verify_df(df_hpi, time_cols=time_ids, region_col=region_id))
    except Exception as exc:
        ds1["warnings"].append(str(exc))
    metadata["datasets"].append(ds1)

    # --- Dataset 2: SSB CPI (08981)
    ds2_name = "ssb_cpi_08981"
    ds2: Dict[str, Any] = {
        "dataset_name": ds2_name,
        "endpoint": SSB_CPI_DATA_URL,
        "parameters": None,
        "output_files": [],
        "row_count": None,
        "column_count": None,
        "column_names": None,
        "warnings": [],
    }
    try:
        meta = ssb_get_metadata(session, SSB_CPI_META_URL)
        params, time_ids, region_id, warn = build_ssb_query(meta, table_kind="cpi")
        ds2["parameters"] = params
        if warn:
            ds2["warnings"].append(warn)

        df_cpi = read_ssb_csv(session, SSB_CPI_DATA_URL, params=params)
        out = write_outputs(df_cpi, raw_dir / f"{ds2_name}_{ts}", write_parquet=True)
        ds2["output_files"] = out["files"]
        ds2["warnings"].extend(out["warnings"])
        ds2.update(verify_df(df_cpi, time_cols=time_ids, region_col=region_id))
    except Exception as exc:
        ds2["warnings"].append(str(exc))
    metadata["datasets"].append(ds2)

    # --- Dataset 3: Norges Bank policy rate (daily CSV via open data API)
    ds3_name = "norgesbank_policy_rate"
    ds3: Dict[str, Any] = {
        "dataset_name": ds3_name,
        "endpoint": NORGESBANK_POLICY_RATE_API_URL,
        "parameters": {"format": "csv", "startPeriod": "1986-01-01", "locale": "en"},
        "output_files": [],
        "row_count": None,
        "column_count": None,
        "column_names": None,
        "warnings": [],
    }
    try:
        df_nb = download_norgesbank_policy_rate_daily(session)
        out = write_outputs(df_nb, raw_dir / f"{ds3_name}_{ts}", write_parquet=True)
        ds3["output_files"] = out["files"]
        ds3["warnings"].extend(out["warnings"])

        # Minimal verification: date parse success if a date-like column exists
        date_col = None
        for cand in ["TIME_PERIOD", "Date", "date"]:
            if cand in df_nb.columns:
                date_col = cand
                break
        if date_col:
            parsed = pd.to_datetime(df_nb[date_col], errors="coerce")
            ds3["date_col"] = date_col
            ds3["date_parse_success_rate"] = float(parsed.notna().mean()) if len(parsed) else None
            ds3["date_min"] = str(parsed.min()) if parsed.notna().any() else None
            ds3["date_max"] = str(parsed.max()) if parsed.notna().any() else None

        ds3["row_count"] = int(df_nb.shape[0])
        ds3["column_count"] = int(df_nb.shape[1])
        ds3["column_names"] = list(df_nb.columns.astype(str))
    except Exception as exc:
        ds3["warnings"].append(str(exc))
    metadata["datasets"].append(ds3)

    # Save metadata + print summary
    meta_path = docs_dir / "metadata_part1.json"
    save_metadata(metadata, meta_path)

    print("DOWNLOAD SUMMARY (Part 1)")
    for d in metadata["datasets"]:
        print(f"- {d['dataset_name']}: rows={d['row_count']} cols={d['column_count']} warnings={len(d['warnings'])}")
        if d.get("time_min_code") is not None:
            print(f"  time_code_range: {d.get('time_min_code')} -> {d.get('time_max_code')}")
        if d.get("distinct_regions") is not None:
            print(f"  distinct_regions: {d.get('distinct_regions')}")
        if d.get("date_parse_success_rate") is not None:
            print(f"  date_parse_success_rate: {d.get('date_parse_success_rate')}")
    print(f"Metadata written: {meta_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

