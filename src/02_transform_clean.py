from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
DOCS_DIR = Path("docs")

SSB_HPI_PREFIX = "ssb_hpi_07221"
SSB_CPI_PREFIX = "ssb_cpi_08981"
NB_POLICY_PREFIX = "norgesbank_policy_rate"

TAG_PATTERN = re.compile(r".*_(\d{8}_\d{4})\.(csv|parquet)$", re.IGNORECASE)

# HPI columns: "Boligindeks 1992K1", "SesJustBoligindeks 2025K3"
HPI_COL_RE = re.compile(r"^(?P<series>.+?)\s(?P<tp>\d{4}K[1-4])$")

# CPI columns: "KpiIndMnd 1992", "KpiIndMnd 2025"
CPI_COL_RE = re.compile(r"^(?P<series>.+?)\s(?P<year>\d{4})$")


@dataclass(frozen=True)
class RawAsset:
    dataset: str
    path: Path
    tag: str
    ext: str


def _extract_tag(p: Path) -> Optional[Tuple[str, str]]:
    m = TAG_PATTERN.match(p.name)
    if not m:
        return None
    return m.group(1), m.group(2).lower()


def find_latest_tag(raw_dir: Path = RAW_DIR) -> str:
    tags: List[str] = []
    for p in raw_dir.glob("*.*"):
        got = _extract_tag(p)
        if got:
            tags.append(got[0])
    if not tags:
        raise FileNotFoundError(
            f"No raw files found in {raw_dir}. Expected files like '*_YYYYMMDD_HHMM.parquet'."
        )
    return sorted(set(tags))[-1]


def resolve_raw_asset(prefix: str, tag: str, raw_dir: Path = RAW_DIR) -> RawAsset:
    p_parquet = raw_dir / f"{prefix}_{tag}.parquet"
    p_csv = raw_dir / f"{prefix}_{tag}.csv"

    if p_parquet.exists():
        return RawAsset(dataset=prefix, path=p_parquet, tag=tag, ext="parquet")
    if p_csv.exists():
        return RawAsset(dataset=prefix, path=p_csv, tag=tag, ext="csv")

    raise FileNotFoundError(
        f"Missing raw input for prefix='{prefix}' tag='{tag}'. Looked for:\n"
        f"  - {p_parquet}\n"
        f"  - {p_csv}"
    )


def read_raw(asset: RawAsset) -> pd.DataFrame:
    if asset.ext == "parquet":
        return pd.read_parquet(asset.path)
    if asset.ext == "csv":
        return pd.read_csv(asset.path)
    raise ValueError(f"Unsupported ext: {asset.ext}")


def ensure_dirs() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def profile_df(df: pd.DataFrame, key_cols: Optional[List[str]] = None) -> Dict[str, Any]:
    prof: Dict[str, Any] = {
        "shape": {"rows": int(df.shape[0]), "cols": int(df.shape[1])},
        "columns": [str(c) for c in df.columns.tolist()],
    }
    if key_cols:
        prof["missing_in_key_cols"] = {
            k: int(df[k].isna().sum()) if k in df.columns else None for k in key_cols
        }
        if all(k in df.columns for k in key_cols):
            prof["duplicate_key_rows"] = int(df.duplicated(subset=key_cols).sum())
        else:
            prof["duplicate_key_rows"] = None
    return prof


def value_sanity_stats(s: pd.Series) -> Dict[str, Any]:
    s_num = pd.to_numeric(s, errors="coerce")
    out: Dict[str, Any] = {
        "non_numeric_or_missing": int(s_num.isna().sum()),
        "negative_count": int((s_num < 0).sum(skipna=True)),
    }
    if s_num.notna().any():
        q1 = float(s_num.quantile(0.25))
        q3 = float(s_num.quantile(0.75))
        iqr = q3 - q1
        lo = q1 - 3 * iqr
        hi = q3 + 3 * iqr
        out.update(
            {
                "min": float(s_num.min()),
                "max": float(s_num.max()),
                "q1": q1,
                "q3": q3,
                "iqr": float(iqr),
                "outlier_lo_3iqr": float(lo),
                "outlier_hi_3iqr": float(hi),
                "outlier_count_3iqr": int(((s_num < lo) | (s_num > hi)).sum(skipna=True)),
            }
        )
    return out


def parse_ssb_quarter(tp: str) -> Optional[str]:
    m = re.match(r"^(?P<y>\d{4})K(?P<q>[1-4])$", str(tp))
    if not m:
        return None
    return f"{m.group('y')}Q{m.group('q')}"


def quarter_start_from_quarter(q: pd.Series) -> pd.Series:
    pi = pd.PeriodIndex(q.astype("string"), freq="Q")
    return pi.to_timestamp(how="start")


def _quarter_from_date(s: pd.Series) -> pd.Series:
    p = pd.PeriodIndex(pd.to_datetime(s, errors="coerce"), freq="Q")
    return pd.Series(p.strftime("%YQ%q"), index=s.index, dtype="string")


def transform_hpi(raw: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    cols = [str(c) for c in raw.columns.tolist()]
    hpi_cols = [c for c in cols if HPI_COL_RE.match(c)]
    if not hpi_cols:
        raise ValueError("No HPI quarter columns matched. Expected columns like 'Boligindeks 1992K1'.")

    id_cols = [c for c in cols if c not in hpi_cols]

    melted = raw.melt(id_vars=id_cols, value_vars=hpi_cols, var_name="raw_series_time", value_name="value")
    extracted = melted["raw_series_time"].str.extract(HPI_COL_RE)
    melted["series"] = extracted["series"]
    melted["time_period"] = extracted["tp"]

    melted["quarter"] = melted["time_period"].apply(parse_ssb_quarter)
    melted["quarter_start"] = quarter_start_from_quarter(melted["quarter"])

    out = pd.DataFrame(
        {
            "dataset": "hpi_07221",
            "boligtype_code": (
                melted["Boligtype"].astype("string")
                if "Boligtype" in melted.columns
                else pd.Series([pd.NA] * len(melted), dtype="string")
            ),
            "series": melted["series"].astype("string"),
            "time_period": melted["time_period"].astype("string"),
            "quarter": melted["quarter"].astype("string"),
            "quarter_start": melted["quarter_start"],
            "value": pd.to_numeric(melted["value"], errors="coerce"),
        }
    )

    key_cols = ["dataset", "boligtype_code", "series", "quarter"]
    meta: Dict[str, Any] = {
        "id_cols_detected": id_cols,
        "time_cols_detected_count": int(len(hpi_cols)),
        "time_parse_success_rate": float(out["quarter"].notna().mean()) if len(out) else 0.0,
        "min_quarter": str(out["quarter"].min()) if out["quarter"].notna().any() else None,
        "max_quarter": str(out["quarter"].max()) if out["quarter"].notna().any() else None,
        "profile": profile_df(out, key_cols=key_cols),
        "value_sanity": value_sanity_stats(out["value"]),
    }
    return out, meta


def transform_cpi(raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    CPI extract: year-columns + single 'Maaned' code (observed 90).
    metadata_part1.json captured request params but not value-text mappings, so:
      - keep maaned_code
      - maaned_label = null
      - infer annual from code>=90 (documented)
      - repeat annual CPI for all 4 quarters per year
    """
    cols = [str(c) for c in raw.columns.tolist()]
    year_cols = [c for c in cols if CPI_COL_RE.match(c)]
    if not year_cols:
        raise ValueError("No CPI year columns matched. Expected columns like 'KpiIndMnd 1992'.")

    id_cols = [c for c in cols if c not in year_cols]

    melted = raw.melt(id_vars=id_cols, value_vars=year_cols, var_name="raw_series_year", value_name="value")
    ex = melted["raw_series_year"].str.extract(CPI_COL_RE)
    melted["series"] = ex["series"]
    melted["year"] = pd.to_numeric(ex["year"], errors="coerce").astype("Int64")

    maaned_code = melted["Maaned"] if "Maaned" in melted.columns else pd.Series([pd.NA] * len(melted))
    try:
        mc = int(str(maaned_code.iloc[0]))
    except Exception:
        mc = None
    period_kind = "annual_assumed_from_code" if (mc is not None and mc >= 90) else "unknown"

    out_long = pd.DataFrame(
        {
            "dataset": "cpi_08981",
            "maaned_code": maaned_code.astype("string"),
            "maaned_label": pd.Series([pd.NA] * len(melted), dtype="string"),
            "period_kind": pd.Series([period_kind] * len(melted), dtype="string"),
            "series": melted["series"].astype("string"),
            "year": melted["year"],
            "time_period": melted["year"].astype("string"),
            "value": pd.to_numeric(melted["value"], errors="coerce"),
        }
    )

    rows = []
    for _, r in out_long.iterrows():
        if pd.isna(r["year"]):
            continue
        y = int(r["year"])
        for q in (1, 2, 3, 4):
            quarter = f"{y}Q{q}"
            rows.append(
                {
                    "dataset": r["dataset"],
                    "maaned_code": r["maaned_code"],
                    "maaned_label": r["maaned_label"],
                    "series": r["series"],
                    "year": r["year"],
                    "quarter": quarter,
                    "quarter_start": pd.Period(quarter, freq="Q").to_timestamp(how="start"),
                    "value": r["value"],
                    "method": "annual_value_repeated_each_quarter",
                }
            )
    out_q = pd.DataFrame(rows)

    key_long = ["dataset", "series", "year", "maaned_code"]
    key_q = ["dataset", "series", "quarter", "maaned_code"]

    meta: Dict[str, Any] = {
        "id_cols_detected": id_cols,
        "year_cols_detected_count": int(len(year_cols)),
        "maaned_code_observed": str(out_long["maaned_code"].dropna().iloc[0]) if out_long["maaned_code"].notna().any() else None,
        "maaned_label_observed": None,
        "maaned_label_source": "metadata_part1_missing_value_texts",
        "period_kind_inferred": period_kind,
        "long_profile": profile_df(out_long, key_cols=key_long),
        "quarterly_profile": profile_df(out_q, key_cols=key_q),
        "long_value_sanity": value_sanity_stats(out_long["value"]),
        "quarterly_value_sanity": value_sanity_stats(out_q["value"]),
        "min_year": int(out_long["year"].min()) if out_long["year"].notna().any() else None,
        "max_year": int(out_long["year"].max()) if out_long["year"].notna().any() else None,
        "min_quarter": str(out_q["quarter"].min()) if out_q["quarter"].notna().any() else None,
        "max_quarter": str(out_q["quarter"].max()) if out_q["quarter"].notna().any() else None,
    }
    return out_long, out_q, meta


def transform_policy_rate(raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    df = raw.copy()

    df["date"] = pd.to_datetime(df["TIME_PERIOD"], errors="coerce")
    df["rate"] = pd.to_numeric(df["OBS_VALUE"], errors="coerce")

    for col in ["INSTRUMENT_TYPE", "TENOR", "UNIT_MEASURE", "COLLECTION"]:
        if col not in df.columns:
            df[col] = pd.NA

    df["series_id"] = (
        df["INSTRUMENT_TYPE"].astype("string")
        + "."
        + df["TENOR"].astype("string")
        + "."
        + df["UNIT_MEASURE"].astype("string")
        + "."
        + df["COLLECTION"].astype("string")
    )

    out = pd.DataFrame(
        {
            "dataset": "norgesbank_policy_rate",
            "series_id": df["series_id"].astype("string"),
            "freq_code": df.get("FREQ", pd.NA),
            "freq_label": df.get("Frequency", pd.NA),
            "instrument_type_code": df.get("INSTRUMENT_TYPE", pd.NA),
            "instrument_type_label": df.get("Instrument Type", pd.NA),
            "tenor_code": df.get("TENOR", pd.NA),
            "tenor_label": df.get("Tenor", pd.NA),
            "unit_code": df.get("UNIT_MEASURE", pd.NA),
            "unit_label": df.get("Unit of Measure", pd.NA),
            "collection_code": df.get("COLLECTION", pd.NA),
            "collection_label": df.get("Collection Indicator", pd.NA),
            "decimals": df.get("DECIMALS", pd.NA),
            "date": df["date"],
            "rate": df["rate"],
        }
    )

    key_cols = ["dataset", "series_id", "date"]
    meta_rate = {
        "time_parse_success_rate": float(out["date"].notna().mean()) if len(out) else 0.0,
        "min_date": str(out["date"].min().date()) if out["date"].notna().any() else None,
        "max_date": str(out["date"].max().date()) if out["date"].notna().any() else None,
        "unique_series_id_count": int(out["series_id"].nunique(dropna=True)),
        "profile": profile_df(out, key_cols=key_cols),
        "value_sanity": value_sanity_stats(out["rate"]),
    }

    ch = out.sort_values(["series_id", "date"]).copy()
    ch["prev_rate"] = ch.groupby("series_id")["rate"].shift(1)
    ch["delta_rate"] = ch["rate"] - ch["prev_rate"]
    changes = ch[ch["delta_rate"].notna() & (ch["delta_rate"] != 0)].copy()

    out_changes = pd.DataFrame(
        {
            "dataset": "norgesbank_policy_rate",
            "series_id": changes["series_id"].astype("string"),
            "change_date": changes["date"],
            "previous_rate": changes["prev_rate"],
            "new_rate": changes["rate"],
            "delta_rate": changes["delta_rate"],
        }
    )

    meta_changes = {
        "profile": profile_df(out_changes, key_cols=["dataset", "series_id", "change_date"]),
        "min_change_date": str(out_changes["change_date"].min().date()) if out_changes["change_date"].notna().any() else None,
        "max_change_date": str(out_changes["change_date"].max().date()) if out_changes["change_date"].notna().any() else None,
        "change_events_count": int(len(out_changes)),
        "delta_sanity": value_sanity_stats(out_changes["delta_rate"]) if len(out_changes) else {"note": "no_rows"},
    }

    return out, out_changes, {"fact_policy_rate": meta_rate, "fact_policy_rate_changes": meta_changes}


def build_analysis_ready_quarterly(tag: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    hpi_path = PROCESSED_DIR / "fact_hpi_long.csv"
    cpi_path = PROCESSED_DIR / "fact_cpi_quarterly.csv"
    if not hpi_path.exists():
        raise FileNotFoundError(f"Missing required input: {hpi_path}")
    if not cpi_path.exists():
        raise FileNotFoundError(f"Missing required input: {cpi_path}")

    hpi = pd.read_csv(hpi_path)
    cpiq = pd.read_csv(cpi_path)

    # ---- HPI pivot (one row per boligtype_code x quarter)
    hpi["quarter"] = hpi["quarter"].astype("string")
    hpi_p = (
        hpi.pivot_table(
            index=["boligtype_code", "quarter"],
            columns="series",
            values="value",
            aggfunc="first",
        )
        .reset_index()
    )
    # stable names
    rename_map = {}
    for col in hpi_p.columns:
        if col == "Boligindeks":
            rename_map[col] = "hpi_index"
        elif col == "SesJustBoligindeks":
            rename_map[col] = "hpi_sa_index"
    hpi_p = hpi_p.rename(columns=rename_map)
    hpi_p["quarter_start"] = quarter_start_from_quarter(hpi_p["quarter"])

    # ---- CPI quarterly: ensure one row per quarter
    cpiq["quarter"] = cpiq["quarter"].astype("string")
    cpi_use = cpiq.rename(columns={"value": "cpi_index"}).copy()
    cpi_use = (
        cpi_use.groupby("quarter", as_index=False)
        .agg(
            cpi_index=("cpi_index", "mean"),
            cpi_method=("method", "first"),
        )
    )
    cpi_use["quarter_start"] = quarter_start_from_quarter(cpi_use["quarter"])

    out = hpi_p.merge(cpi_use, on=["quarter"], how="left", validate="many_to_one", suffixes=("", "_cpi"))
    # keep HPI quarter_start as primary (they should match)
    if "quarter_start_cpi" in out.columns:
        out = out.drop(columns=["quarter_start_cpi"])

    # ---- Policy quarterly context (optional)
    policy_path = PROCESSED_DIR / "fact_policy_rate.csv"
    changes_path = PROCESSED_DIR / "fact_policy_rate_changes.csv"
    policy_meta: Dict[str, Any] = {"included": False}

    if policy_path.exists():
        pr = pd.read_csv(policy_path)
        pr["date"] = pd.to_datetime(pr["date"], errors="coerce")
        pr = pr.sort_values(["series_id", "date"]) if "series_id" in pr.columns else pr.sort_values(["date"])
        pr["quarter"] = _quarter_from_date(pr["date"])

        if "series_id" in pr.columns:
            top_series = pr["series_id"].value_counts().index[0]
            pr = pr[pr["series_id"] == top_series].copy()
            policy_meta["series_id_used"] = str(top_series)

        pr_q = (
            pr.groupby("quarter", as_index=False)
            .agg(
                policy_rate_avg=("rate", "mean"),
                policy_rate_end=("rate", lambda x: x.iloc[-1]),
                policy_rate_min=("rate", "min"),
                policy_rate_max=("rate", "max"),
                policy_obs_count=("rate", "count"),
            )
        )

        if changes_path.exists():
            ch = pd.read_csv(changes_path)
            ch["change_date"] = pd.to_datetime(ch["change_date"], errors="coerce")
            ch["quarter"] = _quarter_from_date(ch["change_date"])
            if "series_id" in ch.columns and "series_id_used" in policy_meta:
                ch = ch[ch["series_id"] == policy_meta["series_id_used"]].copy()
            ch_q = (
                ch.groupby("quarter", as_index=False)
                .agg(
                    policy_change_events=("delta_rate", "count"),
                    policy_delta_sum=("delta_rate", "sum"),
                    policy_delta_abs_sum=("delta_rate", lambda x: (x.abs()).sum()),
                )
            )
            pr_q = pr_q.merge(ch_q, on="quarter", how="left")
        else:
            pr_q["policy_change_events"] = pd.NA
            pr_q["policy_delta_sum"] = pd.NA
            pr_q["policy_delta_abs_sum"] = pd.NA

        out = out.merge(pr_q, on="quarter", how="left")
        policy_meta["included"] = True

    # ---- Derived metrics (deterministic)
    out = out.sort_values(["boligtype_code", "quarter_start"]).reset_index(drop=True)

    for col in ["hpi_index", "hpi_sa_index"]:
        if col in out.columns:
            out[f"{col}_qoq_pct"] = out.groupby("boligtype_code")[col].pct_change(1) * 100.0
            out[f"{col}_yoy_pct"] = out.groupby("boligtype_code")[col].pct_change(4) * 100.0

    out["cpi_yoy_pct"] = out.groupby("boligtype_code")["cpi_index"].pct_change(4) * 100.0

    if "hpi_index" in out.columns:
        out["real_hpi_proxy"] = out["hpi_index"] / out["cpi_index"]
    if "hpi_sa_index" in out.columns:
        out["real_hpi_sa_proxy"] = out["hpi_sa_index"] / out["cpi_index"]

    meta: Dict[str, Any] = {
        "inputs": {
            "fact_hpi_long": str(hpi_path),
            "fact_cpi_quarterly": str(cpi_path),
            "fact_policy_rate": str(policy_path) if policy_path.exists() else None,
            "fact_policy_rate_changes": str(changes_path) if changes_path.exists() else None,
        },
        "policy_context": policy_meta,
        "profile": profile_df(out, key_cols=["boligtype_code", "quarter"]),
        "min_quarter": str(out["quarter"].min()) if out["quarter"].notna().any() else None,
        "max_quarter": str(out["quarter"].max()) if out["quarter"].notna().any() else None,
        "missing_cpi": int(out["cpi_index"].isna().sum()) if "cpi_index" in out.columns else None,
    }
    for col in ["hpi_index", "hpi_sa_index", "cpi_index", "real_hpi_proxy", "real_hpi_sa_proxy"]:
        if col in out.columns:
            meta[f"{col}_negative_count"] = int((pd.to_numeric(out[col], errors="coerce") < 0).sum(skipna=True))

    return out, meta


def write_processed(df: pd.DataFrame, base_name: str, tag: str) -> Dict[str, Any]:
    csv_path = PROCESSED_DIR / f"{base_name}.csv"
    pq_path = PROCESSED_DIR / f"{base_name}.parquet"

    df.to_csv(csv_path, index=False, encoding="utf-8")

    pq_written = False
    try:
        df.to_parquet(pq_path, index=False)
        pq_written = True
    except Exception:
        pass

    return {"csv": str(csv_path), "parquet": str(pq_path), "parquet_written": pq_written, "tag": tag}


def load_or_init_metadata_part2(tag: str) -> Tuple[Path, Dict[str, Any]]:
    metadata_path = DOCS_DIR / "metadata_part2.json"
    metadata: Dict[str, Any] = {}
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            metadata = {}
    metadata.setdefault("run_history", [])
    metadata.setdefault("tables", {})
    metadata["latest_tag"] = tag
    return metadata_path, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Part 2: Transform raw -> processed (tidy long).")
    parser.add_argument("--tag", type=str, default=None, help="Raw timestamp tag like 20251215_1650")
    parser.add_argument("--run", type=str, choices=["hpi", "cpi", "policy", "analysis", "all"], required=True)
    args = parser.parse_args()

    ensure_dirs()
    tag = args.tag or find_latest_tag(RAW_DIR)

    metadata_path, metadata = load_or_init_metadata_part2(tag)
    ran: List[str] = []

    if args.run in ("hpi", "all"):
        asset = resolve_raw_asset(SSB_HPI_PREFIX, tag)
        raw = read_raw(asset)
        out, meta = transform_hpi(raw)
        paths = write_processed(out, "fact_hpi_long", tag)
        metadata["tables"]["fact_hpi_long"] = {
            "source": {"dataset": SSB_HPI_PREFIX, "path": str(asset.path)},
            "outputs": paths,
            "validation": meta,
        }
        ran.append("hpi")
        print(f"\n[run:hpi] rows={out.shape[0]} cols={out.shape[1]} min={meta.get('min_quarter')} max={meta.get('max_quarter')}")

    if args.run in ("cpi", "all"):
        asset = resolve_raw_asset(SSB_CPI_PREFIX, tag)
        raw = read_raw(asset)
        out_long, out_q, meta = transform_cpi(raw)
        paths_long = write_processed(out_long, "fact_cpi_long", tag)
        paths_q = write_processed(out_q, "fact_cpi_quarterly", tag)
        metadata["tables"]["fact_cpi_long"] = {
            "source": {"dataset": SSB_CPI_PREFIX, "path": str(asset.path)},
            "outputs": paths_long,
            "validation": meta,
        }
        metadata["tables"]["fact_cpi_quarterly"] = {
            "source": {"dataset": SSB_CPI_PREFIX, "path": str(asset.path)},
            "outputs": paths_q,
            "validation": meta,
        }
        ran.append("cpi")
        print(f"\n[run:cpi] long_rows={out_long.shape[0]} quarterly_rows={out_q.shape[0]}")

    if args.run in ("policy", "all"):
        asset = resolve_raw_asset(NB_POLICY_PREFIX, tag)
        raw = read_raw(asset)
        out_rate, out_changes, meta = transform_policy_rate(raw)
        paths_rate = write_processed(out_rate, "fact_policy_rate", tag)
        paths_changes = write_processed(out_changes, "fact_policy_rate_changes", tag)
        metadata["tables"]["fact_policy_rate"] = {
            "source": {"dataset": NB_POLICY_PREFIX, "path": str(asset.path)},
            "outputs": paths_rate,
            "validation": meta["fact_policy_rate"],
        }
        metadata["tables"]["fact_policy_rate_changes"] = {
            "source": {"dataset": NB_POLICY_PREFIX, "path": str(asset.path)},
            "outputs": paths_changes,
            "validation": meta["fact_policy_rate_changes"],
        }
        ran.append("policy")
        v = meta["fact_policy_rate"]
        c = meta["fact_policy_rate_changes"]
        print(f"\n[run:policy] rate_rows={out_rate.shape[0]} series={v.get('unique_series_id_count')} date={v.get('min_date')}..{v.get('max_date')}")
        print(f"[run:policy] change_events={c.get('change_events_count')} change_date={c.get('min_change_date')}..{c.get('max_change_date')}")

    if args.run in ("analysis", "all"):
        out_a, meta_a = build_analysis_ready_quarterly(tag)
        paths_a = write_processed(out_a, "analysis_ready_quarterly", tag)
        metadata["tables"]["analysis_ready_quarterly"] = {
            "source": {"dataset": "derived_join", "path": None},
            "outputs": paths_a,
            "validation": meta_a,
        }
        ran.append("analysis")
        print(f"\n[run:analysis] rows={out_a.shape[0]} cols={out_a.shape[1]} min={meta_a.get('min_quarter')} max={meta_a.get('max_quarter')}")

    metadata["run_history"].append({"tag": tag, "ran": ran, "python": sys.version.split()[0]})
    write_json(metadata_path, metadata)
    print("\n[part2] Updated:", metadata_path)


if __name__ == "__main__":
    main()
