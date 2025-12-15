from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import requests


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


@dataclass
class RequestResult:
    status_code: int
    url: str


def safe_request(
    session: requests.Session,
    url: str,
    method: str = "GET",
    params: Optional[Dict[str, Any]] = None,
    json_body: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 60,
    retries: int = 3,
    backoff: float = 1.7,
) -> requests.Response:
    """
    Minimal resilient HTTP request with retry/backoff for transient failures.
    Retries on: 429, 5xx, and network exceptions.
    """
    method = method.upper().strip()
    last_exc: Optional[Exception] = None

    for attempt in range(1, retries + 1):
        try:
            resp = session.request(
                method=method,
                url=url,
                params=params,
                json=json_body,
                headers=headers,
                timeout=timeout,
            )

            if resp.status_code in (429, 500, 502, 503, 504):
                # transient / rate limiting
                if attempt < retries:
                    sleep_s = backoff ** (attempt - 1)
                    time.sleep(sleep_s)
                    continue
            resp.raise_for_status()
            return resp

        except (requests.RequestException,) as exc:
            last_exc = exc
            if attempt < retries:
                sleep_s = backoff ** (attempt - 1)
                time.sleep(sleep_s)
                continue
            raise

    # Defensive: should never reach here
    if last_exc:
        raise last_exc
    raise RuntimeError("safe_request failed without exception (unexpected).")


def save_metadata(metadata: Dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def write_outputs(
    df: pd.DataFrame,
    out_basepath: Path,
    write_parquet: bool = True,
) -> Dict[str, Any]:
    """
    Writes df to CSV and optionally Parquet using out_basepath without suffix.
    Returns dict with output file paths and warnings.
    """
    ensure_dir(out_basepath.parent)
    outputs: Dict[str, Any] = {"files": [], "warnings": []}

    csv_path = out_basepath.with_suffix(".csv")
    df.to_csv(csv_path, index=False)
    outputs["files"].append(str(csv_path))

    if write_parquet:
        pq_path = out_basepath.with_suffix(".parquet")
        try:
            df.to_parquet(pq_path, index=False)
            outputs["files"].append(str(pq_path))
        except Exception as exc:  # noqa: BLE001
            outputs["warnings"].append(f"Parquet write failed: {exc}")

    return outputs
