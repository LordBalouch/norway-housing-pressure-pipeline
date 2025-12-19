from __future__ import annotations

import argparse
from pathlib import Path

import duckdb


def _read_sql(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _strip_line_comments(sql_text: str) -> str:
    # Remove '-- ...' comments to prevent semicolons inside comments from breaking statement splitting
    clean_lines = []
    for line in sql_text.splitlines():
        if "--" in line:
            line = line.split("--", 1)[0]
        clean_lines.append(line)
    return "\n".join(clean_lines)


def _run_multi_select(con: duckdb.DuckDBPyConnection, sql_path: Path, title: str) -> None:
    raw = _read_sql(sql_path)
    clean = _strip_line_comments(raw)
    statements = [s.strip() for s in clean.split(";") if s.strip()]

    print(f"\n== {title} ({sql_path.as_posix()}) ==")
    for i, stmt in enumerate(statements, 1):
        print(f"\n--- Result {i} ---")
        df = con.execute(stmt).fetchdf()
        print(df.to_string(index=False))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run DuckDB SQL scripts and export the housing pressure mart.")
    parser.add_argument("--db", default="data/processed/project.duckdb", help="Path to DuckDB database file.")
    parser.add_argument("--build-sql", default="sql/01_build_mart.sql", help="SQL script to build mart table.")
    parser.add_argument("--validation-sql", default="sql/02_validation_checks.sql", help="SQL script for validation checks.")
    parser.add_argument("--eda-sql", default="sql/03_eda_queries.sql", help="SQL script for EDA queries.")
    parser.add_argument("--run-validation", action="store_true", help="Run validation checks after building the mart.")
    parser.add_argument("--run-eda", action="store_true", help="Run EDA queries after building the mart.")
    parser.add_argument("--out-csv", default="data/processed/mart_housing_pressure_quarterly.csv", help="CSV export path.")
    parser.add_argument("--out-parquet", default="data/processed/mart_housing_pressure_quarterly.parquet", help="Parquet export path.")
    parser.add_argument("--no-parquet", action="store_true", help="Skip parquet export.")
    args = parser.parse_args()

    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    build_sql_path = Path(args.build_sql)
    validation_sql_path = Path(args.validation_sql)
    eda_sql_path = Path(args.eda_sql)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    out_parquet = Path(args.out_parquet)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(db_path))

    # Build mart (01_build_mart.sql ends with a SELECT summary; fetch and print it)
    print(f"Using DuckDB file: {db_path.as_posix()}")
    build_sql = _read_sql(build_sql_path)
    summary_df = con.execute(build_sql).fetchdf()
    print("\nMart build summary:")
    print(summary_df.to_string(index=False))

    # Optional: run validation / EDA
    if args.run_validation:
        _run_multi_select(con, validation_sql_path, "VALIDATION CHECKS")

    if args.run_eda:
        _run_multi_select(con, eda_sql_path, "EDA QUERIES")

    # Export mart to CSV (+ optional parquet)
    con.execute(
        f"""
        COPY (
          SELECT *
          FROM mart_housing_pressure_quarterly
          ORDER BY boligtype_code, quarter_start
        )
        TO '{out_csv.as_posix()}'
        (HEADER, DELIMITER ',');
        """
    )
    print(f"\nExported CSV: {out_csv.as_posix()}")

    if not args.no_parquet:
        con.execute(
            f"""
            COPY (
              SELECT *
              FROM mart_housing_pressure_quarterly
              ORDER BY boligtype_code, quarter_start
            )
            TO '{out_parquet.as_posix()}'
            (FORMAT PARQUET);
            """
        )
        print(f"Exported Parquet: {out_parquet.as_posix()}")

    # Quick export verification
    row_count = con.execute("SELECT COUNT(*) AS n FROM mart_housing_pressure_quarterly").fetchone()[0]
    print(f"\nMart row_count in DuckDB: {row_count}")

    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
