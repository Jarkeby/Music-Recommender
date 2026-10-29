"""Run sql/pull_sample.sql on BigQuery and save the result as Parquet.

Setup: `gcloud auth application-default login` (or set GOOGLE_APPLICATION_CREDENTIALS),
then edit sql/pull_sample.sql. Usage:
    python scripts/pull_from_bigquery.py --project YOUR_GCP_PROJECT
"""
import argparse
from pathlib import Path
from google.cloud import bigquery


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project", required=True, help="Your GCP project (billed to your free allowance)")
    p.add_argument("--sql", default="sql/pull_sample.sql")
    p.add_argument("--out", default="data/raw/listens.parquet")
    args = p.parse_args()

    client = bigquery.Client(project=args.project)
    query = Path(args.sql).read_text()
    df = client.query(query).to_dataframe()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"Saved {len(df):,} rows to {args.out}")


if __name__ == "__main__":
    main()
