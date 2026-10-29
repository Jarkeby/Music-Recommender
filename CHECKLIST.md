# Your build checklist

## Week 1: data + pipeline
- [x] Run the synthetic pipeline end to end (proves your setup works)
- [x] ~~Create GCP account, find the ListenBrainz BigQuery table~~ BigQuery no longer documented -> use official dumps (scripts/pull_from_dumps.py)
- [x] Pull 14 days of data -> data/raw/listens.parquet
- [x] EDA in notebooks/: listens per user, tracks per user, popularity skew (long tail plot)
- [x] Re-run run_experiment.py on real data

## Week 2: models + evaluation
- [x] Get the results table; understand why popularity does/doesn't win
- [x] Tune ALS (factors, regularization, alpha); log each run
- [x] Add BPR (implicit.bpr) as a 4th model
- [x] Break results down by user activity level

## Week 3: extension
- [x] Pull tags (MusicBrainz/Last.fm) for tracks (scripts/pull_tags.py, via ListenBrainz metadata API)
- [x] Cold-start experiment: hold out some tracks entirely, recommend them via tag similarity
- [x] Hybrid: blend ALS score + content score

## Week 4: ship it
- [x] Improve the Streamlit app (show artist/title, not raw IDs)
- [x] Deploy (Streamlit Community Cloud is free)
- [x] Fill in README with real numbers + charts
- [x] Write resume bullets with your actual metrics (RESUME.md)
