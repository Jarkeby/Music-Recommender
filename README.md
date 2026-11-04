# Music Recommender

A music recommender that gives users a top 10 list of similar artist or songs \
**Demo:** https://music-recommender-jarkeby.streamlit.app/

## Data
- **Source:** ListenBrainz listens (open data, CC0), from the official daily incremental dumps (`data.metabrainz.org/pub/musicbrainz/listenbrainz/incremental/`, Parquet), with MusicBrainz/Last.fm tags.
- **Sample used:** listens from 11 - 25 Oct 2026 (14 days).
  17.1M listens / 40,093 users / 3.94M tracks raw -> 8.77M listens / 27,884 users / 90,885 tracks after filtering.

## Run it
```bash
# uses virtual environment if needed to use Python 3.12.10
python -m venv .venv && source .venv/bin/activate  
pip install -r requirements.txt

# Try pipeline on fake data
python scripts/make_synthetic_data.py
python scripts/run_experiment.py --data data/raw/synthetic.parquet --min-user 5 --min-track 5

# Download the last 14 days of ListenBrainz dumps (~2.4 GB) for real data
python scripts/pull_from_dumps.py --days 14
python scripts/run_experiment.py --save-artifacts

# Hyperparameter search on validation split 
python scripts/tune.py
python scripts/tune.py --models als --grid "{\"factors\": [512, 1024], \"alpha\": [40]}"

# Extension experiments, charts, demo app
python scripts/pull_tags.py
python scripts/cold_start.py
python scripts/hybrid.py
python scripts/make_figures.py
# Pulls artists spotify links, takes a while so use --limit if needed
python scripts/pull_artists_links.py --limit 1000
streamlit run app/streamlit_app.py