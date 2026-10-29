-- Pull a sample of listens from ListenBrainz on BigQuery.
-- NOTE (Oct 2026): the current ListenBrainz docs no longer mention BigQuery, only data dumps.
-- The project uses scripts/pull_from_dumps.py instead (free, no GCP account). Keep this file
-- only if you find a live BigQuery table; confirm its project/dataset/table and column names.
--
-- Goal: a flat table with columns  user_id | track_id | listened_at
-- track_id should be a stable identifier (MusicBrainz recording MBID if available;
-- otherwise something like LOWER(artist || ' - ' || track_name)).

SELECT
  user_name                              AS user_id,
  COALESCE(recording_mbid, LOWER(CONCAT(artist_name, ' - ', track_name))) AS track_id,
  listened_at                            AS listened_at
FROM `YOUR_PROJECT.YOUR_DATASET.listen`   -- TODO: replace
WHERE listened_at >= TIMESTAMP('2024-01-01')
  AND listened_at <  TIMESTAMP('2024-02-01')   -- start with ONE month
  AND track_name IS NOT NULL
  AND artist_name IS NOT NULL
LIMIT 3000000;  -- keep it small while developing
