-- TDMEC relational schema (RDBMS layer). AGE graphs are created per-run in Python.
CREATE SCHEMA IF NOT EXISTS tdmec;

CREATE TABLE IF NOT EXISTS tdmec.runs (
  run_id TEXT PRIMARY KEY,
  pipeline TEXT NOT NULL,
  config_hash TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'UNVALIDATED',
  artifact_status TEXT NOT NULL DEFAULT 'PROVISIONAL',
  age_graph_name TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  meta JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS tdmec.source_files (
  run_id TEXT NOT NULL REFERENCES tdmec.runs(run_id) ON DELETE CASCADE,
  source_file TEXT NOT NULL,
  sha256 TEXT NOT NULL,
  size_bytes BIGINT,
  PRIMARY KEY (run_id, source_file)
);

CREATE TABLE IF NOT EXISTS tdmec.file_chunks (
  run_id TEXT NOT NULL REFERENCES tdmec.runs(run_id) ON DELETE CASCADE,
  source_file TEXT NOT NULL,
  chunk_idx INTEGER NOT NULL,
  rows_inspected INTEGER,
  events_inserted INTEGER,
  completed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  PRIMARY KEY (run_id, source_file, chunk_idx)
);

CREATE TABLE IF NOT EXISTS tdmec.nodes (
  node_index INTEGER PRIMARY KEY,
  author_account_id TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS tdmec.snapshots (
  snapshot_id INTEGER PRIMARY KEY,
  quarter_label TEXT NOT NULL,
  start_utc TIMESTAMPTZ NOT NULL,
  end_utc_exclusive TIMESTAMPTZ NOT NULL,
  status TEXT NOT NULL DEFAULT 'provisional'
);

CREATE TABLE IF NOT EXISTS tdmec.events (
  run_id TEXT NOT NULL REFERENCES tdmec.runs(run_id) ON DELETE CASCADE,
  signature TEXT NOT NULL,
  snapshot_id INTEGER NOT NULL,
  relation_id SMALLINT NOT NULL,
  source_idx INTEGER NOT NULL,
  target_idx INTEGER NOT NULL,
  cleaned_text TEXT,
  text_hash TEXT,
  text_quality TEXT,
  source_file TEXT,
  source_row_number BIGINT,
  PRIMARY KEY (run_id, signature)
);

CREATE INDEX IF NOT EXISTS events_edge_idx
  ON tdmec.events (run_id, snapshot_id, relation_id, source_idx, target_idx);

CREATE TABLE IF NOT EXISTS tdmec.authored_tweets (
  run_id TEXT NOT NULL REFERENCES tdmec.runs(run_id) ON DELETE CASCADE,
  source_file TEXT NOT NULL,
  source_row_number BIGINT NOT NULL,
  snapshot_id INTEGER NOT NULL,
  source_idx INTEGER NOT NULL,
  PRIMARY KEY (run_id, source_file, source_row_number)
);

CREATE TABLE IF NOT EXISTS tdmec.edges (
  run_id TEXT NOT NULL REFERENCES tdmec.runs(run_id) ON DELETE CASCADE,
  snapshot_id INTEGER NOT NULL,
  relation_id SMALLINT NOT NULL,
  src_index INTEGER NOT NULL,
  dst_index INTEGER NOT NULL,
  count_raw BIGINT NOT NULL,
  weight_log1p DOUBLE PRECISION NOT NULL,
  PRIMARY KEY (run_id, snapshot_id, relation_id, src_index, dst_index)
);

CREATE INDEX IF NOT EXISTS edges_src_idx
  ON tdmec.edges (run_id, snapshot_id, relation_id, src_index);
CREATE INDEX IF NOT EXISTS edges_dst_idx
  ON tdmec.edges (run_id, snapshot_id, relation_id, dst_index);

CREATE TABLE IF NOT EXISTS tdmec.node_text_units (
  run_id TEXT NOT NULL REFERENCES tdmec.runs(run_id) ON DELETE CASCADE,
  tweet_id TEXT NOT NULL,
  node_index INTEGER,
  snapshot_id INTEGER,
  cleaned_text TEXT,
  text_normalized TEXT,
  text_hash TEXT,
  text_quality TEXT,
  record_status TEXT NOT NULL,
  source_file TEXT NOT NULL,
  source_row_number BIGINT NOT NULL,
  artifact_status TEXT NOT NULL DEFAULT 'PROVISIONAL',
  PRIMARY KEY (run_id, source_file, source_row_number)
);

CREATE INDEX IF NOT EXISTS node_text_units_node_snap_idx
  ON tdmec.node_text_units (run_id, node_index, snapshot_id)
  WHERE record_status = 'retained';

-- pgvector embedding tables (empty until authorized pilot)
CREATE TABLE IF NOT EXISTS tdmec.node_tweet_embeddings (
  run_id TEXT NOT NULL REFERENCES tdmec.runs(run_id) ON DELETE CASCADE,
  tweet_id TEXT NOT NULL,
  model_hash TEXT NOT NULL,
  dim INTEGER NOT NULL,
  embedding vector,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  PRIMARY KEY (run_id, tweet_id, model_hash)
);

CREATE TABLE IF NOT EXISTS tdmec.edge_event_embeddings (
  run_id TEXT NOT NULL REFERENCES tdmec.runs(run_id) ON DELETE CASCADE,
  signature TEXT NOT NULL,
  model_hash TEXT NOT NULL,
  dim INTEGER NOT NULL,
  embedding vector,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  PRIMARY KEY (run_id, signature, model_hash)
);

CREATE TABLE IF NOT EXISTS tdmec.node_snapshot_embeddings (
  run_id TEXT NOT NULL REFERENCES tdmec.runs(run_id) ON DELETE CASCADE,
  snapshot_id INTEGER NOT NULL,
  node_index INTEGER NOT NULL,
  model_hash TEXT NOT NULL,
  dim INTEGER NOT NULL,
  embedding vector,
  valid_count INTEGER NOT NULL DEFAULT 0,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  PRIMARY KEY (run_id, snapshot_id, node_index, model_hash)
);

CREATE TABLE IF NOT EXISTS tdmec.edge_embeddings (
  run_id TEXT NOT NULL REFERENCES tdmec.runs(run_id) ON DELETE CASCADE,
  snapshot_id INTEGER NOT NULL,
  relation_id SMALLINT NOT NULL,
  src_index INTEGER NOT NULL,
  dst_index INTEGER NOT NULL,
  model_hash TEXT NOT NULL,
  dim INTEGER NOT NULL,
  embedding vector,
  valid_count INTEGER NOT NULL DEFAULT 0,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  PRIMARY KEY (run_id, snapshot_id, relation_id, src_index, dst_index, model_hash)
);
