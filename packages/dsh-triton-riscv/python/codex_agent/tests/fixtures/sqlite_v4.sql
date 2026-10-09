-- Frozen SQLite schemas from commit 1a1be6b, used only to test legacy imports.
CREATE TABLE IF NOT EXISTS sessions (
id TEXT PRIMARY KEY,
title TEXT NOT NULL,
status TEXT NOT NULL,
created_at TEXT NOT NULL,
updated_at TEXT NOT NULL,
pinned INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS messages (
id TEXT PRIMARY KEY,
session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
role TEXT NOT NULL,
content TEXT NOT NULL,
metadata_json TEXT NOT NULL,
created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, created_at);
CREATE TABLE IF NOT EXISTS runs (
id TEXT PRIMARY KEY,
session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
intent TEXT NOT NULL,
operator TEXT,
status TEXT NOT NULL,
phase TEXT NOT NULL,
request_json TEXT NOT NULL,
result_json TEXT NOT NULL,
created_at TEXT NOT NULL,
updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_runs_session ON runs(session_id, created_at);
CREATE TABLE IF NOT EXISTS events (
id INTEGER PRIMARY KEY AUTOINCREMENT,
run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
sequence INTEGER NOT NULL,
event_type TEXT NOT NULL,
payload_json TEXT NOT NULL,
created_at TEXT NOT NULL,
UNIQUE(run_id, sequence)
);
CREATE INDEX IF NOT EXISTS idx_events_run ON events(run_id, sequence);
CREATE TABLE IF NOT EXISTS context_checkpoints (
session_id TEXT PRIMARY KEY REFERENCES sessions(id) ON DELETE CASCADE,
summary TEXT NOT NULL,
through_message_id TEXT,
source_ids_json TEXT NOT NULL,
metrics_json TEXT NOT NULL,
updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS metadata (
key TEXT PRIMARY KEY,
value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS memories (
id INTEGER PRIMARY KEY AUTOINCREMENT,
fingerprint TEXT NOT NULL UNIQUE,
memory_type TEXT NOT NULL,
operator TEXT NOT NULL,
semantics TEXT NOT NULL,
pytorch_reference TEXT NOT NULL,
tl_ops_json TEXT NOT NULL,
failure_stage TEXT,
error_signature TEXT,
environment_json TEXT NOT NULL,
summary TEXT NOT NULL,
searchable_text TEXT NOT NULL,
outcome TEXT NOT NULL,
evidence_json TEXT NOT NULL,
confidence_grade TEXT NOT NULL,
confidence REAL NOT NULL,
source_run TEXT NOT NULL,
test_sha256 TEXT,
active INTEGER NOT NULL DEFAULT 1,
archive_reason TEXT,
created_at TEXT NOT NULL,
updated_at TEXT NOT NULL,
last_used_at TEXT,
useful_count INTEGER NOT NULL DEFAULT 0,
embedding_json TEXT,
embedding_provider TEXT,
embedding_model TEXT,
embedding_dim INTEGER
);
CREATE INDEX IF NOT EXISTS idx_memories_active ON memories(active);
CREATE INDEX IF NOT EXISTS idx_memories_operator ON memories(operator);
CREATE INDEX IF NOT EXISTS idx_memories_stage ON memories(failure_stage);
CREATE INDEX IF NOT EXISTS idx_memories_error ON memories(error_signature);
CREATE TABLE IF NOT EXISTS memory_chunks (
memory_id INTEGER NOT NULL,
kind TEXT NOT NULL,
position INTEGER NOT NULL,
text TEXT NOT NULL,
source_field TEXT NOT NULL DEFAULT '',
embedding_json TEXT,
embedding_provider TEXT,
embedding_model TEXT,
embedding_dim INTEGER,
PRIMARY KEY (memory_id, kind, position)
);
CREATE INDEX IF NOT EXISTS idx_memory_chunks_memory ON memory_chunks(memory_id);
CREATE TABLE references_catalog (id TEXT PRIMARY KEY, admitted INTEGER NOT NULL, payload TEXT NOT NULL);
