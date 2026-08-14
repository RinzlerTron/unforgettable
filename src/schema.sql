-- Unforgettable memory schema for CockroachDB.
-- Applied idempotently by src/db.py ensure_schema() at startup.
--
-- Four tables, all plain SQL rows:
--   episodes      - episodic memory: every conversation event, verbatim
--   facts         - semantic memory, APPEND-ONLY and versioned: each row is
--                   one belief version with valid_from/superseded_at, so the
--                   agent's belief state at any past moment is reconstructable
--                   forever - and recent moments can also be read natively
--                   with AS OF SYSTEM TIME (see src/timetravel.py)
--   tasks         - task state the agent has been asked to track
--   recall_traces - decision audit: which memory rows each reply was given
--
-- Embeddings use CockroachDB's native VECTOR type with distributed vector
-- indexes (created by db.py after these tables exist).

CREATE TABLE IF NOT EXISTS conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title STRING NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS episodes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID NOT NULL REFERENCES conversations (id),
    role STRING NOT NULL,
    content STRING NOT NULL,
    embedding VECTOR(256),
    embedding_model STRING NOT NULL DEFAULT 'local-hash-v1',
    meta JSONB NOT NULL DEFAULT '{}',
    consolidated_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT episodes_role_check CHECK (role IN ('user', 'assistant', 'system')),
    INDEX episodes_conv_time_idx (conversation_id, created_at)
);

CREATE TABLE IF NOT EXISTS facts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    subject STRING NOT NULL,
    content STRING NOT NULL,
    confidence FLOAT NOT NULL DEFAULT 0.8,
    provenance JSONB NOT NULL DEFAULT '{}',
    embedding VECTOR(256),
    embedding_model STRING NOT NULL DEFAULT 'local-hash-v1',
    replaces_id UUID,
    valid_from TIMESTAMPTZ NOT NULL DEFAULT now(),
    superseded_at TIMESTAMPTZ,
    CONSTRAINT facts_confidence_check CHECK (confidence >= 0.0 AND confidence <= 1.0),
    INDEX facts_subject_idx (subject, superseded_at),
    INDEX facts_validity_idx (valid_from, superseded_at)
);

CREATE TABLE IF NOT EXISTS tasks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID REFERENCES conversations (id),
    title STRING NOT NULL,
    status STRING NOT NULL DEFAULT 'open',
    payload JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT tasks_status_check CHECK (status IN ('open', 'done', 'cancelled')),
    INDEX tasks_status_idx (status, updated_at)
);

CREATE TABLE IF NOT EXISTS recall_traces (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID NOT NULL,
    user_episode_id UUID NOT NULL,
    reply_episode_id UUID,
    query STRING NOT NULL,
    recalled JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    INDEX recall_traces_reply_idx (reply_episode_id),
    INDEX recall_traces_conv_idx (conversation_id, created_at)
);
