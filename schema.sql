-- ForenScreen: constraints SQLAlchemy cannot express.
--
-- Run through scripts/migrate.py, which applies this file after the tables have
-- been created from models.py. Every statement is written to be safe to re-run.

-- ---------------------------------------------------------------------------
-- Column reconciliation.
--
-- SQLAlchemy's create_all only checks whether a TABLE exists, never whether its
-- columns are up to date. A table created before a column was added to models.py
-- is therefore silently left stale, and the app fails at runtime instead of at
-- migration time. New columns must be added here explicitly.
-- ---------------------------------------------------------------------------
ALTER TABLE custody_entries
    ADD COLUMN IF NOT EXISTS seq BIGINT GENERATED ALWAYS AS IDENTITY;

CREATE UNIQUE INDEX IF NOT EXISTS uq_custody_seq ON custody_entries (seq);

-- Per-account ownership. original_assets is NOT NULL because registering
-- requires an account; evidence_checks allows NULL for anonymous public checks.
-- The NOT NULL is added only when every row already has an owner, so this file
-- stays safe to re-run against older data.
ALTER TABLE original_assets ADD COLUMN IF NOT EXISTS owner VARCHAR(128);
ALTER TABLE evidence_checks ADD COLUMN IF NOT EXISTS owner VARCHAR(128);

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM original_assets WHERE owner IS NULL) THEN
        ALTER TABLE original_assets ALTER COLUMN owner SET NOT NULL;
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS ix_original_assets_owner ON original_assets (owner);
CREATE INDEX IF NOT EXISTS ix_evidence_checks_owner ON evidence_checks (owner);

-- The earlier, similar original this one was registered after (see models.py).
ALTER TABLE original_assets
    ADD COLUMN IF NOT EXISTS similar_to UUID REFERENCES original_assets (id);

-- Registration context, labelled by source (see metadata.py).
ALTER TABLE original_assets ADD COLUMN IF NOT EXISTS registration_metadata JSONB;

-- ---------------------------------------------------------------------------
-- custody_entries is append-only.
--
-- Hashed fields can never change: an entry that could be edited would make the
-- whole hash chain meaningless. The only permitted mutation is filling in the
-- on-chain receipt (anchor_tx_hash / anchor_block / anchor_status), which the
-- background confirmation poller needs and which is outside the hashed
-- material. Anything else raises, so an accidental UPDATE fails loudly rather
-- than quietly breaking the chain.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION custody_entries_guard_mutation()
RETURNS trigger AS $$
BEGIN
    IF TG_OP = 'DELETE' THEN
        RAISE EXCEPTION
            'custody_entries is append-only: refusing to DELETE entry %', OLD.id;
    END IF;

    IF (OLD.id, OLD.seq, OLD.subject_type, OLD.subject_id, OLD.event_type, OLD.actor,
        OLD.occurred_at, OLD.payload, OLD.prev_hash, OLD.entry_hash)
       IS DISTINCT FROM
       (NEW.id, NEW.seq, NEW.subject_type, NEW.subject_id, NEW.event_type, NEW.actor,
        NEW.occurred_at, NEW.payload, NEW.prev_hash, NEW.entry_hash)
    THEN
        RAISE EXCEPTION
            'custody_entries is append-only: only anchor_* columns may change (entry %)',
            OLD.id;
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS custody_entries_append_only ON custody_entries;
CREATE TRIGGER custody_entries_append_only
    BEFORE UPDATE OR DELETE ON custody_entries
    FOR EACH ROW EXECUTE FUNCTION custody_entries_guard_mutation();
