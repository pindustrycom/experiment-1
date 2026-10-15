"""Append-only guarantees for research records.

Prediction snapshots, realized outcomes, index scores and their component calculations are
reproducibility records: once written they can never be updated or deleted, even by buggy
application code. Changes to methodology create new model versions and new rows instead.

Revision ID: 0002
Revises: 0001
"""
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None

TABLES = ["prediction_snapshots", "realized_outcomes", "index_scores", "si_component_calculations", "model_versions"]


def upgrade() -> None:
    op.execute(
        """
        CREATE OR REPLACE FUNCTION sm_reject_modification() RETURNS trigger AS $$
        BEGIN
            -- model_versions may toggle is_active only
            IF TG_TABLE_NAME = 'model_versions' AND TG_OP = 'UPDATE'
               AND NEW.index_name = OLD.index_name AND NEW.version = OLD.version
               AND NEW.weights = OLD.weights AND NEW.params = OLD.params
               AND NEW.content_hash = OLD.content_hash THEN
                RETURN NEW;
            END IF;
            RAISE EXCEPTION 'table % is append-only (% rejected)', TG_TABLE_NAME, TG_OP
                USING ERRCODE = 'integrity_constraint_violation';
        END;
        $$ LANGUAGE plpgsql;
        """
    )
    for t in TABLES:
        op.execute(
            f"CREATE TRIGGER {t}_append_only BEFORE UPDATE OR DELETE ON {t} "
            f"FOR EACH ROW EXECUTE FUNCTION sm_reject_modification();"
        )


def downgrade() -> None:
    for t in TABLES:
        op.execute(f"DROP TRIGGER IF EXISTS {t}_append_only ON {t};")
    op.execute("DROP FUNCTION IF EXISTS sm_reject_modification();")
