"""add track appearances

Holds one appearance vector per (camera node, track), so a person seen by
two cameras that never had them in view at the same moment can still be
recognised as one person. See app/models/appearance.py for why this is one
row per track rather than per detection, and app/routers/ingest.py for the
retention window that prunes it.

Revision ID: 2f9186e62cf0
Revises: c9d3e57a1f24
Create Date: 2026-08-26 16:25:00.351751

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "2f9186e62cf0"
down_revision: str | None = "c9d3e57a1f24"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create the track_appearances table and the index retention prunes against."""
    op.create_table(
        "track_appearances",
        sa.Column("camera_node_id", sa.String(), nullable=False),
        sa.Column("track_id", sa.String(), nullable=False),
        # JSON rather than a Postgres array or a pgvector column: comparison
        # happens in Python over the tracks in one query's range, so there is
        # nothing yet to gain from in-database distance operators, and JSON is
        # the representation that behaves the same on the SQLite used in tests.
        sa.Column("embedding", sa.JSON(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        # A track is identified by its node and its id together; neither is
        # unique alone, since nodes number their tracks independently.
        sa.PrimaryKeyConstraint("camera_node_id", "track_id"),
    )
    # Retention deletes by age on every ingest batch, so that predicate is the
    # one access path that must not degrade as the table grows.
    op.create_index("ix_track_appearances_updated_at", "track_appearances", ["updated_at"])

    # Deliberately not a hypertable. It is small and upsert-heavy -- one row
    # per track, rewritten as the node refines it -- which is the opposite of
    # the append-only, time-partitioned shape hypertables exist for.


def downgrade() -> None:
    """Drop the table and its index."""
    op.drop_index("ix_track_appearances_updated_at", table_name="track_appearances")
    op.drop_table("track_appearances")
