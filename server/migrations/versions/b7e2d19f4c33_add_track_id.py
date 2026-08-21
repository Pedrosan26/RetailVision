"""add track id to detection_events

Camera nodes now report one record per person per change rather than one
per person per frame, and carry the track that groups those records
together. Without it every row is its own person, so a person who stayed
five minutes is indistinguishable from a crowd who passed through, and
any count of people is really a count of frames.

The column is only unique within a camera node and a run of its process.
Recognising the same person across two cameras stays the server's spatial
job, using the shared-world-frame position -- this identifies a person
within one camera, not across the deployment.

Nullable because a node that has not been upgraded keeps ingesting
unchanged; those rows count as one person each, exactly as before.

Revision ID: b7e2d19f4c33
Revises: a3f1c47d2b90
Create Date: 2026-08-19 19:41:02.554310

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'b7e2d19f4c33'
down_revision: Union[str, Sequence[str], None] = 'a3f1c47d2b90'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add the track_id column and the index the per-person rollups read it through."""
    op.add_column('detection_events', sa.Column('track_id', sa.String(), nullable=True))
    # Counting distinct people means grouping by node and track over a time
    # range, which is the order this index puts them in. Partial on the rows
    # that carry a track, since pre-upgrade rows are never grouped this way.
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_detection_events_track "
        "ON detection_events (camera_node_id, track_id, timestamp DESC) "
        "WHERE track_id IS NOT NULL"
    )


def downgrade() -> None:
    """Drop the track_id column and its index."""
    op.execute("DROP INDEX IF EXISTS ix_detection_events_track")
    op.drop_column('detection_events', 'track_id')
