"""add world position to detection_events

Camera nodes running with marker-based zones report where each person was
standing, in the shared world frame every camera watching that zone agrees
on. Without it the server can only sum per-camera counts, which
double-counts anyone visible to more than one camera -- and with several
cameras covering one area, that is most people.

Nullable because nodes running without zones have no position to report,
and because a node that has not been upgraded keeps ingesting unchanged.

Revision ID: a3f1c47d2b90
Revises: 0112c9ff8039
Create Date: 2026-08-18 10:12:44.108221

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'a3f1c47d2b90'
down_revision: Union[str, Sequence[str], None] = '0112c9ff8039'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add the world_x/world_y floor position columns."""
    op.add_column('detection_events', sa.Column('world_x', sa.Float(), nullable=True))
    op.add_column('detection_events', sa.Column('world_y', sa.Float(), nullable=True))

    # Deduplication reads a short recent window for one zone at a time, so the
    # ordering that matters is by zone then time. The existing zone_id index is
    # partial on zone_id IS NOT NULL, which is exactly the rows that carry a
    # position, so no additional index is needed here.


def downgrade() -> None:
    """Drop the world position columns."""
    op.drop_column('detection_events', 'world_y')
    op.drop_column('detection_events', 'world_x')
