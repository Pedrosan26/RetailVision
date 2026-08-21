"""add zone geometry table

Camera nodes compute zone polygons from the surveyed marker map, which
the server never sees. Storing the geometry the nodes upload at startup
lets the dashboard draw a floor map that world positions can land on --
without it, positions are numbers with nothing to be inside of.

One row per zone, last writer wins: every node loads the same surveyed
map file, so a disagreement means a stale node rather than a conflict.

Revision ID: c9d3e57a1f24
Revises: b7e2d19f4c33
Create Date: 2026-08-24 12:20:41.221540

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'c9d3e57a1f24'
down_revision: Union[str, Sequence[str], None] = 'b7e2d19f4c33'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create the zone_geometry table."""
    op.create_table(
        'zone_geometry',
        sa.Column('zone_id', sa.String(), primary_key=True),
        sa.Column('camera_node_id', sa.String(), nullable=False),
        sa.Column('polygon', sa.JSON(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )


def downgrade() -> None:
    """Drop the zone_geometry table."""
    op.drop_table('zone_geometry')
