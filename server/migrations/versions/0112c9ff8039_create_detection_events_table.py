"""create detection_events hypertable

Revision ID: 0112c9ff8039
Revises:
Create Date: 2026-08-03 15:39:09.591035

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '0112c9ff8039'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create detection_events as a TimescaleDB hypertable, partitioned on timestamp."""
    op.execute("CREATE EXTENSION IF NOT EXISTS timescaledb")

    op.create_table(
        'detection_events',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('camera_node_id', sa.String(), nullable=False),
        sa.Column('timestamp', sa.DateTime(timezone=True), nullable=False),
        sa.Column('ingested_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('zone_id', sa.String(), nullable=True),
        sa.Column('count', sa.Integer(), nullable=True),
        sa.Column('age_group', sa.String(), nullable=False),
        sa.Column('gender', sa.String(), nullable=False),
        sa.Column('emotion', sa.String(), nullable=False),
        sa.Column('dwell_seconds', sa.Float(), nullable=True),
        sa.Column('engagement_score', sa.Float(), nullable=True),
        # Composite primary key, not just id -- TimescaleDB requires the
        # partitioning column (timestamp) in every unique constraint.
        sa.PrimaryKeyConstraint('id', 'timestamp'),
    )

    op.execute("SELECT create_hypertable('detection_events', by_range('timestamp'))")

    op.create_index(
        'ix_detection_events_camera_node_id_timestamp',
        'detection_events',
        ['camera_node_id', sa.text('timestamp DESC')],
    )
    # Partial index: near-free today since zone_id is always null, ready
    # the moment zone configuration starts populating it, no migration
    # needed at that point.
    op.execute(
        "CREATE INDEX ix_detection_events_zone_id_timestamp "
        "ON detection_events (zone_id, timestamp DESC) "
        "WHERE zone_id IS NOT NULL"
    )


def downgrade() -> None:
    """Drop detection_events and its indexes."""
    op.drop_index('ix_detection_events_zone_id_timestamp', table_name='detection_events')
    op.drop_index('ix_detection_events_camera_node_id_timestamp', table_name='detection_events')
    op.drop_table('detection_events')
