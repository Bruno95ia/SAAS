"""Initial schema for cameras, events and frame labels."""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "0001_create_core_tables"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "cameras",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(length=128), nullable=False, unique=True),
        sa.Column("rtsp", sa.String(length=512), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
    )

    op.create_table(
        "events",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("camera_id", sa.Integer(), sa.ForeignKey("cameras.id", ondelete="CASCADE"), nullable=False),
        sa.Column("start_ts", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
        sa.Column("end_ts", sa.DateTime(), nullable=True),
        sa.Column("label", sa.String(length=64), nullable=False),
        sa.Column("score", sa.Numeric(5, 2), nullable=True),
        sa.Column("clip_path", sa.String(length=512), nullable=True),
    )

    op.create_table(
        "frame_labels",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("event_id", sa.Integer(), sa.ForeignKey("events.id", ondelete="CASCADE"), nullable=False),
        sa.Column("ts", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
        sa.Column("x1", sa.Integer(), nullable=False),
        sa.Column("y1", sa.Integer(), nullable=False),
        sa.Column("x2", sa.Integer(), nullable=False),
        sa.Column("y2", sa.Integer(), nullable=False),
        sa.Column("cls", sa.String(length=64), nullable=False),
        sa.Column("score", sa.Numeric(5, 2), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("frame_labels")
    op.drop_table("events")
    op.drop_table("cameras")
