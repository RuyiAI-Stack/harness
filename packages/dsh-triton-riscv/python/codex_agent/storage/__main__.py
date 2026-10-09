"""Explicit schema upgrades; runtime connections never create or migrate tables."""
import argparse
from pathlib import Path

from alembic import command
from alembic.config import Config
from codex_agent.storage.database import engine


def upgrade():
    config = Config()
    config.set_main_option("script_location", str(Path(__file__).parent / "migrations"))
    with engine().connect() as connection:
        config.attributes["connection"] = connection
        command.upgrade(config, "head")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["upgrade"])
    parser.parse_args()
    try:
        upgrade()
    except Exception:
        raise SystemExit("MySQL schema upgrade failed; check connection and migration configuration") from None
    print("MySQL schema is up to date")


if __name__ == "__main__":
    main()
