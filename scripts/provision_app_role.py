"""
Provision the API's least-privilege database role `iia_app` (Phase 2, RLS).

    APP_DB_PASSWORD=... python scripts/provision_app_role.py
    python scripts/provision_app_role.py --grants-only      # no password change

Connects with DATABASE_URL (the OWNER/admin role; needs CREATEROLE or
superuser the first time), runs the idempotent iia_grant_app_privileges()
created by migration 0006, and gives the role LOGIN + the password from the
required APP_DB_PASSWORD env var (no default). scripts/restore_db.py calls
--grants-only after a restore, because pg_restore --no-privileges drops
grants. Point the API at the role with APP_DATABASE_URL.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def provision(url: str, password: str | None) -> None:
    engine = create_engine(url)
    with engine.begin() as conn:
        conn.execute(text("SELECT iia_grant_app_privileges()"))
        if password is not None:
            stmt = conn.execute(
                text(
                    "SELECT format('ALTER ROLE iia_app LOGIN PASSWORD %L', CAST(:pw AS text))"),
                {"pw": password}).scalar()
            conn.exec_driver_sql(stmt.replace("%", "%%"))
    engine.dispose()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--grants-only", action="store_true",
                        help="Re-apply grants only; leave LOGIN/password alone.")
    args = parser.parse_args()

    load_dotenv()
    url = os.environ.get("DATABASE_URL")
    if not url:
        logger.error("DATABASE_URL not set (the owner/admin connection).")
        return 1
    password = None
    if not args.grants_only:
        password = os.environ.get("APP_DB_PASSWORD")
        if not password:
            logger.error(
                "APP_DB_PASSWORD not set (required; there is no default).")
            return 1
    provision(url, password)
    logger.info("iia_app grants applied%s.",
                "" if args.grants_only else " and LOGIN password set")
    return 0


if __name__ == "__main__":
    sys.exit(main())
