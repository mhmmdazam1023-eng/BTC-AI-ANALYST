"""Database health checks shared by the scripts.

Currently one check, but the one that matters most in practice: whether the
application role can actually read and write the tables it is about to use.

Why this deserves its own preflight
-----------------------------------
``GRANT ALL ON SCHEMA public`` — which ``setup_database.sql`` issues — lets
a role CREATE objects in that schema. It grants nothing on tables another
role already created. So the moment any table ends up owned by ``postgres``
(a manual ``psql`` session, a restore, an ``init_db`` run under different
credentials), the app loses access to that table completely while every
other table keeps working.

The symptom is misleading: a daily run gets most of the way through, then
emits a wall of ``permission denied for table ...`` tracebacks from inside
several isolated collector stages at once. It reads like a code fault. This
check turns it into one line naming every affected table and the exact
command that fixes it.
"""

from __future__ import annotations

import logging

from sqlalchemy import text

from src.db.engine import session_scope

logger = logging.getLogger(__name__)

#: Query returning every public table the current role cannot fully use.
_INACCESSIBLE_TABLES_SQL = """
    SELECT c.relname,
           pg_get_userbyid(c.relowner) AS owner,
           has_table_privilege(current_user, c.oid, 'SELECT') AS can_select,
           has_table_privilege(current_user, c.oid, 'INSERT') AS can_insert
    FROM pg_class c
    JOIN pg_namespace n ON n.oid = c.relnamespace
    WHERE n.nspname = 'public'
      AND c.relkind = 'r'
      AND NOT (
          has_table_privilege(current_user, c.oid, 'SELECT')
          AND has_table_privilege(current_user, c.oid, 'INSERT')
      )
    ORDER BY c.relname
"""


def privilege_fix_hint(database: str) -> str:
    """The exact command that repairs a broken grant set."""
    return (
        f"psql -U postgres -h localhost -d {database} "
        f"-f scripts/fix_permissions.sql"
    )


async def find_inaccessible_tables() -> list[tuple[str, str, bool, bool]]:
    """``[(table, owner, can_select, can_insert), ...]`` — empty when healthy."""
    async with session_scope() as session:
        rows = (await session.execute(text(_INACCESSIBLE_TABLES_SQL))).all()
    return [(row[0], row[1], row[2], row[3]) for row in rows]


async def assert_tables_accessible(database: str) -> None:
    """Raise with an actionable message when any table is unusable."""
    inaccessible = await find_inaccessible_tables()
    if not inaccessible:
        logger.info(
            "Table privileges OK: every public table is readable & writable"
        )
        return

    for name, owner, can_select, can_insert in inaccessible:
        logger.error(
            "NO ACCESS: %-28s owner=%-14s select=%s insert=%s",
            name, owner, can_select, can_insert,
        )
    raise PermissionError(
        f"{len(inaccessible)} table(s) are not readable/writable by the "
        f"application role: {', '.join(name for name, *_ in inaccessible)}.\n"
        f"GRANT ON SCHEMA alone does NOT cover tables created by another "
        f"role — they need an explicit table-level GRANT.\n"
        f"Fix (as the postgres superuser):\n    {privilege_fix_hint(database)}"
    )
