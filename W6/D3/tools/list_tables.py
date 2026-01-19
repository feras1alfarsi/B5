"""Tool for listing all tables in the database."""

from .database import get_db_connection


def list_tables() -> list[str]:
    """Retrieve the names of all tables in the database."""
    db_conn = get_db_connection()
    cursor = db_conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    return [t[0] for t in cursor.fetchall()]
