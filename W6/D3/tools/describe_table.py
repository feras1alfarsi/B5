"""Tool for describing table schema."""

from .database import get_db_connection


def describe_table(table_name: str) -> list[tuple[str, str]]:
    """Look up the table schema.
    
    Args:
        table_name: The name of the table to describe.
        
    Returns:
        List of columns, where each entry is a tuple of (column_name, column_type).
    """
    db_conn = get_db_connection()
    cursor = db_conn.cursor()
    # Safely executing PRAGMA with f-string is generally okay for local trusted tools, 
    # but be wary of SQL injection in production with arbitrary user input.
    cursor.execute(f"PRAGMA table_info({table_name});")
    # schema: [cid, name, type, notnull, dflt_value, pk]
    return [(col[1], col[2]) for col in cursor.fetchall()]
