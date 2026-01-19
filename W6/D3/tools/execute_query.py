"""Tool for executing SQL queries."""

from .database import get_db_connection


def execute_query(sql: str) -> list[list[str]]:
    """Execute an SQL SELECT statement and return the results.
    
    Args:
        sql: The SQL query to execute.
        
    Returns:
        A list of rows, where each row is a list of string values.
    """
    db_conn = get_db_connection()
    cursor = db_conn.cursor()
    cursor.execute(sql)
    return cursor.fetchall()
