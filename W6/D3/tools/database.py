"""Database connection and setup utilities."""

import sqlite3
from pathlib import Path

DB_FILE = Path(__file__).parent.parent / "sample.db"
_db_conn = None


def get_db_connection() -> sqlite3.Connection:
    """Get or create the database connection."""
    global _db_conn
    if _db_conn is None:
        _db_conn = sqlite3.connect(DB_FILE)
    return _db_conn


def setup_database():
    """Create and populate the sample database."""
    db_conn = get_db_connection()
    cursor = db_conn.cursor()
    
    # Reset tables if they exist
    cursor.execute("DROP TABLE IF EXISTS orders")
    cursor.execute("DROP TABLE IF EXISTS products")
    cursor.execute("DROP TABLE IF EXISTS staff")
    
    # Create tables
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS products (
        product_id INTEGER PRIMARY KEY AUTOINCREMENT,
        product_name VARCHAR(255) NOT NULL,
        price DECIMAL(10, 2) NOT NULL
    )
    """)
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS staff (
        staff_id INTEGER PRIMARY KEY AUTOINCREMENT,
        first_name VARCHAR(255) NOT NULL,
        last_name VARCHAR(255) NOT NULL
    )
    """)
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS orders (
        order_id INTEGER PRIMARY KEY AUTOINCREMENT,
        customer_name VARCHAR(255) NOT NULL,
        staff_id INTEGER NOT NULL,
        product_id INTEGER NOT NULL,
        FOREIGN KEY (staff_id) REFERENCES staff (staff_id),
        FOREIGN KEY (product_id) REFERENCES products (product_id)
    )
    """)
    
    # Insert data
    cursor.execute("INSERT INTO products (product_name, price) VALUES ('Laptop', 799.99), ('Keyboard', 129.99), ('Mouse', 29.99)")
    cursor.execute("INSERT INTO staff (first_name, last_name) VALUES ('Alice', 'Smith'), ('Bob', 'Johnson'), ('Charlie', 'Williams')")
    cursor.execute("INSERT INTO orders (customer_name, staff_id, product_id) VALUES ('David Lee', 1, 1), ('Emily Chen', 2, 2), ('Frank Brown', 1, 3)")
    
    db_conn.commit()
