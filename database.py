import sqlite3
import os
from datetime import datetime


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_DIR = os.path.join(BASE_DIR, "data")
DATABASE_PATH = os.path.join(DATABASE_DIR, "fall_detection.db")


def initialize_database():
    """Create database and fall_events table."""

    os.makedirs(DATABASE_DIR, exist_ok=True)

    connection = sqlite3.connect(DATABASE_PATH)

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS fall_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            person_id INTEGER,
            confidence REAL,
            posture TEXT,
            status TEXT,
            snapshot TEXT,
            acknowledged INTEGER DEFAULT 0
        )
    """)

    connection.commit()

    connection.close()

    print("Database initialized successfully.")


def add_fall_event(
    person_id,
    confidence,
    posture,
    status,
    snapshot
):
    """Store a confirmed fall event."""

    connection = sqlite3.connect(DATABASE_PATH)

    cursor = connection.cursor()

    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    cursor.execute("""
        INSERT INTO fall_events
        (
            timestamp,
            person_id,
            confidence,
            posture,
            status,
            snapshot,
            acknowledged
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        timestamp,
        person_id,
        confidence,
        posture,
        status,
        snapshot,
        0
    ))

    connection.commit()

    event_id = cursor.lastrowid

    connection.close()

    print(
        f"Fall event stored in database. "
        f"Event ID: {event_id}"
    )

    return event_id


def get_all_fall_events():
    """Return all fall events."""

    connection = sqlite3.connect(DATABASE_PATH)

    connection.row_factory = sqlite3.Row

    cursor = connection.cursor()

    cursor.execute("""
        SELECT *
        FROM fall_events
        ORDER BY id DESC
    """)

    events = cursor.fetchall()

    connection.close()

    return [dict(event) for event in events]


def get_statistics():
    """Return basic fall statistics."""

    connection = sqlite3.connect(DATABASE_PATH)

    cursor = connection.cursor()

    cursor.execute(
        "SELECT COUNT(*) FROM fall_events"
    )

    total_falls = cursor.fetchone()[0]

    cursor.execute("""
        SELECT COUNT(*)
        FROM fall_events
        WHERE date(timestamp) = date('now', 'localtime')
    """)

    today_falls = cursor.fetchone()[0]

    cursor.execute("""
        SELECT AVG(confidence)
        FROM fall_events
    """)

    average_confidence = cursor.fetchone()[0]

    if average_confidence is None:
        average_confidence = 0

    cursor.execute("""
        SELECT COUNT(*)
        FROM fall_events
        WHERE acknowledged = 0
    """)

    active_alerts = cursor.fetchone()[0]

    connection.close()

    return {
        "total_falls": total_falls,
        "today_falls": today_falls,
        "average_confidence": average_confidence,
        "active_alerts": active_alerts
    }


def acknowledge_event(event_id):
    """Mark a fall event as acknowledged."""

    connection = sqlite3.connect(DATABASE_PATH)

    cursor = connection.cursor()

    cursor.execute("""
        UPDATE fall_events
        SET acknowledged = 1
        WHERE id = ?
    """, (event_id,))

    connection.commit()

    updated = cursor.rowcount

    connection.close()

    return updated > 0


if __name__ == "__main__":

    initialize_database()

    print()
    print("Testing database...")
    print(get_statistics())