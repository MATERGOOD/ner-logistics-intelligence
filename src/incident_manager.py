import sqlite3
import pandas as pd
import os
import datetime

DB_PATH = "data/incidents.db"

def init_db():
    os.makedirs("data", exist_ok=True)
    os.makedirs("data/uploads", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS incident_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            reporter_name TEXT,
            official_role TEXT,
            incident_type TEXT,
            severity TEXT,
            nearest_segment_id TEXT,
            latitude REAL,
            longitude REAL,
            description TEXT,
            photo_path TEXT,
            status TEXT,
            verified_by TEXT,
            verification_confidence REAL
        )
    ''')
    # Try adding columns to existing DB if it already exists (ignores error if already there)
    try:
        c.execute('ALTER TABLE incident_reports ADD COLUMN verified_by TEXT')
        c.execute('ALTER TABLE incident_reports ADD COLUMN verification_confidence REAL')
    except:
        pass
    conn.commit()
    conn.close()

def submit_report(reporter_name, official_role, incident_type, severity, segment_id, lat, lon, description, photo_path):
    if not os.path.exists(DB_PATH):
        init_db()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    tz = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute('''
        INSERT INTO incident_reports 
        (timestamp, reporter_name, official_role, incident_type, severity, nearest_segment_id, latitude, longitude, description, photo_path, status, verified_by, verification_confidence)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (tz, reporter_name, official_role, incident_type, severity, segment_id, lat, lon, description, photo_path, "Reported - Pending Verification", None, None))
    conn.commit()
    conn.close()

def get_active_incidents():
    if not os.path.exists(DB_PATH):
        init_db()
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query("SELECT * FROM incident_reports WHERE status IN ('Reported - Pending Verification', 'Verified & Confirmed')", conn)
    except:
        df = pd.DataFrame()
    conn.close()
    return df

def resolve_incident(incident_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE incident_reports SET status='Resolved / Cleared' WHERE id=?", (incident_id,))
    conn.commit()
    conn.close()

def verify_incident(incident_id, authority_name="Ri-Bhoi District Control Room", confidence=0.92):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE incident_reports SET status='Verified & Confirmed', verified_by=?, verification_confidence=? WHERE id=?", (authority_name, confidence, incident_id))
    conn.commit()
    conn.close()

def dismiss_incident(incident_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE incident_reports SET status='Dismissed / False Alarm' WHERE id=?", (incident_id,))
    conn.commit()
    conn.close()
