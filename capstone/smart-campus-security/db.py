"""
Database module for Smart Campus Security & Attendance 2.0
Handles SQLite operations for roles, logs, and alerts
"""

import sqlite3
import json
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pickle
from loguru import logger
import threading

class Database:
    """SQLite database manager for attendance and security system - Optimized for Concurrency"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(Database, cls).__new__(cls)
        return cls._instance

    def __init__(self, db_path: str = "data/database.db"):
        # Singleton initialization check
        if hasattr(self, 'initialized') and self.initialized:
            return
            
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Lock for write operations
        self.write_lock = threading.Lock()
        
        self.init_database()
        self.initialized = True
    
    def get_connection(self):
        """Create database connection with timeout"""
        # check_same_thread=False allows using connection across threads (carefully)
        # timeout=30.0 gives SQLite time to wait for locks to clear
        return sqlite3.connect(self.db_path, timeout=30.0, check_same_thread=False)
    
    def init_database(self):
        """Initialize database schema and set WAL mode"""
        try:
            with self.write_lock:
                conn = self.get_connection()
                # Set WAL mode once during initialization
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL") # Faster writes, safe enough for WAL
                
                cursor = conn.cursor()
                
                # Roles table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS roles (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        role_type TEXT NOT NULL CHECK(role_type IN ('student', 'staff', 'labour')),
                        embedding_path TEXT NOT NULL,
                        registered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        active BOOLEAN DEFAULT 1
                    )
                """)
                
                # Logs table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        person_id INTEGER,
                        role_type TEXT NOT NULL,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        cam_id INTEGER NOT NULL,
                        confidence REAL NOT NULL,
                        embedding BLOB,
                        image_path TEXT,
                        has_mask BOOLEAN DEFAULT 0,
                        has_helmet BOOLEAN DEFAULT 0,
                        FOREIGN KEY (person_id) REFERENCES roles(id)
                    )
                """)
                
                # Alerts table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS alerts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        alert_type TEXT NOT NULL,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        cam_id INTEGER NOT NULL,
                        image_path TEXT,
                        details TEXT,
                        acknowledged BOOLEAN DEFAULT 0
                    )
                """)
                
                # Create indexes
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON logs(timestamp)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_logs_person ON logs(person_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp)")
                
                conn.commit()
                conn.close()
                logger.info("Database initialized successfully with WAL mode")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
    
    def register_role(self, name: str, role_type: str, embeddings: List[np.ndarray]) -> int:
        """Register a new person"""
        avg_embedding = np.mean(embeddings, axis=0)
        
        embeddings_dir = Path("data/embeddings")
        embeddings_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        embedding_path = embeddings_dir / f"{role_type}_{name}_{timestamp}.pkl"
        
        with open(embedding_path, 'wb') as f:
            pickle.dump(avg_embedding, f)
            
        with self.write_lock:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO roles (name, role_type, embedding_path)
                    VALUES (?, ?, ?)
                """, (name, role_type, str(embedding_path)))
                person_id = cursor.lastrowid
                conn.commit()
        
        logger.info(f"Registered {role_type} '{name}' with ID {person_id}")
        return person_id
    
    def mark_attendance(self, person_id: Optional[int], role_type: str, cam_id: int,
                       confidence: float, embedding: np.ndarray, image_path: str,
                       has_mask: bool = False, has_helmet: bool = False) -> int:
        """Log attendance entry"""
        embedding_blob = pickle.dumps(embedding)
        
        with self.write_lock:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO logs (person_id, role_type, cam_id, confidence, embedding, 
                                    image_path, has_mask, has_helmet)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (person_id, role_type, cam_id, confidence, embedding_blob, 
                      image_path, has_mask, has_helmet))
                log_id = cursor.lastrowid
                conn.commit()
        
        return log_id
    
    def get_person_by_embedding(self, embedding: np.ndarray, 
                               threshold: float = 0.6) -> Optional[Tuple[int, str, float]]:
        """Find person by embedding"""
        # Read operations don't strictly need the write lock in WAL mode, 
        # but creating a fresh connection is good practice.
        conn = self.get_connection()
        conn.row_factory = sqlite3.Row
        
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT id, role_type, embedding_path FROM roles WHERE active = 1")
            roles = cursor.fetchall()
        finally:
            conn.close()
        
        best_match = None
        best_similarity = threshold
        
        for role in roles:
            try:
                with open(role['embedding_path'], 'rb') as f:
                    stored_embedding = pickle.load(f)
                
                similarity = self._cosine_similarity(embedding, stored_embedding)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = (role['id'], role['role_type'], similarity)
            except Exception:
                continue
        
        return best_match
    
    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6)
    
    def log_alert(self, alert_type: str, cam_id: int, image_path: str, 
                  details: Dict) -> int:
        """Log security alert"""
        with self.write_lock:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO alerts (alert_type, cam_id, image_path, details)
                    VALUES (?, ?, ?, ?)
                """, (alert_type, cam_id, image_path, json.dumps(details)))
                alert_id = cursor.lastrowid
                conn.commit()
        
        logger.warning(f"Alert logged: {alert_type} at camera {cam_id}")
        return alert_id
    
    def get_analytics(self, start_date: datetime, end_date: datetime, 
                     role_type: Optional[str] = None) -> Dict:
        """Get analytics data"""
        conn = self.get_connection()
        try:
            query = """
                SELECT 
                    DATE(timestamp) as date,
                    role_type,
                    COUNT(DISTINCT person_id) as unique_persons,
                    COUNT(*) as total_entries,
                    AVG(confidence) as avg_confidence,
                    SUM(CASE WHEN has_mask THEN 1 ELSE 0 END) as mask_count,
                    SUM(CASE WHEN has_helmet THEN 1 ELSE 0 END) as helmet_count
                FROM logs
                WHERE timestamp BETWEEN ? AND ?
            """
            
            params = [start_date, end_date]
            
            if role_type:
                query += " AND role_type = ?"
                params.append(role_type)
            
            query += " GROUP BY DATE(timestamp), role_type ORDER BY date DESC"
            
            df = pd.read_sql_query(query, conn, params=params)
            return df.to_dict('records')
        finally:
            conn.close()

    def get_daily_summary(self, date: datetime) -> List[Dict]:
        """
        Get daily summary with Entry and Exit times
        
        Args:
            date: Date to summarize
        
        Returns:
            List of dicts with name, role, entry_time, exit_time
        """
        conn = self.get_connection()
        try:
            query = """
                SELECT 
                    r.name,
                    r.role_type,
                    MIN(l.timestamp) as entry_time,
                    MAX(l.timestamp) as exit_time,
                    COUNT(*) as detections
                FROM logs l
                JOIN roles r ON l.person_id = r.id
                WHERE DATE(l.timestamp) = ?
                GROUP BY l.person_id
                ORDER BY entry_time ASC
            """
            
            df = pd.read_sql_query(query, conn, params=[date.date()])
            
            # Calculate duration
            if not df.empty:
                df['entry_time'] = pd.to_datetime(df['entry_time'])
                df['exit_time'] = pd.to_datetime(df['exit_time'])
                df['duration'] = df['exit_time'] - df['entry_time']
                
                # Format for display
                df['entry_str'] = df['entry_time'].dt.strftime("%H:%M:%S")
                df['exit_str'] = df['exit_time'].dt.strftime("%H:%M:%S")
                df['duration_str'] = df['duration'].apply(lambda x: str(x).split('.')[0])
                
                return df.to_dict('records')
            return []
        finally:
            conn.close()
    
    def get_late_arrivals(self, date: datetime, threshold_minutes: int = 30) -> List[Dict]:
        """Get late arrivals"""
        conn = self.get_connection()
        try:
            query = """
                WITH person_avg AS (
                    SELECT 
                        person_id,
                        AVG(CAST(strftime('%H', timestamp) AS INTEGER) * 60 + 
                            CAST(strftime('%M', timestamp) AS INTEGER)) as avg_minutes
                    FROM logs
                    WHERE person_id IS NOT NULL
                      AND DATE(timestamp) < ?
                    GROUP BY person_id
                )
                SELECT 
                    l.person_id,
                    r.name,
                    r.role_type,
                    l.timestamp,
                    CAST(strftime('%H', l.timestamp) AS INTEGER) * 60 + 
                        CAST(strftime('%M', l.timestamp) AS INTEGER) as arrival_minutes,
                    pa.avg_minutes
                FROM logs l
                JOIN roles r ON l.person_id = r.id
                LEFT JOIN person_avg pa ON l.person_id = pa.person_id
                WHERE DATE(l.timestamp) = ?
                  AND l.person_id IS NOT NULL
                  AND (arrival_minutes - pa.avg_minutes) > ?
                ORDER BY arrival_minutes DESC
            """
            df = pd.read_sql_query(query, conn, params=[date, date, threshold_minutes])
            return df.to_dict('records')
        finally:
            conn.close()
    
    def get_absentees(self, date: datetime, expected_ids: List[int]) -> List[Dict]:
        """Get absentees"""
        if not expected_ids:
            return []
            
        conn = self.get_connection()
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT DISTINCT person_id
                FROM logs
                WHERE DATE(timestamp) = ?
                  AND person_id IN ({})
            """.format(','.join('?' * len(expected_ids))), [date] + expected_ids)
            
            attended_ids = {row[0] for row in cursor.fetchall()}
            absentee_ids = set(expected_ids) - attended_ids
            
            if not absentee_ids:
                return []
            
            cursor.execute("""
                SELECT id, name, role_type
                FROM roles
                WHERE id IN ({})
            """.format(','.join('?' * len(absentee_ids))), list(absentee_ids))
            
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()
    
    def get_recent_alerts(self, limit: int = 50) -> List[Dict]:
        """Get recent alerts"""
        conn = self.get_connection()
        try:
            df = pd.read_sql_query("""
                SELECT * FROM alerts
                ORDER BY timestamp DESC
                LIMIT ?
            """, conn, params=[limit])
            return df.to_dict('records')
        finally:
            conn.close()
    
    def get_all_roles(self) -> List[Dict]:
        """Get all registered roles"""
        conn = self.get_connection()
        try:
            df = pd.read_sql_query("""
                SELECT id, name, role_type, registered_at
                FROM roles
                WHERE active = 1
                ORDER BY role_type, name
            """, conn)
            return df.to_dict('records')
        finally:
            conn.close()


# Import pandas here to avoid circular dependency
import pandas as pd


if __name__ == "__main__":
    # Test database
    db = Database()
    logger.info("Database test successful")
