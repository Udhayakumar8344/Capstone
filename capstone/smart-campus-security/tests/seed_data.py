"""
Seed data generator for Smart Campus Security & Attendance 2.0
Creates sample data for testing and demonstration
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from db import Database
from utils.logger import setup_logger
from loguru import logger

setup_logger()


class DataSeeder:
    """Generate seed data for testing"""
    
    def __init__(self):
        self.db = Database()
    
    def generate_sample_embeddings(self, num_persons: int = 15) -> dict:
        """Generate sample face embeddings for persons"""
        np.random.seed(42)
        
        persons = {
            'students': [],
            'staff': [],
            'labour': []
        }
        
        # Generate students (5)
        for i in range(5):
            name = f"Student_{i+1}"
            embeddings = [np.random.randn(512) for _ in range(10)]
            person_id = self.db.register_role(name, 'student', embeddings)
            persons['students'].append(person_id)
            logger.info(f"Registered {name} (ID: {person_id})")
        
        # Generate staff (5)
        for i in range(5):
            name = f"Staff_{i+1}"
            embeddings = [np.random.randn(512) for _ in range(10)]
            person_id = self.db.register_role(name, 'staff', embeddings)
            persons['staff'].append(person_id)
            logger.info(f"Registered {name} (ID: {person_id})")
        
        # Generate labour (5)
        for i in range(5):
            name = f"Labour_{i+1}"
            embeddings = [np.random.randn(512) for _ in range(10)]
            person_id = self.db.register_role(name, 'labour', embeddings)
            persons['labour'].append(person_id)
            logger.info(f"Registered {name} (ID: {person_id})")
        
        return persons
    
    def generate_attendance_logs(self, persons: dict, days: int = 30):
        """Generate attendance logs for past N days"""
        logger.info(f"Generating {days} days of attendance logs...")
        
        start_date = datetime.now() - timedelta(days=days)
        
        for day in range(days):
            current_date = start_date + timedelta(days=day)
            
            # Skip weekends
            if current_date.weekday() >= 5:
                continue
            
            # Generate entries for each person
            for role_type, person_ids in persons.items():
                for person_id in person_ids:
                    # 90% attendance rate
                    if np.random.random() > 0.1:
                        # Random arrival time (8:00 - 10:00 AM)
                        hour = np.random.randint(8, 11)
                        minute = np.random.randint(0, 60)
                        
                        timestamp = current_date.replace(hour=hour, minute=minute)
                        
                        # Random camera
                        cam_id = np.random.choice([0, 1])
                        
                        # Random confidence (0.6 - 0.99)
                        confidence = np.random.uniform(0.6, 0.99)
                        
                        # Random gear
                        has_mask = np.random.random() > 0.7
                        has_helmet = (role_type == 'labour' and np.random.random() > 0.3)
                        
                        # Generate embedding
                        embedding = np.random.randn(512)
                        
                        # Create log entry
                        conn = self.db.get_connection()
                        cursor = conn.cursor()
                        
                        import pickle
                        embedding_blob = pickle.dumps(embedding)
                        
                        cursor.execute("""
                            INSERT INTO logs (person_id, role_type, timestamp, cam_id, 
                                            confidence, embedding, has_mask, has_helmet)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """, (person_id, role_type, timestamp, cam_id, confidence, 
                              embedding_blob, has_mask, has_helmet))
                        
                        conn.commit()
                        conn.close()
        
        logger.info(f"Generated attendance logs for {days} days")
    
    def generate_alerts(self, num_alerts: int = 20):
        """Generate sample alerts"""
        logger.info(f"Generating {num_alerts} sample alerts...")
        
        alert_types = ['UNKNOWN_PERSON', 'LATE_ENTRY', 'NO_HELMET', 'NO_MASK']
        
        for i in range(num_alerts):
            alert_type = np.random.choice(alert_types)
            cam_id = np.random.choice([0, 1])
            
            # Random timestamp in past week
            days_ago = np.random.randint(0, 7)
            hours_ago = np.random.randint(0, 24)
            timestamp = datetime.now() - timedelta(days=days_ago, hours=hours_ago)
            
            details = {
                'confidence': float(np.random.uniform(0.3, 0.6)),
                'reason': f'Test alert {i+1}'
            }
            
            conn = self.db.get_connection()
            cursor = conn.cursor()
            
            import json
            cursor.execute("""
                INSERT INTO alerts (alert_type, timestamp, cam_id, details)
                VALUES (?, ?, ?, ?)
            """, (alert_type, timestamp, cam_id, json.dumps(details)))
            
            conn.commit()
            conn.close()
        
        logger.info(f"Generated {num_alerts} alerts")
    
    def seed_all(self):
        """Run complete seeding process"""
        logger.info("Starting data seeding...")
        
        # Generate persons
        persons = self.generate_sample_embeddings(15)
        
        # Generate attendance logs
        self.generate_attendance_logs(persons, days=30)
        
        # Generate alerts
        self.generate_alerts(20)
        
        logger.info("Data seeding completed!")
        logger.info("\nSummary:")
        logger.info(f"  - Registered: 15 persons (5 students, 5 staff, 5 labour)")
        logger.info(f"  - Generated: ~30 days of attendance logs")
        logger.info(f"  - Created: 20 sample alerts")
        logger.info("\nYou can now run: streamlit run main.py")


if __name__ == "__main__":
    seeder = DataSeeder()
    seeder.seed_all()
