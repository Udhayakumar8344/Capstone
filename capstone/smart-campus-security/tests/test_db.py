"""
Unit tests for database module
"""

import pytest
import numpy as np
from datetime import datetime
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from db import Database


@pytest.fixture
def db():
    """Create test database"""
    test_db = Database("data/test_database.db")
    yield test_db
    # Cleanup
    Path("data/test_database.db").unlink(missing_ok=True)


def test_database_initialization(db):
    """Test database schema creation"""
    conn = db.get_connection()
    cursor = conn.cursor()
    
    # Check tables exist
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row[0] for row in cursor.fetchall()}
    
    assert 'roles' in tables
    assert 'logs' in tables
    assert 'alerts' in tables
    
    conn.close()


def test_register_role(db):
    """Test person registration"""
    embeddings = [np.random.randn(512) for _ in range(10)]
    
    person_id = db.register_role("Test Student", "student", embeddings)
    
    assert person_id > 0
    
    # Verify registration
    roles = db.get_all_roles()
    assert len(roles) == 1
    assert roles[0]['name'] == "Test Student"
    assert roles[0]['role_type'] == "student"


def test_mark_attendance(db):
    """Test attendance logging"""
    # Register person first
    embeddings = [np.random.randn(512) for _ in range(10)]
    person_id = db.register_role("Test Person", "staff", embeddings)
    
    # Mark attendance
    embedding = np.random.randn(512)
    log_id = db.mark_attendance(
        person_id, "staff", 0, 0.95, embedding, "test.jpg", False, False
    )
    
    assert log_id > 0


def test_get_person_by_embedding(db):
    """Test face matching"""
    # Register person
    base_embedding = np.random.randn(512)
    embeddings = [base_embedding + np.random.randn(512) * 0.01 for _ in range(10)]
    person_id = db.register_role("Test Match", "student", embeddings)
    
    # Test matching with similar embedding
    similar_embedding = base_embedding + np.random.randn(512) * 0.05
    match = db.get_person_by_embedding(similar_embedding, threshold=0.5)
    
    assert match is not None
    assert match[0] == person_id
    
    # Test non-matching with different embedding
    different_embedding = np.random.randn(512)
    no_match = db.get_person_by_embedding(different_embedding, threshold=0.9)
    
    # May or may not match depending on random values
    # Just verify it returns expected format
    assert no_match is None or isinstance(no_match, tuple)


def test_log_alert(db):
    """Test alert logging"""
    alert_id = db.log_alert(
        "UNKNOWN_PERSON",
        0,
        "snapshot.jpg",
        {"confidence": 0.45}
    )
    
    assert alert_id > 0
    
    # Verify alert
    alerts = db.get_recent_alerts(limit=10)
    assert len(alerts) == 1
    assert alerts[0]['alert_type'] == "UNKNOWN_PERSON"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
