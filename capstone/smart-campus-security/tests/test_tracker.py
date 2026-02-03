"""
Unit tests for tracker module
"""

import pytest
import numpy as np
from datetime import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.tracker import MultiCamTracker, PersonTrack


@pytest.fixture
def tracker():
    """Create tracker instance"""
    return MultiCamTracker(similarity_threshold=0.7)


def test_tracker_initialization(tracker):
    """Test tracker initialization"""
    assert tracker.similarity_threshold == 0.7
    assert len(tracker.tracks) == 0
    assert tracker.next_track_id == 1


def test_create_track(tracker):
    """Test track creation"""
    embedding = np.random.randn(512)
    
    track_id = tracker.update(embedding, 0, 1, 'student', 0.95)
    
    assert track_id == 1
    assert len(tracker.tracks) == 1
    assert tracker.tracks[1].person_id == 1
    assert tracker.tracks[1].role_type == 'student'


def test_match_track(tracker):
    """Test track matching"""
    # Create initial track
    base_embedding = np.random.randn(512)
    track_id1 = tracker.update(base_embedding, 0, 1, 'student', 0.95)
    
    # Update with similar embedding (should match)
    similar_embedding = base_embedding + np.random.randn(512) * 0.1
    track_id2 = tracker.update(similar_embedding, 1, 1, 'student', 0.92)
    
    # Should be same track
    assert track_id1 == track_id2
    
    # Create different track
    different_embedding = np.random.randn(512)
    track_id3 = tracker.update(different_embedding, 0, 2, 'staff', 0.88)
    
    # Should be different track
    assert track_id3 != track_id1


def test_tracker_stats(tracker):
    """Test tracker statistics"""
    # Create multiple tracks
    for i in range(5):
        embedding = np.random.randn(512)
        tracker.update(embedding, 0, i, 'student', 0.9)
    
    stats = tracker.get_stats()
    
    assert stats['total_tracks'] == 5
    assert stats['active_tracks'] == 5
    assert 'student' in stats['role_distribution']


def test_cleanup_stale_tracks(tracker):
    """Test stale track cleanup"""
    # Create track
    embedding = np.random.randn(512)
    track_id = tracker.update(embedding, 0, 1, 'student', 0.95)
    
    # Manually set old timestamp
    tracker.tracks[track_id].last_seen = datetime(2020, 1, 1)
    
    # Cleanup
    tracker.cleanup_stale_tracks()
    
    # Track should be removed
    assert len(tracker.tracks) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
