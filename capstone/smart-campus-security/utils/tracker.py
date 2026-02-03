"""
Person tracking and re-identification across multiple cameras
Uses embedding similarity for cross-camera tracking
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import threading
from loguru import logger


@dataclass
class PersonTrack:
    """Represents a tracked person across cameras"""
    track_id: int
    person_id: Optional[int]  # Database person ID (None if unknown)
    role_type: str
    embeddings: List[np.ndarray] = field(default_factory=list)
    camera_history: List[Tuple[int, datetime]] = field(default_factory=list)
    last_seen: datetime = field(default_factory=datetime.now)
    confidence: float = 0.0
    
    def update(self, embedding: np.ndarray, cam_id: int, confidence: float):
        """Update track with new detection"""
        self.embeddings.append(embedding)
        self.camera_history.append((cam_id, datetime.now()))
        self.last_seen = datetime.now()
        self.confidence = max(self.confidence, confidence)
        
        # Keep only recent embeddings (last 10)
        if len(self.embeddings) > 10:
            self.embeddings = self.embeddings[-10:]
    
    def get_average_embedding(self) -> np.ndarray:
        """Get average embedding for matching"""
        return np.mean(self.embeddings, axis=0)


class MultiCamTracker:
    """Multi-camera person tracker with re-identification"""
    
    def __init__(self, similarity_threshold: float = 0.7, 
                 track_timeout: int = 300, max_tracks: int = 100):
        """
        Initialize tracker
        
        Args:
            similarity_threshold: Minimum similarity to match tracks
            track_timeout: Seconds before track is considered stale
            max_tracks: Maximum number of active tracks
        """
        self.similarity_threshold = similarity_threshold
        self.track_timeout = track_timeout
        self.max_tracks = max_tracks
        
        self.tracks: Dict[int, PersonTrack] = {}
        self.next_track_id = 1
        self.lock = threading.Lock()
        
        logger.info(f"MultiCamTracker initialized (threshold={similarity_threshold})")
    
    def update(self, embedding: np.ndarray, cam_id: int, person_id: Optional[int],
               role_type: str, confidence: float) -> int:
        """
        Update tracker with new detection
        
        Args:
            embedding: Face embedding
            cam_id: Camera ID
            person_id: Database person ID (None if unknown)
            role_type: Role type
            confidence: Recognition confidence
        
        Returns:
            track_id: ID of matched or new track
        """
        with self.lock:
            # Try to match existing track
            matched_track_id = self.match_track(embedding, cam_id)
            
            if matched_track_id is not None:
                # Update existing track
                track = self.tracks[matched_track_id]
                track.update(embedding, cam_id, confidence)
                
                # Maintain identity: Only update person_id if:
                # 1. Track was unknown and we have a match (confidence > threshold)
                # 2. OR we have a very high confidence match (>0.9) to a different person
                if track.person_id is None and person_id is not None:
                    # Unknown track - assign identity
                    track.person_id = person_id
                    track.role_type = role_type
                elif track.person_id is not None and person_id is not None:
                    # Track has identity - only change if very high confidence and different person
                    if confidence > 0.9 and person_id != track.person_id:
                        # High confidence different match - update identity
                        track.person_id = person_id
                        track.role_type = role_type
                    # Otherwise, keep existing identity (prevents flickering)
                
                logger.debug(f"Updated track {matched_track_id} at camera {cam_id}")
                return matched_track_id
            else:
                # Create new track
                track_id = self._create_track(embedding, cam_id, person_id, 
                                             role_type, confidence)
                logger.info(f"Created new track {track_id} for {role_type} at camera {cam_id}")
                return track_id
    
    def match_track(self, embedding: np.ndarray, cam_id: int) -> Optional[int]:
        """
        Find matching track for embedding
        
        Args:
            embedding: Query embedding
            cam_id: Camera ID
        
        Returns:
            track_id or None if no match
        """
        best_match_id = None
        best_similarity = self.similarity_threshold
        
        for track_id, track in self.tracks.items():
            # Skip stale tracks
            if self._is_stale(track):
                continue
            
            # Compute similarity with track's average embedding
            avg_embedding = track.get_average_embedding()
            similarity = self._cosine_similarity(embedding, avg_embedding)
            
            # Boost similarity if same camera (likely same person)
            if track.camera_history and track.camera_history[-1][0] == cam_id:
                similarity *= 1.1
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_id = track_id
        
        return best_match_id
    
    def _create_track(self, embedding: np.ndarray, cam_id: int, 
                     person_id: Optional[int], role_type: str, 
                     confidence: float) -> int:
        """Create new track"""
        track_id = self.next_track_id
        self.next_track_id += 1
        
        track = PersonTrack(
            track_id=track_id,
            person_id=person_id,
            role_type=role_type,
            confidence=confidence
        )
        track.update(embedding, cam_id, confidence)
        
        self.tracks[track_id] = track
        
        # Cleanup if too many tracks
        if len(self.tracks) > self.max_tracks:
            self.cleanup_stale_tracks(force=True)
        
        return track_id
    
    def cleanup_stale_tracks(self, force: bool = False):
        """
        Remove stale tracks
        
        Args:
            force: If True, remove oldest tracks even if not stale
        """
        with self.lock:
            stale_ids = []
            
            for track_id, track in self.tracks.items():
                if self._is_stale(track):
                    stale_ids.append(track_id)
            
            # If forced and still too many, remove oldest
            if force and len(self.tracks) - len(stale_ids) > self.max_tracks:
                sorted_tracks = sorted(
                    self.tracks.items(),
                    key=lambda x: x[1].last_seen
                )
                num_to_remove = len(self.tracks) - self.max_tracks
                for track_id, _ in sorted_tracks[:num_to_remove]:
                    if track_id not in stale_ids:
                        stale_ids.append(track_id)
            
            # Remove stale tracks
            for track_id in stale_ids:
                del self.tracks[track_id]
            
            if stale_ids:
                logger.info(f"Cleaned up {len(stale_ids)} stale tracks")
    
    def _is_stale(self, track: PersonTrack) -> bool:
        """Check if track is stale"""
        age = (datetime.now() - track.last_seen).total_seconds()
        return age > self.track_timeout
    
    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def get_track(self, track_id: int) -> Optional[PersonTrack]:
        """Get track by ID"""
        with self.lock:
            return self.tracks.get(track_id)
    
    def get_active_tracks(self) -> List[PersonTrack]:
        """Get all active (non-stale) tracks"""
        with self.lock:
            return [
                track for track in self.tracks.values()
                if not self._is_stale(track)
            ]
    
    def get_stats(self) -> Dict:
        """Get tracker statistics"""
        with self.lock:
            active_tracks = [t for t in self.tracks.values() if not self._is_stale(t)]
            
            return {
                'total_tracks': len(self.tracks),
                'active_tracks': len(active_tracks),
                'next_track_id': self.next_track_id,
                'role_distribution': self._get_role_distribution(active_tracks)
            }
    
    @staticmethod
    def _get_role_distribution(tracks: List[PersonTrack]) -> Dict[str, int]:
        """Get distribution of roles in active tracks"""
        distribution = {}
        for track in tracks:
            role = track.role_type
            distribution[role] = distribution.get(role, 0) + 1
        return distribution


if __name__ == "__main__":
    # Test tracker
    tracker = MultiCamTracker()
    
    # Simulate detections
    embedding1 = np.random.randn(512)
    embedding2 = embedding1 + np.random.randn(512) * 0.1  # Similar
    embedding3 = np.random.randn(512)  # Different
    
    # First detection
    track1 = tracker.update(embedding1, cam_id=0, person_id=1, 
                           role_type='student', confidence=0.95)
    print(f"Track 1: {track1}")
    
    # Similar detection (should match)
    track2 = tracker.update(embedding2, cam_id=1, person_id=1, 
                           role_type='student', confidence=0.92)
    print(f"Track 2: {track2} (should match track 1)")
    
    # Different detection (should create new track)
    track3 = tracker.update(embedding3, cam_id=0, person_id=2, 
                           role_type='staff', confidence=0.88)
    print(f"Track 3: {track3} (should be new)")
    
    # Print stats
    print(f"Tracker stats: {tracker.get_stats()}")
    
    logger.info("Tracker test completed")
