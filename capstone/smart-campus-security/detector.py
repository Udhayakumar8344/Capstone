"""
Core detection module for Smart Campus Security & Attendance 2.0
Handles face detection, recognition, role classification, and gear detection
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import yaml
from loguru import logger
from datetime import datetime

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    logger.warning("DeepFace not available")
    DEEPFACE_AVAILABLE = False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    logger.warning("YOLO not available")
    YOLO_AVAILABLE = False


class FaceDetector:
    """Multi-face detection and recognition with gear detection"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize detector with models"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.recognition_config = self.config['recognition']
        self.model_config = self.config['models']
        self.role_colors = self.config['role_colors']
        
        # Initialize DeepFace
        if DEEPFACE_AVAILABLE:
            logger.info("DeepFace initialized successfully (model will load on first run)")
        
        # Fallback to Haar Cascade if DeepFace/InsightFace not available
        self.haar_cascade = None
        if not DEEPFACE_AVAILABLE:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.haar_cascade = cv2.CascadeClassifier(cascade_path)
            logger.info("Using Haar Cascade as fallback for face detection")
    
    def detect_faces(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect all faces in frame
        
        Args:
            frame: Input frame (BGR)
        
        Returns:
            List of face detections with bboxes and embeddings
        """
        detections = []
        
        if DEEPFACE_AVAILABLE:
            # Use DeepFace for detection and recognition
            try:
                # Use DeepFace's represent method which handles detection + embedding
                # We use 'opencv' as detector because it's fast, or 'retinaface' for better accuracy
                objs = DeepFace.represent(
                    img_path=frame,
                    model_name="Facenet", # 512-dim matches config
                    detector_backend="opencv",
                    enforce_detection=False,
                    align=True
                )
                
                for obj in objs[:self.recognition_config['max_faces_per_frame']]:
                    area = obj["facial_area"]
                    bbox = [area["x"], area["y"], area["x"] + area["w"], area["y"] + area["h"]]
                    
                    detection = {
                        'bbox': bbox,
                        'embedding': np.array(obj["embedding"]),
                        'confidence': 0.9, # DeepFace doesn't expose raw det score easily
                        'landmarks': None
                    }
                    detections.append(detection)
            except Exception as e:
                logger.error(f"DeepFace representation failed: {e}")
        
        elif self.haar_cascade:
            # Fallback to Haar Cascade
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.haar_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            
            for (x, y, w, h) in faces[:self.recognition_config['max_faces_per_frame']]:
                # Extract face for embedding (simplified)
                face_crop = frame[y:y+h, x:x+w]
                embedding = self._simple_embedding(face_crop)
                
                detection = {
                    'bbox': [x, y, x+w, y+h],
                    'embedding': embedding,
                    'confidence': 0.8,  # Haar doesn't provide confidence
                    'landmarks': None
                }
                detections.append(detection)
        
        return detections
    
    @staticmethod
    def _simple_embedding(face_crop: np.ndarray, dim: int = 512) -> np.ndarray:
        """
        Generate simple embedding from face crop (fallback)
        Uses histogram of oriented gradients (HOG) features
        """
        # Resize to fixed size
        face_resized = cv2.resize(face_crop, (128, 128))
        
        # Convert to grayscale
        gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
        
        # Compute HOG features
        win_size = (128, 128)
        block_size = (16, 16)
        block_stride = (8, 8)
        cell_size = (8, 8)
        nbins = 9
        
        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
        features = hog.compute(gray)
        
        # Pad or truncate to desired dimension
        if len(features) < dim:
            embedding = np.pad(features.flatten(), (0, dim - len(features)))
        else:
            embedding = features.flatten()[:dim]
        
        # Normalize
        embedding = embedding / (np.linalg.norm(embedding) + 1e-6)
        
        return embedding
    
    def detect_gear(self, frame: np.ndarray, bbox: List[int]) -> Tuple[bool, bool]:
        """
        Detect mask and helmet in face region
        
        Args:
            frame: Input frame
            bbox: Face bounding box [x1, y1, x2, y2]
        
        Returns:
            (has_mask, has_helmet)
        """
        if not self.yolo_model:
            return False, False
        
        # Expand bbox for better detection
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        
        # Expand by 20%
        margin_x = int((x2 - x1) * 0.2)
        margin_y = int((y2 - y1) * 0.2)
        
        x1 = max(0, x1 - margin_x)
        y1 = max(0, y1 - margin_y)
        x2 = min(w, x2 + margin_x)
        y2 = min(h, y2 + margin_y)
        
        # Crop region
        roi = frame[y1:y2, x1:x2]
        
        if roi.size == 0:
            return False, False
        
        # Run YOLO detection
        try:
            results = self.yolo_model(roi, verbose=False)
            
            has_mask = False
            has_helmet = False
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    if conf > self.model_config['yolo']['confidence_threshold']:
                        # Assuming class 0=mask, 1=helmet (adjust based on your model)
                        if cls == 0:
                            has_mask = True
                        elif cls == 1:
                            has_helmet = True
            
            return has_mask, has_helmet
        
        except Exception as e:
            logger.error(f"Gear detection error: {e}")
            return False, False
    
    def get_role_color(self, role_type: str) -> Tuple[int, int, int]:
        """Get BGR color for role type"""
        return tuple(self.role_colors.get(role_type, self.role_colors['unknown']))
    
    def draw_detection(self, frame: np.ndarray, bbox: List[int], 
                      role_type: str, confidence: float, 
                      name: Optional[str] = None,
                      has_mask: bool = False, has_helmet: bool = False) -> np.ndarray:
        """
        Draw detection on frame
        
        Args:
            frame: Input frame
            bbox: Bounding box [x1, y1, x2, y2]
            role_type: Role type
            confidence: Recognition confidence
            name: Person name (optional)
            has_mask: Whether wearing mask
            has_helmet: Whether wearing helmet
        
        Returns:
            Annotated frame
        """
        x1, y1, x2, y2 = bbox
        color = self.get_role_color(role_type)
        
        # Draw bounding box
        thickness = 3 if role_type == 'unknown' else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Prepare label
        label_parts = []
        if name:
            label_parts.append(name)
        label_parts.append(f"{role_type.upper()}")
        label_parts.append(f"{confidence:.2f}")
        
        label = " | ".join(label_parts)
        
        # Draw label background
        (label_w, label_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        cv2.rectangle(frame, (x1, y1 - label_h - 10), 
                     (x1 + label_w, y1), color, -1)
        
        # Draw label text
        cv2.putText(frame, label, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw gear icons
        icon_y = y2 + 20
        if has_mask:
            cv2.putText(frame, "MASK", (x1, icon_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            icon_y += 20
        
        if has_helmet:
            cv2.putText(frame, "HELMET", (x1, icon_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame
    
    def process_frame(self, frame: np.ndarray, cam_id: int, 
                     db, tracker, alert_system) -> Tuple[np.ndarray, List[Dict]]:
        """
        Main processing pipeline for a frame
        
        Args:
            frame: Input frame
            cam_id: Camera ID
            db: Database instance
            tracker: MultiCamTracker instance
            alert_system: AlertSystem instance
        
        Returns:
            (annotated_frame, detections_list)
        """
        if not hasattr(self, 'last_logged'):
            self.last_logged = {}  # track_id -> timestamp
            
        detections_list = []
        
        # Detect faces
        faces = self.detect_faces(frame)
        
        for face in faces:
            bbox = face['bbox']
            embedding = face['embedding']
            det_confidence = face['confidence']
            
            # FIRST: Try to match with existing track (maintain identity)
            matched_track_id = tracker.match_track(embedding, cam_id)
            
            # Initialize variables
            person_id = None
            role_type = 'unknown'
            confidence = det_confidence
            person_name = "UNKNOWN"
            
            if matched_track_id is not None:
                # Use existing track's identity (prevents flickering)
                track = tracker.get_track(matched_track_id)
                if track and track.person_id is not None:
                    # Track already has identity - use it (maintains consistency)
                    person_id = track.person_id
                    role_type = track.role_type
                    confidence = max(track.confidence, det_confidence)
                # If track exists but no identity, we'll query DB below
            
            # Only query database if we don't have identity from track
            if person_id is None:
                match = db.get_person_by_embedding(
                    embedding,
                    threshold=self.recognition_config['confidence_threshold']
                )
                
                if match:
                    person_id, role_type, confidence = match
            
            # Update tracker (will use existing track if matched, or create new)
            track_id = tracker.update(
                embedding, cam_id, person_id, role_type, confidence
            )
            
            # Get final identity from tracker (ensures we use tracker's maintained identity)
            final_track = tracker.get_track(track_id)
            if final_track:
                # Always use tracker's identity (it maintains consistency)
                person_id = final_track.person_id
                role_type = final_track.role_type
                confidence = final_track.confidence
            
            # Get person name based on final identity
            if person_id is not None:
                roles = db.get_all_roles()
                person_name = next(
                    (r['name'] for r in roles if r['id'] == person_id),
                    f"ID_{person_id}"
                )
            else:
                person_name = "UNKNOWN"
            
            # Conditional Gear Detection (Optimization)
            has_mask = False
            has_helmet = False
            
            # Check if we need to run YOLO (expensive)
            # Only run if: 
            # 1. Role is labour (need helmet check)
            # 2. OR Mask alert is enabled
            need_gear_check = (
                (role_type == 'labour' and self.config['alerts']['no_helmet']['enabled']) or
                self.config['alerts']['no_mask']['enabled']
            )
            
            if need_gear_check:
                has_mask, has_helmet = self.detect_gear(frame, bbox)
            
            # Throttle Logging & Snapshots (Optimization)
            # Only log every 2 seconds per person to save Disk/DB I/O
            current_time = datetime.now()
            should_log = False
            
            if track_id not in self.last_logged:
                should_log = True
            else:
                elapsed = (current_time - self.last_logged[track_id]).total_seconds()
                if elapsed > 2.0:  # Log every 2 seconds
                    should_log = True
            
            log_id = None
            if should_log:
                self.last_logged[track_id] = current_time
                
                # Alerts
                if role_type == 'unknown':
                     if confidence < self.recognition_config['confidence_threshold']:
                        alert_system.trigger_alert(
                            'UNKNOWN_PERSON',
                            cam_id,
                            frame,
                            {'confidence': confidence, 'bbox': bbox}
                        )
                
                if role_type == 'labour' and not has_helmet:
                    alert_system.trigger_alert(
                        'NO_HELMET',
                        cam_id,
                        frame,
                        {'person_id': person_id, 'name': person_name}
                    )
                
                # Save snapshot
                timestamp_str = current_time.strftime("%Y%m%d_%H%M%S")
                snapshot_dir = Path("data/snapshots/detections")
                snapshot_dir.mkdir(parents=True, exist_ok=True)
                snapshot_path = snapshot_dir / f"cam{cam_id}_{timestamp_str}_{track_id}.jpg"
                cv2.imwrite(str(snapshot_path), frame)
                
                # Log attendance
                log_id = db.mark_attendance(
                    person_id, role_type, cam_id, confidence,
                    embedding, str(snapshot_path), has_mask, has_helmet
                )
            
            # Draw detection
            frame = self.draw_detection(
                frame, bbox, role_type, confidence,
                person_name, has_mask, has_helmet
            )
            
            # Add to detections list (always return for UI display)
            detections_list.append({
                'track_id': track_id,
                'person_id': person_id,
                'name': person_name,
                'role_type': role_type,
                'confidence': confidence,
                'bbox': bbox,
                'has_mask': has_mask,
                'has_helmet': has_helmet,
                'log_id': log_id
            })
            
            # Cleanup old track logs
            if len(self.last_logged) > 100:
                # Keep only recent tracks
                self.last_logged = {k:v for k,v in self.last_logged.items() 
                                   if (current_time - v).total_seconds() < 60}
        
        return frame, detections_list


if __name__ == "__main__":
    # Test detector
    from utils.logger import setup_logger
    setup_logger()
    
    detector = FaceDetector()
    
    # Test with webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        logger.error("Cannot open camera")
        exit()
    
    logger.info("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect faces
        faces = detector.detect_faces(frame)
        
        # Draw detections
        for face in faces:
            bbox = face['bbox']
            conf = face['confidence']
            
            frame = detector.draw_detection(
                frame, bbox, 'unknown', conf
            )
        
        cv2.imshow('Face Detection Test', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    logger.info("Detector test completed")
