"""
Alert system for Smart Campus Security & Attendance 2.0
Handles sound, email, and snapshot alerts
"""

import pygame
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from pathlib import Path
from datetime import datetime
import cv2
import yaml
from loguru import logger
from typing import Dict, Optional
import threading


class AlertSystem:
    """Multi-channel alert system"""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.alert_config = self.config['alerts']
        self.email_config = self.config['email']
        self.storage_config = self.config['storage']
        
        # Initialize pygame for sound
        pygame.mixer.init()
        self.alert_sound = None
        
        # Load alert sound
        sound_path = Path("assets/alert.wav")
        if sound_path.exists():
            self.alert_sound = pygame.mixer.Sound(str(sound_path))
        else:
            logger.warning("Alert sound file not found")
        
        # Create snapshots directory
        Path(self.storage_config['snapshots_path']).mkdir(parents=True, exist_ok=True)
    
    def trigger_alert(self, alert_type: str, cam_id: int, frame, 
                     details: Optional[Dict] = None):
        """
        Main alert dispatcher
        
        Args:
            alert_type: Type of alert (UNKNOWN_PERSON, LATE_ENTRY, etc.)
            cam_id: Camera ID
            frame: Video frame (numpy array)
            details: Additional details dictionary
        """
        if details is None:
            details = {}
        
        alert_settings = self.alert_config.get(alert_type.lower(), {})
        
        if not alert_settings.get('enabled', False):
            return
        
        logger.warning(f"Alert triggered: {alert_type} at camera {cam_id}")
        
        # Save snapshot
        snapshot_path = None
        if alert_settings.get('snapshot', False):
            snapshot_path = self.save_snapshot(frame, alert_type, cam_id)
            details['snapshot_path'] = snapshot_path
        
        # Play sound (non-blocking)
        if alert_settings.get('sound', False):
            threading.Thread(target=self.play_sound, daemon=True).start()
        
        # Send email (non-blocking)
        if alert_settings.get('email', False) and self.email_config['enabled']:
            threading.Thread(
                target=self.send_email,
                args=(alert_type, cam_id, details, snapshot_path),
                daemon=True
            ).start()
    
    def play_sound(self):
        """Play alert sound (non-blocking)"""
        try:
            if self.alert_sound:
                self.alert_sound.play()
                logger.info("Alert sound played")
        except Exception as e:
            logger.error(f"Error playing sound: {e}")
    
    def save_snapshot(self, frame, alert_type: str, cam_id: int) -> str:
        """
        Save frame snapshot
        
        Args:
            frame: Video frame
            alert_type: Type of alert
            cam_id: Camera ID
        
        Returns:
            Path to saved snapshot
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{alert_type}_{cam_id}_{timestamp}.jpg"
            snapshot_path = Path(self.storage_config['snapshots_path']) / filename
            
            cv2.imwrite(str(snapshot_path), frame)
            logger.info(f"Snapshot saved: {snapshot_path}")
            
            return str(snapshot_path)
        except Exception as e:
            logger.error(f"Error saving snapshot: {e}")
            return ""
    
    def send_email(self, alert_type: str, cam_id: int, details: Dict, 
                   snapshot_path: Optional[str] = None):
        """
        Send email alert with optional snapshot attachment
        
        Args:
            alert_type: Type of alert
            cam_id: Camera ID
            details: Alert details
            snapshot_path: Path to snapshot image
        """
        try:
            # Get camera name
            cam_name = next(
                (c['name'] for c in self.config['cameras'] if c['id'] == cam_id),
                f"Camera {cam_id}"
            )
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.email_config['sender_email']
            msg['To'] = ', '.join(self.email_config['recipient_emails'])
            msg['Subject'] = f"Security Alert: {alert_type} - {cam_name}"
            
            # Email body
            body = f"""
Security Alert Notification

Alert Type: {alert_type}
Camera: {cam_name} (ID: {cam_id})
Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Details:
{self._format_details(details)}

This is an automated alert from Smart Campus Security System.
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Attach snapshot if available
            if snapshot_path and Path(snapshot_path).exists():
                with open(snapshot_path, 'rb') as f:
                    img = MIMEImage(f.read())
                    img.add_header('Content-Disposition', 'attachment', 
                                 filename=Path(snapshot_path).name)
                    msg.attach(img)
            
            # Send email with retry logic
            for attempt in range(self.email_config['retry_attempts']):
                try:
                    server = smtplib.SMTP(
                        self.email_config['smtp_server'],
                        self.email_config['smtp_port']
                    )
                    server.starttls()
                    server.login(
                        self.email_config['sender_email'],
                        self.email_config['sender_password']
                    )
                    server.send_message(msg)
                    server.quit()
                    
                    logger.info(f"Email alert sent successfully (attempt {attempt + 1})")
                    return
                except Exception as e:
                    logger.warning(f"Email send attempt {attempt + 1} failed: {e}")
                    if attempt == self.email_config['retry_attempts'] - 1:
                        raise
        
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
    
    @staticmethod
    def _format_details(details: Dict) -> str:
        """Format details dictionary for email body"""
        lines = []
        for key, value in details.items():
            if key != 'snapshot_path':
                lines.append(f"  - {key}: {value}")
        return '\n'.join(lines) if lines else "  No additional details"


if __name__ == "__main__":
    # Test alert system
    import numpy as np
    
    alert_system = AlertSystem()
    
    # Create dummy frame
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(test_frame, "TEST ALERT", (50, 240), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
    
    # Trigger test alert
    alert_system.trigger_alert(
        "UNKNOWN_PERSON",
        cam_id=0,
        frame=test_frame,
        details={"confidence": 0.45, "reason": "Below threshold"}
    )
    
    logger.info("Alert system test completed")
