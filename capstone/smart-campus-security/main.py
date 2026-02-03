"""
Main application for Smart Campus Security & Attendance 2.0
Combines Streamlit dashboard with FastAPI backend
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from pathlib import Path
import threading
import queue
import time
from typing import Dict, List
import yaml

# Import local modules
from db import Database
from detector import FaceDetector
from utils.tracker import MultiCamTracker
from utils.alerts import AlertSystem
from utils.anomaly import AnomalyDetector, LateArrivalPredictor, AnalyticsEngine
from utils.logger import setup_logger
from loguru import logger

# Setup logger
setup_logger()

# Page config
st.set_page_config(
    page_title="Smart Campus Security",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load configuration
@st.cache_resource
def load_config():
    with open("config.yaml", 'r') as f:
        return yaml.safe_load(f)

config = load_config()

# Initialize components
@st.cache_resource
def init_components():
    """Initialize all system components"""
    db = Database()
    detector = FaceDetector()
    tracker = MultiCamTracker(
        similarity_threshold=config['tracking']['similarity_threshold'],
        track_timeout=config['tracking']['track_timeout']
    )
    alert_system = AlertSystem()
    
    return db, detector, tracker, alert_system

db, detector, tracker, alert_system = init_components()

# Camera capture class
class CameraCapture:
    """Thread-safe camera capture"""
    
    def __init__(self, cam_id: int, name: str):
        self.cam_id = cam_id
        self.name = name
        self.cap = None
        self.frame_queue = queue.Queue(maxsize=2)
        self.running = False
        self.thread = None
        
    def start(self):
        """Start capture thread"""
        if self.running:
            return
        
        self.cap = cv2.VideoCapture(self.cam_id)
        if not self.cap.isOpened():
            logger.error(f"Cannot open camera {self.cam_id}")
            return False
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config['performance']['resize_width'])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config['performance']['resize_height'])
        self.cap.set(cv2.CAP_PROP_FPS, config['performance']['max_fps'])
        
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        
        logger.info(f"Started camera {self.cam_id}: {self.name}")
        return True
    
    def _capture_loop(self):
        """Capture loop running in thread"""
        frame_count = 0
        
        while self.running:
            ret, frame = self.cap.read()
            
            if not ret:
                logger.warning(f"Failed to read from camera {self.cam_id}")
                time.sleep(0.1)
                continue
            
            # Frame skipping for performance
            frame_count += 1
            if frame_count % config['performance']['frame_skip'] != 0:
                continue
            
            # Put frame in queue (drop old frames if queue full)
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            
            try:
                self.frame_queue.put(frame, block=False)
            except queue.Full:
                pass
    
    def get_frame(self):
        """Get latest frame"""
        try:
            return self.frame_queue.get(timeout=1.0)
        except queue.Empty:
            return None
    
    def stop(self):
        """Stop capture"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()
        logger.info(f"Stopped camera {self.cam_id}")

# Initialize session state
if 'cameras' not in st.session_state:
    st.session_state.cameras = {}
    st.session_state.camera_running = False
    st.session_state.detections_log = []

# Sidebar navigation
st.sidebar.title("🎓 Smart Campus Security")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["🎥 Live View", "📊 Analytics", "🚨 Alerts", "👤 Registration", "📄 Reports", "⚙️ Settings"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("System Status")

# Display system stats
tracker_stats = tracker.get_stats()
st.sidebar.metric("Active Tracks", tracker_stats['active_tracks'])
st.sidebar.metric("Total Registered", len(db.get_all_roles()))

# Role distribution
if tracker_stats['role_distribution']:
    st.sidebar.markdown("**Current Roles:**")
    for role, count in tracker_stats['role_distribution'].items():
        color = {
            'student': '🟢',
            'staff': '🔵',
            'labour': '🟠',
            'unknown': '🔴'
        }.get(role, '⚪')
        st.sidebar.write(f"{color} {role.title()}: {count}")

# Main content area
if page == "🎥 Live View":
    st.title("Live Camera Feed")
    
    # Camera controls
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("▶️ Start Cameras" if not st.session_state.camera_running else "⏸️ Stop Cameras"):
            if not st.session_state.camera_running:
                # Start cameras
                for cam_config in config['cameras']:
                    if cam_config['enabled']:
                        cam = CameraCapture(cam_config['id'], cam_config['name'])
                        if cam.start():
                            st.session_state.cameras[cam_config['id']] = cam
                st.session_state.camera_running = True
                st.success("Cameras started!")
            else:
                # Stop cameras
                for cam in st.session_state.cameras.values():
                    cam.stop()
                st.session_state.cameras = {}
                st.session_state.camera_running = False
                st.info("Cameras stopped")
            st.rerun()
    
    with col2:
        if st.button("🔄 Refresh"):
            st.rerun()
    
    # Camera Settings (Sidebar)
    st.sidebar.markdown("---")
    st.sidebar.subheader("📷 Camera Adjustments")
    cam_rotation = st.sidebar.select_slider(
        "Rotation", options=[0, 90, 180, 270], value=0,
        help="Rotate camera feed if it's sideways"
    )
    cam_flip_h = st.sidebar.checkbox("Flip Horizontal (Mirror)", value=False)
    cam_flip_v = st.sidebar.checkbox("Flip Vertical", value=False)
    
    # Display camera feeds
    if st.session_state.camera_running:
        # Create placeholders for each camera
        cam_placeholders = {}
        cols = st.columns(len(st.session_state.cameras))
        
        for idx, (cam_id, cam) in enumerate(st.session_state.cameras.items()):
            with cols[idx]:
                st.subheader(f"📹 {cam.name}")
                cam_placeholders[cam_id] = st.empty()
        
        # Dashboard stats placeholders
        st.markdown("---")
        st.subheader("Recent Detections")
        table_placeholder = st.empty()
        
        # Main Capture Loop
        while st.session_state.camera_running:
            frames_processed = False
            
            for cam_id, cam in st.session_state.cameras.items():
                frame = cam.get_frame()
                
                if frame is not None:
                    frames_processed = True
                    
                    # Apply Rotation/Flip
                    if cam_rotation == 90:
                        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                    elif cam_rotation == 180:
                        frame = cv2.rotate(frame, cv2.ROTATE_180)
                    elif cam_rotation == 270:
                        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    
                    if cam_flip_h:
                        frame = cv2.flip(frame, 1)
                    if cam_flip_v:
                        frame = cv2.flip(frame, 0)
                        
                    # Process frame
                    processed_frame, detections = detector.process_frame(
                        frame, cam_id, db, tracker, alert_system
                    )
                    
                    # Store detections
                    if detections:
                        for det in detections:
                            det['timestamp'] = datetime.now()
                            det['camera'] = cam.name
                            st.session_state.detections_log.append(det)
                        
                        # Keep only recent detections
                        st.session_state.detections_log = st.session_state.detections_log[-50:]
                    
                    # Convert BGR to RGB for display
                    display_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    
                    # Update placeholder
                    cam_placeholders[cam_id].image(display_frame, caption=f"Detected: {len(detections)}", use_container_width=True)
            
            # Update table occasionally (every 10 frames) to save UI render time
            if frames_processed and len(st.session_state.detections_log) > 0:
                if int(time.time() * 10) % 10 == 0:  # ~1 sec update
                    recent_df = pd.DataFrame(st.session_state.detections_log[-10:])
                    recent_df = recent_df[['timestamp', 'name', 'role_type', 'confidence']]
                    recent_df['confidence'] = recent_df['confidence'].apply(lambda x: f"{x:.2%}")
                    table_placeholder.dataframe(recent_df, use_container_width=True, hide_index=True)
            
            # Small sleep to prevent UI freeze
            if not frames_processed:
                time.sleep(0.01)
            else:
                time.sleep(0.001)
                
            # Check if stop button pressed (requires unique key in loop)
            # Note: In loop, we can't easily check button clicks without rerunning.
            # So we rely on the Sidebar 'Stop' button which triggers rerun.
            
    else:
        st.info("Click 'Start Cameras' to begin live monitoring")

elif page == "📊 Analytics":
    st.title("Analytics Dashboard")
    
    # Date range selector
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", datetime.now() - timedelta(days=7))
    with col2:
        end_date = st.date_input("End Date", datetime.now())
    
    # Get analytics data
    analytics_data = db.get_analytics(
        datetime.combine(start_date, datetime.min.time()),
        datetime.combine(end_date, datetime.max.time())
    )
    
    if analytics_data:
        df = pd.DataFrame(analytics_data)
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Entries", df['total_entries'].sum())
        with col2:
            st.metric("Unique Persons", df['unique_persons'].sum())
        with col3:
            st.metric("Avg Confidence", f"{df['avg_confidence'].mean():.2%}")
        with col4:
            mask_rate = (df['mask_count'].sum() / df['total_entries'].sum() * 100) if df['total_entries'].sum() > 0 else 0
            st.metric("Mask Compliance", f"{mask_rate:.1f}%")
        
        st.markdown("---")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Role distribution pie chart
            st.subheader("Role Distribution")
            role_counts = df.groupby('role_type')['total_entries'].sum().reset_index()
            
            fig = px.pie(
                role_counts,
                values='total_entries',
                names='role_type',
                color='role_type',
                color_discrete_map={
                    'student': '#00FF00',
                    'staff': '#0000FF',
                    'labour': '#FFA500',
                    'unknown': '#FF0000'
                },
                title="Entries by Role"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Daily trend line chart
            st.subheader("Daily Attendance Trend")
            daily_trend = df.groupby('date')['total_entries'].sum().reset_index()
            
            fig = px.line(
                daily_trend,
                x='date',
                y='total_entries',
                title="Daily Entry Count",
                markers=True
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Late arrivals
        st.markdown("---")
        st.subheader("Late Arrivals")
        
        late_arrivals = db.get_late_arrivals(
            datetime.now(),
            threshold_minutes=config['alerts']['late_entry']['threshold_minutes']
        )
        
        if late_arrivals:
            late_df = pd.DataFrame(late_arrivals)
            st.dataframe(late_df, use_container_width=True, hide_index=True)
        else:
            st.success("No late arrivals today!")
        
        # Anomaly detection
        if config['analytics']['anomaly_detection']['enabled']:
            st.markdown("---")
            st.subheader("Anomaly Detection")
            
            # Get logs for anomaly detection
            conn = db.get_connection()
            logs_df = pd.read_sql_query("""
                SELECT * FROM logs
                WHERE timestamp BETWEEN ? AND ?
            """, conn, params=[
                datetime.combine(start_date, datetime.min.time()),
                datetime.combine(end_date, datetime.max.time())
            ])
            conn.close()
            
            if len(logs_df) > 10:
                anomaly_detector = AnomalyDetector()
                anomaly_detector.train(logs_df)
                predictions = anomaly_detector.predict(logs_df)
                
                anomaly_count = np.sum(predictions == -1)
                st.metric("Anomalies Detected", anomaly_count)
                
                if anomaly_count > 0:
                    anomaly_df = logs_df[predictions == -1]
                    st.dataframe(anomaly_df[['timestamp', 'role_type', 'cam_id']], 
                               use_container_width=True, hide_index=True)
            else:
                st.info("Insufficient data for anomaly detection")
    
    else:
        st.info("No data available for selected date range")

elif page == "🚨 Alerts":
    st.title("Security Alerts")
    
    # Get recent alerts
    alerts = db.get_recent_alerts(limit=50)
    
    if alerts:
        # Alert summary
        alert_df = pd.DataFrame(alerts)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Alerts", len(alert_df))
        with col2:
            unack_count = len(alert_df[alert_df['acknowledged'] == 0])
            st.metric("Unacknowledged", unack_count)
        with col3:
            if len(alert_df) > 0:
                latest = pd.to_datetime(alert_df['timestamp'].max())
                st.metric("Latest Alert", latest.strftime("%H:%M:%S"))
        
        st.markdown("---")
        
        # Alert type distribution
        alert_counts = alert_df['alert_type'].value_counts()
        
        fig = px.bar(
            x=alert_counts.index,
            y=alert_counts.values,
            labels={'x': 'Alert Type', 'y': 'Count'},
            title="Alert Distribution",
            color=alert_counts.index,
            color_discrete_sequence=px.colors.qualitative.Set1
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Alert list
        st.subheader("Alert History")
        
        for idx, alert in alert_df.iterrows():
            with st.expander(f"{alert['alert_type']} - {alert['timestamp']}", 
                           expanded=(idx < 3)):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**Camera:** {alert['cam_id']}")
                    st.write(f"**Time:** {alert['timestamp']}")
                    
                    if alert['details']:
                        st.write(f"**Details:** {alert['details']}")
                
                with col2:
                    if alert['image_path'] and Path(alert['image_path']).exists():
                        img = cv2.imread(alert['image_path'])
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        st.image(img_rgb, caption="Snapshot", use_container_width=True)
    
    else:
        st.success("No alerts! System running smoothly.")

elif page == "👤 Registration":
    st.title("Register New Person")
    
    with st.form("registration_form"):
        name = st.text_input("Full Name")
        role_type = st.selectbox("Role", ["student", "staff", "labour"])
        
        st.markdown("**Upload 10 images of the person (different angles/lighting)**")
        uploaded_files = st.file_uploader(
            "Choose images",
            type=['jpg', 'jpeg', 'png'],
            accept_multiple_files=True
        )
        
        submit = st.form_submit_button("Register")
        
        if submit:
            if not name:
                st.error("Please enter a name")
            elif not uploaded_files or len(uploaded_files) < 5:
                st.error("Please upload at least 5 images")
            else:
                with st.spinner("Processing images..."):
                    embeddings = []
                    
                    for uploaded_file in uploaded_files:
                        # Read image
                        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                        
                        # Detect face
                        faces = detector.detect_faces(img)
                        
                        if faces:
                            embeddings.append(faces[0]['embedding'])
                        else:
                            st.warning(f"No face detected in {uploaded_file.name}")
                    
                    if len(embeddings) >= 5:
                        # Register person
                        person_id = db.register_role(name, role_type, embeddings)
                        st.success(f"✅ Successfully registered {name} as {role_type} (ID: {person_id})")
                        st.balloons()
                    else:
                        st.error("Could not detect faces in enough images. Please try again.")
    
    # Show registered persons
    st.markdown("---")
    st.subheader("Registered Persons")
    
    roles = db.get_all_roles()
    if roles:
        roles_df = pd.DataFrame(roles)
        st.dataframe(roles_df, use_container_width=True, hide_index=True)
    else:
        st.info("No persons registered yet")

elif page == "📄 Reports":
    st.title("Generate Reports")
    
    report_type = st.radio("Report Type", ["Daily Report", "Weekly Report", "Attendance Summary"])
    
    if report_type == "Attendance Summary":
        report_date = st.date_input("Select Date", datetime.now())
        
        if st.button("Generate Summary"):
            summary_data = db.get_daily_summary(datetime.combine(report_date, datetime.min.time()))
            
            if summary_data:
                st.success(f"Attendance Summary for {report_date}")
                
                # Create nice dataframe
                df = pd.DataFrame(summary_data)
                display_df = df[['name', 'role_type', 'entry_str', 'exit_str', 'duration_str', 'detections']].copy()
                display_df.columns = ['Name', 'Role', 'Entry Time', 'Exit Time', 'Duration', 'Detections']
                
                st.dataframe(display_df, use_container_width=True, hide_index=True)
                
                # CSV Export
                csv = display_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Summary CSV",
                    csv,
                    f"attendance_summary_{report_date}.csv",
                    "text/csv"
                )
            else:
                st.warning("No attendance records found for this date")
                
    elif report_type == "Daily Report":
        report_date = st.date_input("Select Date", datetime.now())
        
        if st.button("Generate Report"):
            # Get logs for the date
            conn = db.get_connection()
            logs_df = pd.read_sql_query("""
                SELECT * FROM logs
                WHERE DATE(timestamp) = ?
            """, conn, params=[report_date])
            conn.close()
            
            if len(logs_df) > 0:
                report = AnalyticsEngine.generate_daily_report(logs_df, datetime.combine(report_date, datetime.min.time()))
                
                # Display report
                st.json(report)
                
                # Export options
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("📥 Export CSV"):
                        csv = logs_df.to_csv(index=False)
                        st.download_button(
                            "Download CSV",
                            csv,
                            f"report_{report_date}.csv",
                            "text/csv"
                        )
                
                with col2:
                    st.info("PDF export coming soon!")
            else:
                st.warning("No data for selected date")
    
    else:  # Weekly Report
        week_start = st.date_input("Week Start Date", datetime.now() - timedelta(days=7))
        
        if st.button("Generate Report"):
            conn = db.get_connection()
            week_end = week_start + timedelta(days=7)
            
            logs_df = pd.read_sql_query("""
                SELECT * FROM logs
                WHERE DATE(timestamp) BETWEEN ? AND ?
            """, conn, params=[week_start, week_end])
            conn.close()
            
            if len(logs_df) > 0:
                report = AnalyticsEngine.generate_weekly_report(logs_df, datetime.combine(week_start, datetime.min.time()))
                
                st.json(report)
                
                # Export
                if st.button("📥 Export CSV"):
                    csv = logs_df.to_csv(index=False)
                    st.download_button(
                        "Download CSV",
                        csv,
                        f"weekly_report_{week_start}.csv",
                        "text/csv"
                    )
            else:
                st.warning("No data for selected week")

elif page == "⚙️ Settings":
    st.title("System Settings")
    
    st.subheader("Camera Configuration")
    st.json(config['cameras'])
    
    st.subheader("Recognition Settings")
    st.json(config['recognition'])
    
    st.subheader("Alert Settings")
    st.json(config['alerts'])
    
    st.info("To modify settings, edit config.yaml and restart the application")

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("Smart Campus Security v2.0")
st.sidebar.caption("Offline-first • Multi-camera • AI-powered")
