import streamlit as st
import cv2
import tempfile
import time
import os
import pandas as pd
import yt_dlp
import numpy as np
import plotly.express as px
from fpdf import FPDF
from io import BytesIO
from src.detection import VehicleDetector
from src.violations import ViolationDetector
from src.analytics import TrafficAnalytics
from src.utils import Visualizer

# --- UTILS ---
def get_stream_url(url):
    ydl_opts = {'format': 'best', 'quiet': True, 'no_warnings': True}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            return ydl.extract_info(url, download=False)['url']
    except Exception as e:
        st.error(f"Error extracting stream: {e}")
        return None

def generate_pdf_report(violations_list):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, "Smart AI traffic system - Violation Report", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(40, 10, "Time", border=1); pdf.cell(40, 10, "Vehicle ID", border=1); pdf.cell(110, 10, "Violation Type", border=1); pdf.ln()
    pdf.set_font("Arial", "", 12)
    for v in violations_list:
        # Clean text to remove emojis/special characters not supported by standard PDF fonts
        clean_type = str(v['Type']).encode('ascii', 'ignore').decode('ascii').strip()
        pdf.cell(40, 10, str(v['Time']), border=1); pdf.cell(40, 10, str(v['ID']), border=1); pdf.cell(110, 10, clean_type, border=1); pdf.ln()
    return bytes(pdf.output())

# --- UI STYLING ---
def set_custom_style():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
        
        * { font-family: 'Inter', sans-serif; }
        
        .main { 
            background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
            background-size: 400% 400%;
            animation: gradient 15s ease infinite;
            padding: 0 !important;
        }

        @keyframes gradient {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        
        .feature-title {
            text-align: center;
            color: white;
            font-size: 3.5rem;
            font-weight: 800;
            margin: 4rem 0 2rem 0;
            text-shadow: 3px 3px 15px rgba(0,0,0,0.3);
            letter-spacing: -1px;
        }
        
        .feature-card-container {
            display: flex;
            justify-content: center;
            gap: 25px;
            padding: 40px;
            flex-wrap: wrap;
        }
        
        .feature-card {
            border-radius: 25px;
            padding: 35px;
            width: 300px;
            min-height: 420px;
            text-align: center;
            color: white;
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            box-shadow: 0 15px 35px rgba(0,0,0,0.2);
            border: 1px solid rgba(255, 255, 255, 0.2);
            position: relative;
            overflow: hidden;
        }

        .card-1 { background: linear-gradient(135deg, #ff6b6b 0%, #ee0979 100%); }
        .card-2 { background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); }
        .card-3 { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
        .card-4 { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); }
        
        .feature-card:hover {
            transform: translateY(-15px) scale(1.03);
            box-shadow: 0 25px 50px rgba(0,0,0,0.3);
            border: 1px solid rgba(255, 255, 255, 0.5);
        }
        
        .feature-icon {
            font-size: 4rem;
            margin-bottom: 25px;
            display: block;
            filter: drop-shadow(2px 4px 6px rgba(0,0,0,0.2));
        }
        
        .feature-name {
            font-size: 1.7rem;
            font-weight: 800;
            margin-bottom: 20px;
            line-height: 1.1;
            text-shadow: 1px 1px 4px rgba(0,0,0,0.2);
        }
        
        .feature-desc {
            font-size: 1.05rem;
            opacity: 0.95;
            line-height: 1.5;
            font-weight: 400;
        }
        
        .tech-title {
            text-align: center;
            color: white;
            font-size: 2.8rem;
            font-weight: 800;
            margin: 5rem 0 3rem 0;
            text-shadow: 2px 2px 10px rgba(0,0,0,0.2);
        }
        
        .tech-badge {
            background: rgba(255, 255, 255, 0.15);
            backdrop-filter: blur(10px);
            padding: 20px;
            border-radius: 20px;
            text-align: center;
            border: 1px solid rgba(255, 255, 255, 0.2);
            transition: 0.3s;
        }
        .tech-badge:hover {
            background: rgba(255, 255, 255, 0.25);
            transform: scale(1.05);
        }

        .control-panel { 
            background: rgba(15, 23, 42, 0.8);
            backdrop-filter: blur(20px);
            padding: 40px;
            border-radius: 30px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            margin-top: 50px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.4);
            position: relative;
            z-index: 10;
        }
        
        .stMetric { 
            background: rgba(30, 41, 59, 0.7); 
            padding: 25px; 
            border-radius: 20px; 
            border: 1px solid rgba(255, 255, 255, 0.1); 
            box-shadow: 0 10px 20px rgba(0,0,0,0.2);
        }
        
        .video-container { 
            border: 10px solid rgba(15, 23, 42, 0.9); 
            border-radius: 30px; 
            overflow: hidden; 
            background: #000; 
            max-width: 900px; 
            margin: auto; 
            box-shadow: 0 30px 60px rgba(0,0,0,0.5);
        }

        .stApp { background: transparent; }
        h1, h2, h3, p, span, label, .stMetric div { color: white !important; }
        .stButton>button {
            background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%) !important;
            border: none !important;
            color: white !important;
            font-weight: 800 !important;
            letter-spacing: 1px;
            text-transform: uppercase;
            transition: all 0.3s ease !important;
        }
        .stButton>button:hover {
            transform: scale(1.05) !important;
            box-shadow: 0 10px 20px rgba(79, 172, 254, 0.4) !important;
            background: linear-gradient(90deg, #00f2fe 0%, #4facfe 100%) !important;
        }

        /* Fix URL input overlap and hide 'Press Enter to apply' */
        div[data-testid="stTextInput"] div[data-testid="InputInstructions"] {
            display: none !important;
        }
        div[data-testid="stTextInput"] input {
            background: rgba(30, 41, 59, 0.7) !important;
            color: white !important;
            border-radius: 10px !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            padding: 10px 15px !important;
        }

        /* Ensure Selectbox doesn't look like an editable input and remove focus cursor */
        div[data-testid="stSelectbox"] [role="combobox"] {
            background: rgba(30, 41, 59, 0.7) !important;
            border-radius: 10px !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
        }
        div[data-testid="stSelectbox"] input {
            cursor: pointer !important;
            caret-color: transparent !important;
            user-select: none !important;
        }
        </style>
    """, unsafe_allow_html=True)

# --- LANDING PAGE SECTION ---
def render_landing_page():
    st.markdown('<h1 class="feature-title">✨ Project Features</h1>', unsafe_allow_html=True)
    
    cards_html = """
    <div class="feature-card-container">
        <div class="feature-card card-1">
            <span class="feature-icon">🚦</span>
            <div class="feature-name">Dynamic Signal Control</div>
            <div class="feature-desc">Automatically adjusts green light duration based on real-time vehicle density, reducing unnecessary waiting times.</div>
        </div>
        <div class="feature-card card-2">
            <span class="feature-icon">🚑</span>
            <div class="feature-name">Ambulance Priority</div>
            <div class="feature-desc">Detects emergency vehicles and provides an immediate green light corridor to reduce response times and save lives.</div>
        </div>
        <div class="feature-card card-3">
            <span class="feature-icon">📸</span>
            <div class="feature-name">AI ANPR System</div>
            <div class="feature-desc">Automatic Number Plate Recognition (ANPR) detects and logs vehicle registration plates for security and tracking.</div>
        </div>
        <div class="feature-card card-4">
            <span class="feature-icon">🔮</span>
            <div class="feature-name">Traffic Prediction</div>
            <div class="feature-desc">Predicts future traffic density trends using AI-based temporal analysis to prevent congestion before it happens.</div>
        </div>
    </div>
    """
    st.markdown(cards_html, unsafe_allow_html=True)
    
    st.markdown('<h2 class="tech-title">🛠️Tools & Technology</h2>', unsafe_allow_html=True)
    
    t1, t2, t3, t4 = st.columns(4)
    with t1: st.markdown('<div class="tech-badge">🐍<br><b>Python</b><br>Core Logic</div>', unsafe_allow_html=True)
    with t2: st.markdown('<div class="tech-badge">🎯<br><b>YOLOv11</b><br>AI Detection</div>', unsafe_allow_html=True)
    with t3: st.markdown('<div class="tech-badge">⚡<br><b>Streamlit</b><br>Pro Dashboard</div>', unsafe_allow_html=True)
    with t4: st.markdown('<div class="tech-badge">👁️<br><b>OpenCV</b><br>Vision Engine</div>', unsafe_allow_html=True)

# --- MAIN DASHBOARD ---
def run_dashboard():
    st.set_page_config(page_title="Smart AI Traffic System", page_icon="🚦", layout="wide")
    set_custom_style()

    # Session State
    if 'is_running' not in st.session_state: st.session_state.is_running = False
    if 'paused' not in st.session_state: st.session_state.paused = False
    if 'violations_data' not in st.session_state: st.session_state.violations_data = []
    if 'frame_idx' not in st.session_state: st.session_state.frame_idx = 0

    # Render landing page features
    render_landing_page()

    # Sidebar Controls
    st.sidebar.title("⚙️ AI Control Center")
    conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.25)
    frame_skip = st.sidebar.slider("Frame Skipping (Speed)", 1, 10, 2)
    show_boxes = st.sidebar.toggle("Show Bounding Boxes", value=True)
    show_labels = st.sidebar.toggle("Show Labels", value=True)
    show_heatmap = st.sidebar.toggle("Show Heatmap", value=False)
    
    st.sidebar.divider()
    st.sidebar.subheader("🚨 Violation Rules")
    enable_signal = st.sidebar.toggle("Signal Violation", value=True)
    enable_lane = st.sidebar.toggle("Wrong Lane Detection", value=True)
    enable_accident = st.sidebar.toggle("Accident Detection", value=True)

    st.markdown("<h1>🚦 Smart AI traffic system</h1>", unsafe_allow_html=True)

    # Top Configuration Panel
    col_config, _ = st.columns([4, 1])
    with col_config:
        st.markdown('<div class="control-panel">', unsafe_allow_html=True)
        c1, c2, c3 = st.columns([3, 2, 1])
        with c1:
            source_type = st.selectbox("Source", ["Upload Video", "Social URL", "Live Camera"], key="source_selectbox")
            if source_type == "Upload Video":
                uploaded = st.file_uploader("Upload", type=["mp4", "avi", "mov"])
                video_path = tempfile.NamedTemporaryFile(delete=False).name if uploaded else None
                if uploaded:
                    with open(video_path, 'wb') as f: f.write(uploaded.read())
            elif source_type == "Social URL":
                url = st.text_input("URL", placeholder="YouTube link...")
                video_path = get_stream_url(url) if url else None
            else: video_path = 0
        
        with c2:
            signal_state = st.radio("Traffic Signal", ["GREEN", "RED"], horizontal=True)
            stop_line_y = st.slider("Stop Line Position", 0, 1000, 500)
        
        with c3:
            st.write("")
            if st.button("🚀START ENGINE" if not st.session_state.is_running else "⏹️ STOP", width="stretch"):
                st.session_state.is_running = not st.session_state.is_running
                if st.session_state.is_running:
                    st.session_state.frame_idx = 0 # Reset when starting new engine run
                else:
                    st.session_state.paused = False
        st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.is_running and video_path is not None:
        # Metrics Row
        m1, m2, m3, m4, m5 = st.columns(5)
        metric_total = m1.empty(); metric_cars = m2.empty(); metric_bikes = m3.empty()
        metric_status = m4.empty(); metric_violation = m5.empty()

        # Video Row
        col_vid, col_stats = st.columns([3, 2])
        with col_vid:
            st.markdown('<div class="video-container">', unsafe_allow_html=True)
            frame_placeholder = st.empty()
            st.markdown('</div>', unsafe_allow_html=True)
            progress_bar = st.progress(0, text="Processing Progress")
            
            pb_c1, pb_c2 = st.columns(2)
            if pb_c1.button("▶️ Resume"): st.session_state.paused = False
            if pb_c2.button("⏸️ Pause"): st.session_state.paused = True

        with col_stats:
            st.subheader("📊 Real-time Analytics")
            chart_timeline = st.empty()
            
            chart_dist = st.empty()
            violation_log = st.empty()

        # Core Engine
        detector = VehicleDetector()
        violation_detector = ViolationDetector()
        analytics = TrafficAnalytics(line_y=stop_line_y)
        visualizer = Visualizer()
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Resume from last frame index if restarted (e.g. on pause/resume click)
        if st.session_state.frame_idx > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.frame_idx)
        
        while cap.isOpened() and st.session_state.is_running:
            if st.session_state.paused: time.sleep(0.1); continue
            
            ret, frame = cap.read()
            if not ret: break
            
            st.session_state.frame_idx += 1
            if st.session_state.frame_idx % frame_skip != 0: continue
            
            # Update Progress
            if total_frames > 0:
                progress_bar.progress(min(st.session_state.frame_idx / total_frames, 1.0), text=f"Processing Frame {st.session_state.frame_idx}/{total_frames}")

            # AI Inference
            results = detector.track(frame, conf=conf_threshold)
            
            current_detections = [] # For accident detection
            if results and results.boxes is not None and results.boxes.id is not None:
                for box in results.boxes:
                    tid = int(box.id[0]); cid = int(box.cls[0])
                    cname = detector.class_names.get(cid, 'car')
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    current_detections.append((tid, (cx, cy), cname))
                    
                    # Logic
                    analytics.update_analytics(tid, (cx, cy), cname)
                    
                    # 🚑 EMERGENCY VEHICLE DETECTION & AUTO GREEN SIGNAL
                    is_emergency = violation_detector.check_emergency(tid, cname)
                    if is_emergency:
                        signal_state = "GREEN" # AUTO OVERRIDE
                        st.session_state.violations_data.append({"Time": time.strftime("%H:%M:%S"), "ID": tid, "Type": "🚑 Emergency Priority (Green Light)"})
                    
                    if enable_signal and violation_detector.check_red_light(tid, (cx, cy), stop_line_y, signal_state):
                        st.session_state.violations_data.append({"Time": time.strftime("%H:%M:%S"), "ID": tid, "Type": "Signal Jumping"})
                    
                    if enable_accident and violation_detector.detect_accident(tid, (cx, cy), current_detections):
                        st.session_state.violations_data.append({"Time": time.strftime("%H:%M:%S"), "ID": tid, "Type": "Potential Accident"})
                    
                    # 📸 ANPR DETECTION
                    plate = violation_detector.extract_number_plate(tid, frame, (x1, y1, x2, y2))
                    
                    violation_detector.update_history(tid, (cx, cy))

            # Visuals
            frame = visualizer.draw_line(frame, ((0, stop_line_y), (frame.shape[1], stop_line_y)), "Stop Line")
            frame = visualizer.draw_tracking(frame, results, show_boxes, show_labels, detector.class_names, violation_detector)
            if show_heatmap:
                frame = visualizer.draw_heatmap(frame, analytics.heatmap_data)
            
            frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", width="stretch")

            # Update Metrics & Charts
            sumry = analytics.get_summary()
            counts = sumry['breakdown']
            metric_total.metric("🚗 Total", sumry['total_vehicles'])
            metric_cars.metric("🚙 Cars", counts.get('car', 0))
            metric_bikes.metric("🏍️ Bikes", counts.get('motorcycle', 0))
            metric_status.metric("🔮 AI Prediction", analytics.predict_future_density())
            metric_violation.metric("🚨 Alarms", len(st.session_state.violations_data))

            # Charts
            if len(sumry['time_history']) > 1:
                df_time = pd.DataFrame(sumry['time_history'], columns=['Time', 'Count'])
                df_time['Time'] = pd.to_datetime(df_time['Time'], unit='s')
                fig_line = px.line(df_time, x='Time', y='Count', title="Vehicle Influx Over Time", template="plotly_dark")
                fig_line.update_layout(showlegend=False, margin=dict(l=20, r=20, t=40, b=20))
                chart_timeline.plotly_chart(fig_line, width="stretch", key=f"timeline_{st.session_state.frame_idx}")

            fig_pie = px.pie(values=list(counts.values()), names=list(counts.keys()), title="Vehicle Distribution", template="plotly_dark")
            fig_pie.update_layout(margin=dict(l=20, r=20, t=40, b=20))
            chart_dist.plotly_chart(fig_pie, width="stretch", key=f"dist_{st.session_state.frame_idx}")

            if st.session_state.violations_data:
                violation_log.dataframe(pd.DataFrame(st.session_state.violations_data).tail(5), width="stretch")

        cap.release()

    # Footer Reports
    if st.session_state.violations_data:
        st.divider()
        st.subheader("📥 Download Reports")
        d1, d2, d3 = st.columns(3)
        d1.download_button("📄 PDF Report", generate_pdf_report(st.session_state.violations_data), "report.pdf")
        d2.download_button("📊 CSV Data", pd.DataFrame(st.session_state.violations_data).to_csv(index=False), "data.csv")
        if d3.button("🗑️ Reset All"):
            st.session_state.violations_data = []
            st.rerun()

if __name__ == "__main__":
    run_dashboard()
