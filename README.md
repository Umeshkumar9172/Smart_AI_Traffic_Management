# 🚦 Smart AI Traffic System

Smart AI Traffic System is a high-performance, real-time traffic monitoring and management solution powered by Computer Vision and Deep Learning. It aims to reduce congestion, improve safety, and provide priority access for emergency vehicles through intelligent video analysis.

---

## ✨ Key Features

- **🚑 Ambulance Priority (Green Corridor)**: Automatically detects emergency vehicles (Ambulances, Fire Trucks) and overrides traffic signals to GREEN to ensure life-saving speed.
- **🚗 Real-time Vehicle Tracking**: Uses YOLOv11 and ByteTrack for persistent identification and counting of Cars, Motorcycles, Buses, and Trucks.
- **📸 AI ANPR (Number Plate Detection)**: Automatically identifies and logs vehicle registration plates for security and violation tracking.
- **🔮 Traffic Density Prediction**: Analyzes vehicle influx trends to forecast future congestion levels, enabling proactive traffic management.
- **🚨 Violation Detection**: 
  - **Signal Jumping**: Detects vehicles crossing the stop line during RED signals.
  - **Wrong Lane Detection**: Identifies vehicles driving in restricted or opposite lanes.
  - **Accident Detection**: Real-time alerts for potential collisions or sudden stops.
- **📊 Interactive Analytics**: High-end dashboard with live charts, vehicle distribution, and downloadable PDF/CSV reports.

---

## 🛠️ Tech Stack

- **Core Engine**: Python 3.10+
- **AI/ML**: YOLOv11 (Ultralytics), ByteTrack
- **Computer Vision**: OpenCV
- **Dashboard**: Streamlit
- **Analytics**: Plotly, Pandas
- **Reports**: FPDF2 (Unicode-safe)

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/traffic-management.git
cd traffic-management
```

### 2. Set Up Virtual Environment
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Dashboard
```bash
streamlit run app.py
```

---

## 📂 Project Structure

- `app.py`: Main Streamlit dashboard and UI logic.
- `src/detection.py`: YOLOv11 inference and tracking engine.
- `src/violations.py`: Logic for emergency detection, ANPR, and traffic violations.
- `src/analytics.py`: Traffic density calculation and AI trend prediction.
- `src/utils.py`: Visualization helpers for drawing bounding boxes and overlays.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
