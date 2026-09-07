# Real-Time Hand Recognition & Landmark Tracking with OpenCV and MediaPipe

A high-performance Computer Vision application built with **OpenCV** and **Google MediaPipe Tasks** (`HandLandmarker`) that detects hands in real-time, tracks **21 3D skeletal landmarks** per hand, identifies **Left vs. Right handedness**, computes **palm center points**, and renders color-coded bounding boxes and hand skeletons over live webcam feeds.

---

## 🖐️ What is Hand Recognition Used For?

Hand recognition and gesture tracking convert raw video frames into structured digital spatial data (x, y, z joint coordinates). This technology forms the foundation for modern touchless, spatial, and intelligent user experiences across various industries:

| Application Domain | Practical Real-World Uses |
| :--- | :--- |
| **Touchless & Gesture Controls** | Control software, media players, smart TVs, digital kiosks, and public displays without physical touch (hygienic and intuitive). |
| **Virtual & Augmented Reality (VR/AR)** | Direct spatial interaction with 3D virtual objects, menus, virtual keyboards, and spatial computing environments (e.g., Apple Vision Pro, Meta Quest). |
| **Sign Language Recognition (SLR)** | Translating deaf/hard-of-hearing sign language gestures into text or audio speech in real-time to aid accessibility. |
| **Gaming & Interactive Media** | Motion-controlled gaming where players interact using natural hand gestures instead of hardware controllers or keyboards. |
| **Human-Robot Interaction (HRI)** | Guiding industrial robotic arms, medical equipment, or drones using intuitive hand signals and directional gestures. |
| **Health & Physical Rehabilitation** | Tracking finger joint mobility, measuring hand tremors, analyzing fine motor skills, and guiding physical therapy exercises. |
| **Automotive & Smart Dashboards** | Driver interaction with infotainment systems (volume control, answering calls) without taking eyes off the road, plus driver safety monitoring. |
| **Biometrics & Security** | Palm geometry analysis and touchless hand gesture authentication systems. |

---

## ✨ Key Features of This Repository

- 📍 **21 Skeletal Landmarks Tracking**: Detects and tracks 21 key joint points per hand with high accuracy and low latency.
- 🖐️ **Left & Right Hand Classification**: Automatically identifies handedness and corrects for mirror-flipped webcam views.
- 🎯 **Palm Center Calculation**: Computes the geometric center of the palm from the wrist and MCP joints.
- 📦 **Dynamic Bounding Boxes & Skeletons**: Renders visual bounding frames and skeletal connections color-coded by hand:
  - **Left Hand**: Cyan indicator (`BGR: 255, 255, 0`)
  - **Right Hand**: Bright Green indicator (`BGR: 0, 255, 0`)
- 🤖 **Automatic Model Downloader**: Automatically checks for and fetches the official Google MediaPipe pretrained `hand_landmarker.task` model file if not locally present.
- 🖼️ **Adaptive OpenCV Window**: Automatically resizes the webcam display output to match window dimensions smoothly.

---

## 📂 Project Structure

```text
openCV/
├── camera_acess.py       # Main application script with live webcam feed & hand tracking pipeline
├── hand_recognition.py    # Hand detection & landmarker implementation script
├── hand_landmarker.task  # Pretrained MediaPipe hand landmarker model file (auto-downloaded)
├── README.md             # Project documentation and guide
└── .gitignore            # Git configuration rules
```

### File Details
- **[`camera_acess.py`](file:///d:/GitHub/Python/openCV/camera_acess.py)**: Captures webcam input (`cv2.VideoCapture(0)`), flips the frame horizontally for natural mirror interaction, converts images to RGB for MediaPipe, calculates landmark coordinates, and draws colored overlays and confidence banners.
- **[`hand_recognition.py`](file:///d:/GitHub/Python/openCV/hand_recognition.py)**: Standalone implementation of MediaPipe Hand Landmarker detection and rendering.
- **`hand_landmarker.task`**: MediaPipe's lightweight neural network task model designed for real-time hand detection and 3D coordinate prediction.

---

## 🦴 MediaPipe 21 Hand Landmarks Reference

MediaPipe maps each detected hand to 21 landmark nodes:

```text
       8      12      16      20
       |       |       |       |
       7      11      15      19
       |       |       |       |
       6      10      14      18
       |       |       |       |
   4   5-------9------13------17
   |  /
   3 /
   |/
   2
   |
   1
   |
   0 (Wrist)
```

| Landmark Index | Anatomical Joint Description |
| :---: | :--- |
| **0** | Wrist |
| **1 - 4** | Thumb (CMC, MCP, IP, Tip) |
| **5 - 8** | Index Finger (MCP, PIP, DIP, Tip) |
| **9 - 12** | Middle Finger (MCP, PIP, DIP, Tip) |
| **13 - 16** | Ring Finger (MCP, PIP, DIP, Tip) |
| **17 - 20** | Pinky Finger (MCP, PIP, DIP, Tip) |

---

## ⚙️ Setup & Installation

### 1. Prerequisites
Ensure you have Python 3.8 or higher installed on your system.

### 2. Install Required Dependencies
Install OpenCV, MediaPipe, and NumPy via `pip`:

```bash
pip install opencv-python mediapipe numpy
```

---

## 🚀 How to Run

Run either script from your terminal:

```bash
python camera_acess.py
```

or

```bash
python hand_recognition.py
```

### Controls & Window Navigation
- **Exit Application**: Press **`q`** or **`x`** on your keyboard while focusing on the video window.
- **Resize Window**: Click and drag the edges of the video window to scale the output.

---

## 🔬 How the Code Works (Step-by-Step)

1. **Model Initialization**:
   The script verifies `hand_landmarker.task`. If missing, it downloads it from Google's official storage. It initializes `HandLandmarker` options with 60% detection/presence/tracking confidence thresholds.
2. **Video Capture & Mirroring**:
   Captures frames at $1280 \times 720$ resolution from webcam index `0`. Flips frames horizontally using `cv2.flip(frame, 1)` so hand movement aligns with mirror expectations.
3. **RGB Conversion & Detection**:
   Converts OpenCV BGR images to MediaPipe `mp.Image` SRGB format and runs `detector.detect(mp_image)`.
4. **Coordinates & Render Pipeline**:
   - Maps normalized $[0.0, 1.0]$ landmark positions to pixel grid coordinates.
   - Calculates hand bounding box around extreme joint points with padding.
   - Computes palm centroid averaging wrist and metacarpal joints ($[0, 1, 5, 9, 13, 17]$).
   - Draws skeleton line connections and joint circles with color differentiation for left/right hands.
   - Displays real-time detection confidence scores.
