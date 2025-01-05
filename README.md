# FALL: Face Away Lock Logic

**FALL** stands for **Face Away Lock Logging** (or “Face-Away Auto-Lock Library”)—a cross-platform Python tool that locks your screen when it no longer detects a face for a specified period. Built using **Mediapipe** and **OpenCV**, it leverages real-time webcam input to detect user presence.

## Features
- **Real-Time Detection:** Continuously monitors webcam input to identify if a face is present.
- **Cross-Platform Screen Locking:** Supports Windows, macOS, and Linux for automatically locking the screen.

## How It Works
1. **Webcam Capture:** Uses OpenCV to grab frames from your default camera.  
2. **Face Detection:** Mediapipe’s FaceMesh locates face landmarks in each frame.  
3. **Face Presence Check:** If no face is detected over a certain number of consecutive frames, the script triggers a screen lock.  
4. **OS-Specific Lock:** Calls built-in commands or functions on Windows, macOS, or Linux to lock the screen.

## Setup & Installation

### Clone the Repo
    git clone https://github.com/YourUsername/FALL.git
    cd FALL

### Install Dependencies
    pip install --upgrade pip
    pip install opencv-python mediapipe numpy

> **Note**: Mediapipe is typically supported on Python 3.7–3.10. If you’re on a newer version (e.g., 3.12), you may need to build Mediapipe from source or use a supported Python version.

### Run the Script
    python fall.py

A window will open displaying your webcam feed.  
If the script doesn’t detect a face for a certain number of consecutive frames (by default **1000**), it locks the screen.

### Debug Messages
The script prints out logs in the console:
- “No face detected this frame.”  
- “No face detected for multiple frames, locking screen...”

## Cross-Platform Locking
- **Windows**: Uses `ctypes.windll.user32.LockWorkStation()`.  
- **macOS**: Uses an AppleScript call with `osascript`.  
- **Linux**: Tries `xdg-screensaver`, then `gnome-screensaver-command`, then `vlock`.  
  You might need to install/enable certain screen-locking packages for Linux (e.g., gnome-screensaver).

## Ideas for Improvement
- **Head Pose-Based Lock:** Originally, the code could lock if your head is turned away too long. Currently, that feature is turned off, but you can restore it for more advanced usage.  
- **Face Recognition:** Expand to lock if the face detected is not the “authorized” user.  
- **Calibration Step:** Capture baseline angles to reduce false positives if the user has a naturally tilted posture or if the camera is at an angle.  
- **GUI:** A small graphical interface with real-time logging and threshold settings.

## License
This project is licensed under the **MIT License**.  
Feel free to **use, modify, and distribute** it, but please **retain the original license** and **credit the author**.

## Disclaimer
- **Security:** This is a proof of concept. It might not be fully secure against all intrusions or system bypasses.  
- **Privacy:** Capturing webcam data should be done carefully. Ensure compliance with relevant privacy regulations or workplace policies if you plan to deploy this beyond personal use.

## Questions or Feedback?
Feel free to reach out via GitHub Issues or **DM me on LinkedIn**
