import cv2
import mediapipe as mp
import time
import platform
import subprocess
import sys
import os
import math
import numpy as np
from collections import deque

#############################################
# SCREEN LOCKING FUNCTION                   #
#############################################
def lock_screen():
    """
    Locks the screen based on the detected operating system.
    """
    os_name = platform.system()

    if os_name == 'Windows':
        # Windows lock using ctypes
        try:
            import ctypes
            ctypes.windll.user32.LockWorkStation()
            print("Locked Windows screen using ctypes.")
        except Exception as e:
            print(f"Error locking Windows screen: {e}")

    elif os_name == 'Darwin':  # macOS
        # Use AppleScript to lock
        try:
            subprocess.call([
                "osascript", "-e",
                'tell application "System Events" to keystroke "q" using {control down, command down}'
            ])
            print("Used AppleScript to lock macOS screen.")
        except Exception as e:
            print(f"macOS lock via AppleScript failed: {e}")

    elif os_name == 'Linux':
        # Try a few different Linux approaches
        # 1) Attempt xdg-screensaver
        ret = subprocess.call("xdg-screensaver lock", shell=True)
        if ret == 0:
            print("Used xdg-screensaver to lock screen.")
        else:
            # 2) Attempt gnome-screensaver-command
            ret = subprocess.call("gnome-screensaver-command -l", shell=True)
            if ret == 0:
                print("Used gnome-screensaver-command to lock screen.")
            else:
                # 3) Attempt vlock (console lock)
                print("Falling back to 'vlock' console lock. Press ESC in the console to exit if unlocked.")
                subprocess.call("vlock", shell=True)
    else:
        print(f"Unsupported OS for locking: {os_name}")

#############################################
# MEDIAPIPE FACE MESH SETUP                 #
#############################################
mp_face_mesh = mp.solutions.face_mesh

# Lowered detection confidences for more lenient face tracking
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.3,  # was 0.5
    min_tracking_confidence=0.3    # was 0.5
)

mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

#############################################
# LANDMARK INDEXES (MEDIAPIPE Face Mesh)    #
#############################################
NOSE_TIP = 4
CHIN = 152
LEFT_EYE_CORNER = 33     # left eye outer corner
RIGHT_EYE_CORNER = 263   # right eye outer corner
LEFT_MOUTH_CORNER = 61
RIGHT_MOUTH_CORNER = 291

#############################################
# 3D MODEL POINTS (Approximate)             #
#############################################
model_points_3D = np.array([
    (0.0, 0.0, 0.0),         # Nose tip
    (0.0, -330.0, -65.0),    # Chin
    (-220.0, 170.0, -135.0), # Left eye corner (outer)
    (220.0, 170.0, -135.0),  # Right eye corner (outer)
    (-150.0, -150.0, -125.0),# Left mouth corner
    (150.0, -150.0, -125.0)  # Right mouth corner
], dtype=np.float64)

#############################################
# HEAD POSE ESTIMATION WITH solvePnP        #
#############################################
def get_head_pose(face_landmarks, image_width, image_height, camera_matrix, dist_coeffs):
    """
    Returns (pitch, yaw, roll) in degrees based on solvePnP.
    If it cannot compute (e.g., some landmarks missing), returns None.
    """
    try:
        landmarks_2d = [
            face_landmarks.landmark[NOSE_TIP],
            face_landmarks.landmark[CHIN],
            face_landmarks.landmark[LEFT_EYE_CORNER],
            face_landmarks.landmark[RIGHT_EYE_CORNER],
            face_landmarks.landmark[LEFT_MOUTH_CORNER],
            face_landmarks.landmark[RIGHT_MOUTH_CORNER],
        ]
    except IndexError:
        return None

    # Convert from normalized [0..1] to absolute pixel coords
    image_points_2D = np.array([
        (landmarks_2d[0].x * image_width, landmarks_2d[0].y * image_height),  
        (landmarks_2d[1].x * image_width, landmarks_2d[1].y * image_height),  
        (landmarks_2d[2].x * image_width, landmarks_2d[2].y * image_height),  
        (landmarks_2d[3].x * image_width, landmarks_2d[3].y * image_height),  
        (landmarks_2d[4].x * image_width, landmarks_2d[4].y * image_height),  
        (landmarks_2d[5].x * image_width, landmarks_2d[5].y * image_height),  
    ], dtype=np.float64)

    # Solve for pose:
    success, rotation_vec, translation_vec = cv2.solvePnP(
        model_points_3D, 
        image_points_2D,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return None

    # Convert rotation vector to rotation matrix
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)

    # Extract pitch, yaw, roll using cv2.RQDecomp3x3
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_mat)
    pitch, yaw, roll = angles[0], angles[1], angles[2]

    return (pitch, yaw, roll)

#############################################
# SMOOTHED "IS HEAD AWAY?" FUNCTION         #
#############################################
def is_head_away_smoothed(pose_buffer, yaw_thresh=60, pitch_thresh=50, roll_thresh=50):
    """
    Currently, we do NOT use this to lock the screen,
    but we can still log or debug if the user's head is away.
    """
    if len(pose_buffer) < pose_buffer.maxlen:
        return False

    avg_pitch = sum(p[0] for p in pose_buffer) / len(pose_buffer)
    avg_yaw   = sum(p[1] for p in pose_buffer) / len(pose_buffer)
    avg_roll  = sum(p[2] for p in pose_buffer) / len(pose_buffer)

    # Debug logging to see average angles:
    print(f"[DEBUG] Averages -> pitch={avg_pitch:.2f}, yaw={avg_yaw:.2f}, roll={avg_roll:.2f}")

    if abs(avg_yaw) > yaw_thresh:
        return True
    if abs(avg_pitch) > pitch_thresh:
        return True
    if abs(avg_roll) > roll_thresh:
        return True

    return False

#############################################
# MAIN LOOP                                 #
#############################################
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        sys.exit(1)

    # ------------------------------------------
    # SETTINGS & BUFFERS
    # ------------------------------------------
    frames_without_face = 0
    max_frames_without_face = 1000  # Only lock after 1000 consecutive frames with no face

    ANGLE_BUFFER_SIZE = 20
    pose_buffer = deque(maxlen=ANGLE_BUFFER_SIZE)

    while True:
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image_height, image_width = frame.shape[:2]

        # Approximate camera matrix
        focal_length = image_width
        center = (image_width / 2, image_height / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)

        dist_coeffs = np.zeros((4, 1))  # Zero distortion

        # Convert to RGB for Mediapipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            # We see a face
            frames_without_face = 0

            face_landmarks = results.multi_face_landmarks[0]
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec
            )

            # Get pose, but we do NOT lock if user is "away"
            pose = get_head_pose(face_landmarks, image_width, image_height, camera_matrix, dist_coeffs)
            if pose is not None:
                pitch, yaw, roll = pose
                pose_buffer.append((pitch, yaw, roll))

                # We can still see if head is away for debugging:
                head_away = is_head_away_smoothed(pose_buffer)
                if head_away:
                    # Just log, do NOT lock
                    print(f"[LOG] Head appears away (pitch={pitch:.2f}, yaw={yaw:.2f}, roll={roll:.2f}).")
                
                # Show pitch, yaw, roll on-screen
                cv2.putText(
                    frame,
                    f"pitch={pitch:.2f}, yaw={yaw:.2f}, roll={roll:.2f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )
        else:
            # No face detected in this frame
            frames_without_face += 1
            print("No face detected this frame.")

            # If too many consecutive frames have no face, lock the screen
            if frames_without_face >= max_frames_without_face:
                print("No face detected for multiple frames, locking screen...")
                lock_screen()
                frames_without_face = 0

        cv2.imshow("Gaze Detection POC (solvePnP)", frame)
        if cv2.waitKey(5) & 0xFF == 27:
            print("Exiting...")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
