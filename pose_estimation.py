import cv2
import mediapipe as mp

# min_detection_confidence=0.7, min_tracking_confidence=0.7
pose = mp.solutions.pose.Pose(min_tracking_confidence=0.3)
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

SELECTED_JOINTS = set([
    0,       # nose
    11, 12,  # shoulders
    13, 14,  # elbows
    15, 16,  # wrists
    23, 24,  # hips
    25, 26,  # knees
    28, 29,  # ankles
    29, 30,  # heels
    31, 32,  # feet
])

def get_and_draw_joints(frame):
    # to improve performance, mark the image as not writeable to pass by reference
    frame.flags.writeable = False
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)
    frame.flags.writeable = True

    if not results.pose_landmarks:
        return None
    
    mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    h, w, _ = frame.shape
    out = {}

    for i, landmark in enumerate(results.pose_landmarks.landmark):
        if i not in SELECTED_JOINTS:
            continue

        x = int(landmark.x * w)
        y = int(landmark.y * h)
        visibility = landmark.visibility

        out[i] = { "x": x, "y": y, "visibility": visibility }
        # print(f"Landmark {i}: x={cx}, y={cy}, z={landmark.z:.3f}, visibility={landmark.visibility:.2f}")
        # print(f"Landmark {i}: x={landmark.x }, y={landmark.y }, z={landmark.z:.3f}, visibility={landmark.visibility:.2f}")


        # Optional: Draw a circle at each landmark
        # cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
    
    return out

