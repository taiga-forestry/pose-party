import cv2
import mediapipe as mp

# Initialize pose and hand solutions
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

pose = mp_pose.Pose(min_tracking_confidence=0.3)

SELECTED_JOINTS = set([
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
    frame.flags.writeable = False
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pose_results = pose.process(frame_rgb)
    frame.flags.writeable = True

    if pose_results.pose_landmarks:
        mp_draw.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    h, w, _ = frame.shape
    out = {}

    # Extract selected pose joints
    if pose_results.pose_landmarks:
        for i, landmark in enumerate(pose_results.pose_landmarks.landmark):
            if i not in SELECTED_JOINTS:
                continue
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            visibility = landmark.visibility
            out[f"pose_{i}"] = {"x": x, "y": y, "visibility": visibility}

    # # Draw and extract hands
    # if hand_results.multi_hand_landmarks:
    #     for hand_idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
    #         mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    #         for i, landmark in enumerate(hand_landmarks.landmark):
    #             x = int(landmark.x * w)
    #             y = int(landmark.y * h)
    #             out[f"hand{hand_idx}_{i}"] = {"x": x, "y": y}

    return out
