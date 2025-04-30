import cv2
import time
import pose_estimation
import pose_matching
import numpy as np

# CONSTANTS
COUNTDOWN_DURATION = 5
players = ["Tiger", "Claire"]
saved_poses = [None, None]
curr_player = 0
start_time = -1
counting_down = False

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot access webcam.")

frame_height, frame_width = None, None

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    if frame_height is None:
        frame_height, frame_width = frame.shape[:2]

    # Split frame for dual view
    h, w, _ = frame.shape
    left = frame[:, :w // 2]
    right = frame[:, w // 2:]

    # Draw joints on live view
    pose_estimation.get_and_draw_joints(frame)

    # Timer and capture logic
    if counting_down:
        elapsed = time.time() - start_time
        remaining = max(0, int(COUNTDOWN_DURATION - elapsed))

        cv2.putText(frame, f"Countdown: {remaining}",
                    (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        if remaining <= 0:
            # Capture pose once countdown completes
            joints = pose_estimation.get_and_draw_joints(frame)
            saved_poses[curr_player] = joints
            print(f"Captured for {players[curr_player]}: {list(joints.keys()) if joints else 'None'}")
            counting_down = False

            # Automatically switch player
            curr_player = (curr_player + 1) % 2

    # Show current player
    cv2.putText(frame, f"Current Player: {players[curr_player]}",
                (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Once both poses are captured, compare them
    if all(p is not None for p in saved_poses):
        score = pose_matching.cosine_distance(saved_poses[0], saved_poses[1])
        print(f"\nPose similarity score: {score:.4f}")
        saved_poses = [None, None]  # Reset for next round

    # Compose combined view
    if saved_poses[0] is not None:
        # Ensure same size and type before concat
        left_resized = cv2.resize(left, (right.shape[1], right.shape[0]))
        combined = cv2.hconcat([left_resized, right])
    else:
        combined = cv2.hconcat([left, right])

    cv2.imshow("Pose Game", combined)

    key = cv2.waitKey(1)
    if key & 0xFF == ord("t"):
        start_time = time.time()
        counting_down = True
    elif key & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
