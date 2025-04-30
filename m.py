import cv2
import time
import pose_estimation
from game import DuelGameState
from player import Player

print("===========================")
print("Welcome to Pose Party!")
name_1 = input("Player 1 Name: ")
name_2 = input("Player 2 Name: ")
player_1 = Player(id=1, name=name_1)
player_2 = Player(id=2, name=name_2)
players = [player_1, player_2]

game_state = DuelGameState(players=players)






# CONSTANTS
MAX_NUM_FRAMES = 5
COUNTDOWN_DURATION = 5

start_time = -1

players = ["Tiger", "Claire"]
saved_frames = [None, None]
curr_player = 0

cap = cv2.VideoCapture(0)
frames = []

screenshot = None


if not cap.isOpened():
    raise IOError("cannot access webcam :(")

while True:
    ret, frame = cap.read()

    if not ret:
        continue

    pose_estimation.get_and_draw_joints(frame)

    frames.append(frame)

    if len(frames) > MAX_NUM_FRAMES:
        frames.pop(0)

    elapsed_time = time.time() - start_time
    remaining = max(0, int(COUNTDOWN_DURATION - elapsed_time))


    h, w, _ = frame.shape
    left = frame[:, :w//2]
    right = frame[:, w//2:]

    cv2.putText(
        img=frame,
        text=f"Current Player: {players[curr_player]}",
        org=(30, 50),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1,
        color=(255, 255, 255),
        thickness=2)
    
    if start_time != -1:
        cv2.putText(
            img=frame,
            text=f"Countdown: {remaining}",
            org=(30, 100),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(255, 255, 255),
            thickness=2)
        
        if remaining == 0:
            cv2.putText(
                img=frame,
                text=f"Photo Taken!",
                org=(30, 150),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(255, 255, 255),
                thickness=2)

            saved_frames[curr_player] = frames # FIXME: average frames

            if screenshot is None:
                print("wat")
                screenshot = left.copy()
        
        # save avg frame somewhere - use to compare later!

    # If screenshot is taken, use it for the left half
    if screenshot is not None:
        print("SDlgijsodgjasdlnas")
        combined = cv2.hconcat([screenshot, right])
    else:
        combined = cv2.hconcat([left, right])
      
    cv2.imshow("Pose Split View", combined)
    # cv2.imshow("FIXME: come up with window name", frame)
    
    # options:
    # - 
    # - press Q to quit
    key = cv2.waitKey(1)

    if key & 0xFF == ord("t"):
        start_time = time.time()
    elif key & 0xFF == ord("y"):
        start_time = time.time()
        curr_player = (curr_player + 1) % 2
    elif key & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()