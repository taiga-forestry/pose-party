import cv2
from pose_estimation import get_and_save_joints, initialize_landmarker

from game import DuelGameState

def write_text(game_state: DuelGameState, text, y=50, centered=False):
    frame = game_state.last_frame
    x = 450 if centered else (30 if game_state.curr_player is game_state.player_1 else 800)
    
    cv2.putText(
        img=frame,
        text=text,
        org=(x, y),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1,
        color=(255, 255, 255),
        thickness=2)
    
def show_countdown_timer(game_state: DuelGameState):
    # FIXME: better x, y values
    write_text(game_state, f"Current Player: {game_state.curr_player.name}", 50)
    write_text(game_state, f"Time Remaining: {game_state.curr_action.time_remaining() // 1000}s", 100)

def take_screenshot(game_state: DuelGameState):
    frame = game_state.last_frame
    h, w, _ = frame.shape
    left = frame[:, :w//2]
    right = frame[:, w//2:]

    if game_state.curr_player == game_state.player_1:
        game_state.player_1.screenshot = left
        game_state.player_1.saved_joints = [joints for joints in game_state.player_joints.values() if joints[0][0] < 0.5][0]
    else:
        game_state.player_2.screenshot = right
        game_state.player_2.saved_joints = [joints for joints in game_state.player_joints.values() if joints[0][0] > 0.5][0]

    save_screenshot(game_state, frame)
    # get_and_save_joints(initialize_landmarker(), frame, game_state)

def save_screenshot(game_state: DuelGameState, frame):
    confidence_threshold = 0.8
    cv2.imwrite(f"screenshot-{game_state.curr_player.name}-{game_state.round}-{game_state.t}.png", frame)

    with open(f"joints-{game_state.curr_player.name}-{game_state.round}-{game_state.t}.txt", "w") as f:
        for i, (x, y, z, visibility) in enumerate(game_state.curr_player.saved_joints):
            if visibility > confidence_threshold: 
                f.write(f"{i}: ({x}, {y}, {z} | viz: {visibility})\n")
            else:
                f.write(f"{i}: --- | viz: {visibility}\n")

    game_state.t += 1

def center_text_x(frame, text):
   text_size = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=2)[0]
   frame_center_x = frame.shape[1] // 2
   text_x = frame_center_x - text_size[0] // 2
   return text_x
