import cv2

from game import DuelGameState

def write_text(frame, text, x, y):
    cv2.putText(
        img=frame,
        text=text,
        org=(x, y),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1,
        color=(255, 255, 255),
        thickness=2)
    
def show_countdown_timer(game_state: DuelGameState):
    frame = game_state.last_frame
    
    # FIXME: better x, y values
    x = 30 if game_state.curr_player is game_state.player_1 else 800
    write_text(frame, f"Current Player: {game_state.curr_player.name}", x, 50)
    write_text(frame, f"Time Remaining: {game_state.curr_action.time_remaining() // 1000}s", x, 100)

def take_screenshot(game_state: DuelGameState):
    frame = game_state.last_frame
    h, w, _ = frame.shape
    left = frame[:, :w//2]
    right = frame[:, w//2:]

    if game_state.curr_player == game_state.player_1:
        game_state.player_1.screenshot = left
    else:
        game_state.player_2.screenshot = right

def center_text_x(frame, text):
   text_size = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=2)[0]
   frame_center_x = frame.shape[1] // 2
   text_x = frame_center_x - text_size[0] // 2
   return text_x