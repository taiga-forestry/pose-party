import cv2
from pose_estimation import initialize_landmarker, get_and_draw_joints
from action import TimedAction
from game import DuelGameState
from player import Player
from util import write_text, show_countdown_timer, take_screenshot

print("===========================")
print("Welcome to Pose Party!")
name_1 = "TIGER" #input("Player 1 Name: ")
name_2 = "CLAIRE" #input("Player 2 Name: ")
player_1 = Player(id=1, name=name_1)
player_2 = Player(id=2, name=name_2)

game_state = DuelGameState(player_1=player_1, player_2=player_2)
actions = [
    TimedAction(
        pending_action=lambda: write_text(game_state.last_frame, f"Player {game_state.curr_player.id}, Get Ready...", 30, 50)
    ),
    TimedAction(
        pending_action=lambda: show_countdown_timer(game_state),
        main_action=lambda: take_screenshot(game_state)
    ),
    TimedAction(
        pending_action=lambda: write_text(game_state.last_frame, f"Screenshot saved!", 30, 50),
        main_action=lambda: game_state.toggle_curr_player()
    ),
    TimedAction(
        pending_action=lambda: write_text(game_state.last_frame, f"Player {game_state.curr_player.id}, Get Ready...", 800, 50)
    ),
    TimedAction(
        pending_action=lambda: show_countdown_timer(game_state),
        main_action=lambda: take_screenshot(game_state)
    ),
    TimedAction(
        pending_action=lambda: write_text(game_state.last_frame, f"Screenshot saved!", 800, 50),
        main_action=lambda: game_state.toggle_curr_player()
    ),
]
  
with initialize_landmarker() as landmarker:
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            raise ValueError("video capture failed :(")
        
        frame = cv2.flip(frame, 1) # flip everything to be mirrored!
        game_state.last_frame = frame
        get_and_draw_joints(landmarker, frame)
        
        # display loading screen / ending screen
        if not game_state.started:
            pass
        elif game_state.is_game_ended():
            pass

        if game_state.curr_action is None:
            pass
        elif game_state.curr_action.countdown_complete():
            game_state.curr_action.main_action()

            # if any more actions remaining, take next one and start it
            # else, round is over, increment
            if actions:
                game_state.curr_action = actions.pop(0)
                game_state.curr_action.start_timer()
            else:
                game_state.round += 1
        else:
            game_state.curr_action.pending_action()

        h, w, _ = frame.shape
        left = frame[:, :w//2]
        right = frame[:, w//2:]
        display = cv2.hconcat([
            player_1.screenshot if player_1.screenshot is not None else left,
            player_2.screenshot if player_2.screenshot is not None else right,
        ])
        cv2.imshow("Pose Party Screen", display)

        # AVAILABLE KEYBOARD INTERACTIONS:
        # - press S to start the game
        # - press Q to quit the game
        key = cv2.waitKey(1)

        if key & 0xFF == ord("s"):
            game_state.curr_action = actions.pop(0)
            game_state.curr_action.start_timer()
        elif key & 0xFF == ord("q"):
            break





