import cv2
from pose_estimation import initialize_landmarker, get_and_draw_joints
from pose_matching import cosine_distance, distance_to_percentage
from action import TimedAction
from game import DuelGameState
from player import Player
from util import center_text_x, write_text, show_countdown_timer, take_screenshot
from collections import defaultdict

print("===========================")
print("Welcome to Pose Party!")
name_1 = "TIGER" #input("Player 1 Name: ")
name_2 = "CLAIRE" #input("Player 2 Name: ")
player_1 = Player(id=1, name=name_1)
player_2 = Player(id=2, name=name_2)

game_state = DuelGameState(player_1=player_1, player_2=player_2)

def get_round_actions():
    return [
        TimedAction(
            pending_action=lambda: write_text(game_state, f"Player {game_state.curr_player.id}, Get Ready..."),
            show_line = True
        ),
        TimedAction(
            # TODO the countdown should start from 3s not 2s so maybe add a second
            pending_action=lambda: show_countdown_timer(game_state),
            main_action=lambda: take_screenshot(game_state),
            show_line = True
        ),
        TimedAction(
            pending_action=lambda: write_text(game_state, f"Screenshot saved!"),
            main_action=lambda: game_state.toggle_curr_player(),
            show_line = True
        ),
        TimedAction(
            pending_action=lambda: write_text(game_state, f"Player {game_state.curr_player.id}, Get Ready..."),
            show_line = True
        ),
        TimedAction(
            pending_action=lambda: show_countdown_timer(game_state),
            main_action=lambda: take_screenshot(game_state),
            show_line = True
        ),
        TimedAction(
            pending_action=lambda: write_text(game_state, f"Screenshot saved!"),
            main_action=lambda: reset_screenshots(),
            show_line = True
            # TODO claire - change this to either let have the text write over the screenshots or merge the screenshots
            # but for now we just reset the screenshots so they aren't in the way
        ),
        TimedAction(
            pending_action=lambda: write_text(game_state, "Accuracy = 85%!", y=300, centered=True), # TODO change to actual accuracy
            main_action=lambda: game_state.swap_players(),
            show_line = False
        ),
        TimedAction(
            pending_action=lambda: write_text(game_state, f"Now it's {game_state.curr_player.name}'s turn!", y=300, centered=True),
            main_action=lambda: reset_for_next_turn(),
            show_line = False
        ),
    ]
  
actions = get_round_actions()

def reset_screenshots():
    player_1.screenshot = None
    player_2.screenshot = None

# Helper function to reset screenshots
# TODO: Save pics from rounds somewhere to display in recap
def reset_for_next_turn():
    reset_screenshots()
    if not game_state.is_game_ended():
        global actions
        actions = get_round_actions()
    print(game_state.saved_frame)
    score = distance_to_percentage(cosine_distance(game_state.saved_frame[0], game_state.saved_frame[1]))
    print(score)
    game_state.saved_frame = [defaultdict(dict), defaultdict(dict)]

with initialize_landmarker() as landmarker:
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, frame = cap.read()
        h, w, _ = frame.shape

        if not success:
            raise ValueError("video capture failed :(")
        
        frame = cv2.flip(frame, 1) # flip everything to be mirrored!
        #game_state.last_frame = frame
        get_and_draw_joints(landmarker, frame, game_state)


        # display loading screen / ending screen
        if not game_state.started:
            write_text(game_state, f"Press 's' to start!", center_text_x(frame, "Press 's' to start!"), 150)
        elif game_state.is_game_ended():
            game_state.curr_action = None
            write_text(game_state, f"GAME OVER! {game_state.curr_player.id} WINS", y=50, centered=True) # TODO change to actual winner
        else:
           if game_state.curr_action.show_line:
                color = (255, 0, 0) #BGR
                thickness = 9
                divider = cv2.line(game_state.last_frame, (w//2, 0), (w//2, h), color, thickness)


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
                write_text(game_state, f"GAME OVER! {game_state.curr_player.id} WINS", y=50, centered=True) # TODO change to actual winner
                game_state.round += 1

        else:
            game_state.curr_action.pending_action()

        #h, w, _ = frame.shape
        left = frame[:, :w//2]
        right = frame[:, w//2:]
        display = cv2.hconcat([
            player_1.screenshot if player_1.screenshot is not None else left,
            player_2.screenshot if player_2.screenshot is not None else right,
        ])
        
        if not game_state.paused:
            game_state.last_frame = frame
            cv2.imshow("Pose Party Screen", display)
        else:
            paused_display = game_state.last_frame.copy()
            write_text(game_state, f"GAME PAUSED", center_text_x(paused_display, "GAME PAUSED"), 150, frame=paused_display)
            cv2.imshow("Pose Party Screen", paused_display)


        # AVAILABLE KEYBOARD INTERACTIONS:
        # - press S to start the game
        # - press Q to quit the game
        key = cv2.waitKey(1)

        if key & 0xFF == ord("s"):
            game_state.curr_action = actions.pop(0)
            game_state.curr_action.start_timer()
            game_state.started = True
        elif key & 0xFF == ord("r"):

            # go back to starting screen
            if game_state.started:

                cv2.waitKey(0)
                print("reset")
                game_state = DuelGameState(player_1=player_1, player_2=player_2)
                actions = get_round_actions()
                game_state.started = False
                write_text(game_state, f"Press 's' to start!", center_text_x(frame, "Press 's' to start!"), 150)
        

            # should we do a check for if they actually want to reset? like a yes/no
        elif key & 0xFF == ord("p"):
            game_state.paused = not game_state.paused

            print("game paused: ", game_state.paused)
            if game_state.paused:
                print("making copy of display")
                game_state.last_frame = display.copy()
            
            
        # should there be a pause feature? p for pause/unpause
        # pause field
        elif key & 0xFF == ord("q"):
            break




