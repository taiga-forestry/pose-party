import cv2
from pose_estimation import initialize_landmarker, get_and_draw_joints
# from pose_matching import cosine_distance, distance_to_percentage
from pose_matching import calculate_similarity
from action import TimedAction
from game import DuelGameState
from player import Player
from util import center_text_x, write_text, show_countdown_timer, take_screenshot
from collections import defaultdict

print("===========================")
print("Welcome to Pose Party!")
name_1 = input("Player 1 Name: ")
name_2 = input("Player 2 Name: ")
player_1 = Player(id=0, name=name_1)
player_2 = Player(id=1, name=name_2)
won = False

game_state = DuelGameState(player_1=player_1, player_2=player_2)

def get_round_actions():
    return [
        TimedAction(
            pending_action=lambda: display_player_message()
        ),
        TimedAction(
            # TODO the countdown should start from 3s not 2s so maybe add a second
            pending_action=lambda: show_countdown_timer(game_state),
            main_action=lambda: take_screenshot(game_state),
        ),
        TimedAction(
            pending_action=lambda: write_text(game_state, f"Screenshot saved!"),
            main_action=lambda: game_state.toggle_curr_player()
        ),
        TimedAction(
            pending_action=lambda: display_player_message()
        ),
        TimedAction(
            pending_action=lambda: show_countdown_timer(game_state),
            main_action=lambda: take_screenshot(game_state)
        ),
        TimedAction(
            pending_action=lambda: write_text(game_state, f"Screenshot saved!"),
            main_action=lambda: save_results(game_state.player_1, game_state.player_2, True)
            # TODO claire - change this to either let have the text write over the screenshots or merge the screenshots
            # but for now we just reset the screenshots so they aren't in the way
        ),
        TimedAction(
            pending_action=lambda: write_text(game_state, f"Accuracy = {print_score():.3f}", y=300, centered=True), # TODO change to actual accuracy
            main_action=lambda: save_score_and_swap_players()
        ),
        TimedAction(
            pending_action=lambda: write_text(game_state, f"Now it's {game_state.curr_player.name}'s turn!", y=300, centered=True),
            main_action=lambda: reset_for_next_turn()
        ),
    ]
  
actions = get_round_actions()

def display_player_message():
    write_text(game_state, f"{game_state.curr_player.name.upper()}, Get Ready...")

def save_score_and_swap_players():
    # print(print_score(), game_state.curr_player.name)
    # game_state.curr_player.score += print_score()
    game_state.swap_players()
    # print("Now the curr player is: ", game_state.curr_player)
    # print("We are on round ", game_state.round)

def print_score():
    similarity = calculate_similarity(game_state)

    # with open(f"scores-{game_state.round}-{game_state.t}.txt", "w") as f:
    #     f.write(f"dist: {distance}")
    #     f.write(f"score: {score}")

    return similarity

def save_results(player1, player2, update_score):
    if player_1.screenshot is not None and player_2.screenshot is not None:
        left = player_1.screenshot
        right = player_2.screenshot
        frame = cv2.hconcat([left, right])
        cv2.imwrite(f"{game_state.round}-{game_state.t}-screenshot.png", frame)
        calculate_similarity(game_state, should_log=True)
        game_state.t += 1

    player_1.screenshot = None
    player_2.screenshot = None

    if update_score:
        game_state.curr_player.score += round(print_score(), 2)
    # if game_state.score[0] != 0 and game_state.score[1] != 0:
    #     print(game_state.score[0], game_state.score[1])
    #     if player1 is game_state.curr_player:
    #         game_state.score[0] += 1
    #     elif player2 is game_state.curr_player:
    #         game_state.score[1] += 1

# Helper function to reset screenshots
# TODO: Save pics from rounds somewhere to display in recap
def reset_for_next_turn():
    save_results(game_state.player_1, game_state.player_2, False)
    if not game_state.is_game_ended():
        global actions
        actions = get_round_actions()
    # print(game_state.saved_frame)
    # score = distance_to_percentage(cosine_distance(game_state.saved_frame[0], game_state.saved_frame[1]))
    # print(score)
    # game_state.saved_frame = [defaultdict(dict), defaultdict(dict)]
    player_1.screenshot = None
    player_2.screenshot = None
    game_state.player_1.saved_joints = None
    game_state.player_2.saved_joints = None


with initialize_landmarker() as landmarker:
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, frame = cap.read()
        h, w, _ = frame.shape

        if not success:
            raise ValueError("video capture failed :(")
        
        frame = cv2.flip(frame, 1) # flip everything to be mirrored!
        game_state.last_frame = frame
        player_joints = get_and_draw_joints(landmarker, frame, game_state)
        game_state.player_joints = player_joints
       
        # display loading screen / ending screen
        if not game_state.started:
            write_text(game_state, f"Press 's' to start!", center_text_x(frame, "Press 's' to start!"), y=150)
        elif game_state.is_game_ended():
            game_state.curr_action = None
            winner = game_state.player_1 if game_state.player_1.score > game_state.player_2.score else game_state.player_2
            write_text(game_state, f"GAME OVER! {winner.name} WINS", y=50, centered=True) # TODO change to actual winner
            # print(winner.name)
            won = True # TODO delete
            game_state.score = [0, 0]

        else:
            color = (255, 0, 0) #BGR
            thickness = 9
            divider = cv2.line(game_state.last_frame, (w//2, 0), (w//2, h), color, thickness)

        if game_state.curr_action is None:
            pass
        elif game_state.curr_action.countdown_complete():
            try:
                game_state.curr_action.main_action()

                # if any more actions remaining, take next one and start it
                # else, round is over, increment
                if actions:
                    game_state.curr_action = actions.pop(0)
                    game_state.curr_action.start_timer()
                else:
                    write_text(game_state, f"GAME OVER! {game_state.curr_player.id} WINS", y=50, centered=True) # TODO change to actual winner
            except Exception as e:
                print("Error: ", e)
        else:
            game_state.curr_action.pending_action()

        write_text(game_state, f"PLAYER 1 SCORE: " + f"{game_state.player_1.score:.2f}", x=30, y=700) 
        write_text(game_state, f"PLAYER 2 SCORE: " + f"{game_state.player_2.score:.2f}", x=850, y=700) 

        #h, w, _ = frame.shape
        left = frame[:, :w//2]
        right = frame[:, w//2:]
        display = cv2.hconcat([
            player_1.screenshot if player_1.screenshot is not None else left,
            player_2.screenshot if player_2.screenshot is not None else right,
        ])

        # add divider in middle of screen
        # if game_state.started:
        #    color = (255, 0, 0) #BGR
        #    thickness = 9
        
        cv2.imshow("Pose Party Screen", display)

        # AVAILABLE KEYBOARD INTERACTIONS:
        # - press S to start the game
        # - press Q to quit the game
        key = cv2.waitKey(1)

        if key & 0xFF == ord("s"):
            game_state.curr_action = actions.pop(0)
            game_state.curr_action.start_timer()
            game_state.started = True
        elif key & 0xFF == ord("q"):
            break
        





