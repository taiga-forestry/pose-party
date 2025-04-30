from action import TimedAction
from typing import Union
from player import Player
from dataclasses import dataclass, field
from cv2.typing import MatLike

NUM_ROUNDS = 1

# NOTE: player_1 will always be left side of screen, and player_2 right side
@dataclass
class DuelGameState:
    player_1: Player
    player_2: Player
    lead_player: Player = field(init=False)
    follow_player: Player = field(init=False)
    curr_player: Player = field(init=False)
    started: bool = False
    round: int = 1
    curr_action: Union[TimedAction, None] = None
    # history: list[list[]]
    last_frame: Union[MatLike, None] = None

    def __post_init__(self):
        self.lead_player = self.player_1
        self.follow_player = self.player_2
        self.curr_player = self.player_1

    def is_game_ended(self):
        return self.round > NUM_ROUNDS
  
    def toggle_curr_player(self):
        self.curr_player = self.follow_player if self.curr_player is self.lead_player else self.lead_player
    
    def swap_players(self):
        print("CHANGING TURNS")
        self.lead_player, self.follow_player = self.follow_player, self.lead_player
        self.curr_player = self.lead_player
        if self.lead_player is self.player_1:
            self.round += 1
