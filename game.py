from player import Player
from dataclasses import dataclass

NUM_ROUNDS = 3

@dataclass
class DuelGameState:
  players: list[Player]
  lead_player: int = 0    # LEFT
  follow_player: int = 1  # RIGHT
  started: bool = False
  round: int = 1
  # history: list[list[]]
