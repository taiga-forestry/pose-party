from cv2.typing import MatLike
from dataclasses import dataclass
from typing import Union

@dataclass
class Player:
  id: int
  name: str
  score: float = 0.0
  screenshot: Union[MatLike, None] = None
  saved_joints = None
  