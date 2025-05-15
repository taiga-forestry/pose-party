import time
from typing import Callable
from dataclasses import dataclass

@dataclass
class TimedAction:
    duration: int = 4000
    pending_action: Callable[[], None] = lambda: None
    main_action: Callable[[], None] = lambda: None
    start_time: int = -1 # Declare when action begins
    show_line: bool = False # Indicates if divider should be shown

    def countdown_complete(self):
        if self.start_time == -1: 
            raise ValueError("timer not started")
        
        current_time = int(time.time() * 1000)
        return current_time >= self.start_time + self.duration
    
    def time_remaining(self):
        if self.start_time == -1: 
            raise ValueError("timer not started")

        current_time = int(time.time() * 1000)
        elapsed_time = current_time - self.start_time
        return max(0, int(self.duration - elapsed_time))
    
    def start_timer(self):
        self.start_time = int(time.time() * 1000)