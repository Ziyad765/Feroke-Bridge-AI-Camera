from datetime import datetime
from enum import Enum

class SignalState(Enum):
    ALL_RED = "ALL_RED"
    SIDE_A_GREEN = "SIDE_A_GREEN"
    SIDE_B_GREEN = "SIDE_B_GREEN"
    EMERGENCY = "EMERGENCY"

class SignalController:
    def __init__(self, min_green=10, max_green=30, max_clearance=15, min_clearance=5):
        self.min_green = min_green
        self.max_green = max_green
        self.max_clearance = max_clearance
        self.min_clearance = min_clearance  # Minimum seconds in ALL_RED even if bridge is empty
        
        self.state = SignalState.ALL_RED
        self.last_switch = datetime.now()
        self.emergency = False
        
        # 4-camera system
        self.queue_a = 0
        self.queue_b = 0
        self.interior_a = 0
        self.interior_b = 0

        self.last_green_side = None
        self.history = [] 

        # Auto-Alarm system
        self.block_start = None
        self.auto_alarm_enabled = False

    def update_config(self, min_green, max_green, max_clearance, min_clearance=5):
        self.min_green = min_green
        self.max_green = max_green
        self.max_clearance = max_clearance
        self.min_clearance = min_clearance
        self._log(f"Config: Min={min_green}, Max={max_green}, Clear={max_clearance}, MinClear={min_clearance}")

    def update_counts(self, q_a, q_b, i_a, i_b):
        self.queue_a = q_a
        self.queue_b = q_b
        self.interior_a = i_a
        self.interior_b = i_b

        if self.auto_alarm_enabled:
            # Check for permanent bridge block > 45s
            if i_a > 0 or i_b > 0:
                if self.block_start is None:
                    self.block_start = datetime.now()
                elif (datetime.now() - self.block_start).total_seconds() >= 45:
                    if not self.emergency:
                        self.set_emergency(True)
                        self._log("EMERGENCY AUTO-BRAKE: Bridge Blocked >45s")
            else:
                self.block_start = None
                if self.emergency:
                    self.set_emergency(False)
                    self._log("EMERGENCY LIFTED: Bridge Cleared")

    def set_emergency(self, active: bool):
        self.emergency = active
        if active:
            self.state = SignalState.EMERGENCY
            self.last_switch = datetime.now()
            self._log("EMERGENCY ACTIVATED")

    def get_status(self):
        return {
            "state": self.state.value,
            "counts": [self.queue_a, self.queue_b, self.interior_a, self.interior_b],
            "elapsed": (datetime.now() - self.last_switch).total_seconds(),
            "history": self.history[-60:] # Store more history for UI log integration
        }
    
    def _log(self, message):
        t = datetime.now().strftime("%H:%M:%S")
        self.history.append({"time": t, "event": message})
        print(f"[Signal] {t} - {message}")

    def run_logic(self):
        if self.emergency: return

        elapsed = (datetime.now() - self.last_switch).total_seconds()

        # ── ALL_RED (BRIDGE CLEARANCE PHASE) ──
        if self.state == SignalState.ALL_RED:
            # Must always wait at least min_clearance seconds before switching,
            # even if the bridge clears instantly.
            if elapsed < self.min_clearance:
                return

            is_empty = (self.interior_a == 0 and self.interior_b == 0)

            # Switch only if bridge is empty OR we've hit the max clearance timeout
            if is_empty or elapsed >= self.max_clearance:
                # Decide which side to green
                if self.queue_a == 0 and self.queue_b == 0:
                    return # Stay red if no cars waiting

                if self.queue_a > self.queue_b:
                    self._switch(SignalState.SIDE_A_GREEN)
                elif self.queue_b > self.queue_a:
                    self._switch(SignalState.SIDE_B_GREEN)
                else:
                    # Alternating
                    if self.last_green_side == SignalState.SIDE_A_GREEN:
                        self._switch(SignalState.SIDE_B_GREEN)
                    else:
                        self._switch(SignalState.SIDE_A_GREEN)

        # ── GREEN PHASES ──
        elif self.state == SignalState.SIDE_A_GREEN:
            if elapsed < self.min_green: return
            
            should_switch = False
            if elapsed >= self.max_green: should_switch = True
            elif self.queue_a == 0 and self.queue_b > 0: should_switch = True
            
            if should_switch:
                self.last_green_side = SignalState.SIDE_A_GREEN
                self._switch(SignalState.ALL_RED)

        elif self.state == SignalState.SIDE_B_GREEN:
            if elapsed < self.min_green: return
            
            should_switch = False
            if elapsed >= self.max_green: should_switch = True
            elif self.queue_b == 0 and self.queue_a > 0: should_switch = True
            
            if should_switch:
                self.last_green_side = SignalState.SIDE_B_GREEN
                self._switch(SignalState.ALL_RED)

    def _switch(self, new_state):
        prev = self.state.value
        curr = new_state.value
        self._log(f"Switch: {prev} -> {curr}")
        self.state = new_state
        self.last_switch = datetime.now()
