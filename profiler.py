import cProfile
import os
from datetime import datetime
from typing import Dict, List
import time

# Get absolute path to profiles directory (relative to this script)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROFILE_DIR = os.path.join(SCRIPT_DIR, "profiles")


class Profiler:
    def __init__(self, verbose: bool = True):
        self.timings: Dict[str, List[float]] = {}
        self.current_frame = 0
        self.profile = cProfile.Profile()
        self._profile_enabled = False
        self.verbose = verbose

    def start_frame(self):
        if not self._profile_enabled:
            self.profile.enable()
            self._profile_enabled = True
        self.frame_start = time.time()

    def record(self, name: str):
        if name not in self.timings:
            self.timings[name] = []
        self.timings[name].append(
            (time.time() - self.frame_start) * 1000)  # ms
        self.frame_start = time.time()

    def end_frame(self):
        self.current_frame += 1
        if self.verbose and self.current_frame % 10 == 0:  # Print stats every 10 frames
            self.print_stats()

    def print_stats(self):
        print("\n=== Performance Metrics (ms) ===")
        for name, times in self.timings.items():
            if times:
                print(
                    f"{name}: {sum(times[-10:]) / len(times[-10:]):.2f}ms (last 10 avg)")
        print("=" * 30)

    def save_profile(self):
        os.makedirs(PROFILE_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{PROFILE_DIR}/profile_{timestamp}.prof"
        self.profile.disable()
        self.profile.dump_stats(filename)
        print(f"\nProfile saved to {filename}")
