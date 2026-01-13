import time
import math
import threading
from nav import Nav
from typing import Tuple
from connection import config
from connection.ComputerReceiver import ComputerReceiver


class MovementCommander:
    def __init__(self, computerReceiver: ComputerReceiver):
        self.running = True

        self.planned_moves: list[Tuple[float, float, float, float]] = []
        self._lock = threading.Lock()

        self.computerReceiver = computerReceiver
        self.nav = Nav()

        threading.Thread(target=self._commandLoop, daemon=True).start()

    def queue_xy(self, x, y):
        """
        take in x,y in mm and plan send out instructions
        """
        distance = math.sqrt(x * x + y * y)

        # forward is y axis, so we want angle from y axis
        # while atan calculates angle from x axis
        theta = math.atan2(y, x) - math.pi / 2

        rotate = (time.time(), *self.nav.get_rotate(theta))
        move = (time.time() + 1, *self.nav.get_forward_mm(distance))

        print(
            f'sent movement x={x} y={y} theta={theta} distance={distance} rotate={rotate} move={move}')

        with self._lock:
            self.planned_moves = [rotate, move]

    def _commandLoop(self):
        while self.running:
            movement = []
            with self._lock:
                if self.planned_moves:
                    # get the earliest planned move
                    plan = self.planned_moves[0]
                    # check if the first plan is ready to be executed
                    if (time.time() >= plan[0]):
                        self.planned_moves = self.planned_moves[1:]
                        movement = plan[1:]
            if movement:
                self.computerReceiver.send_movement(*movement)

            time.sleep(1 / config.DEFAULT_MAX_FPS)

    def stop(self):
        self.running = False
