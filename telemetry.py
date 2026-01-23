from streamer import Streamer
import time
import cv2
import random
import math

stream = Streamer()

pos = [0, 0, random.uniform(0, 360)]  # x, y, theta

# display an image
stream.set_img(cv2.imread("test.jpg"))

# continuously update data and odometry
curr_data = {}
while True:
    pos[0] += math.cos(math.radians(pos[2])) * 0.2
    pos[1] += math.sin(math.radians(pos[2])) * 0.2
    pos[2] += random.uniform(-5, 5)
    odom = {
        "x": pos[0],
        "y": pos[1],
        "theta": pos[2],
        "circles": [
            {"x": 40, "y": 20, "c": "red"},
            {"x": 130, "y": -40, "c": "green"},
            {"x": 40, "y": 100, "c": "blue"},
        ],
        "lines": [
            {"x1": 0, "y1": 0, "x2": 100, "y2": 0, "c": "red"},
        ],
    }
    curr_data["timestamp"] = time.time()
    curr_data["random_value"] = random.random()
    curr_data["odometry"] = odom

    # stream data and odometry
    stream.set_data(curr_data)
    stream.set_odometry(odom)

    time.sleep(0.05)


class Telemetry():
    def __init__(self):
        self.stream = Streamer()
        pass
