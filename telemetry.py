from streamer import Streamer
import time
import cv2
import random
import math

stream = Streamer()

pos = [0, 0, random.uniform(0, 2 * math.pi)]  # x, y, theta

# display an image
stream.set_img(cv2.imread("test.jpg"))

# continuously update data and odometry
curr_data = {}
while True:
    pos[0] += math.cos(pos[2]) * 0.01
    pos[1] += math.sin(pos[2]) * 0.01
    pos[2] += random.uniform(-0.2, 0.2)
    curr_data["timestamp"] = time.time()
    curr_data["random_value"] = random.random()

    # stream data and odometry
    stream.set_data(curr_data)
    stream.update_odom_state(pos[0], pos[1], pos[2])
    stream.update_circles(
        [
            (0.40, 0.20, "red"),
            (0.13, -0.40, "green"),
        ]
    )
    stream.update_lines(
        [
            (0, 0, 0.75, 0.75, "blue"),
        ]
    )
    time.sleep(0.05)
