import math


robot_x, robot_y = 0, 0

# Get claw offset in robot frame
claw_offset = 5.00  # mm

angle = 30

cos_angle = math.cos(angle)
sin_angle = math.sin(angle)

x_world = robot_x + claw_offset * cos_angle
y_world = robot_y + claw_offset * sin_angle

print(x_world, y_world)
