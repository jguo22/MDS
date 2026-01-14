import math
BASE_D = 209

def move_to(dx, dy):
    # Make sure dx isn't 0
    if dx == 0:
        dx = 1
    # Calculate distance to travel in arc for center of bot
    c_r = (dx * dx + dy * dy) / (2 * abs(dx))
    print(f'c_r: {c_r}')
    c_theta = math.atan(dy/(c_r-dx)) if dx != dy else math.pi / 2
    print(f'c_theta: {c_theta}')
    c_d = c_r * c_theta

    print(f'distance: {c_d}')

    # Calculate l_c
    l_dx = dx + (BASE_D / 2)
    l_dy = dy + (BASE_D / 2)

    l_r = (l_dx * l_dx + l_dy * l_dy) / (2 * abs(l_dx))
    print(f'l_r: {l_r}')
    l_theta = math.atan(l_dy/(l_r-l_dx)) if l_dx != l_dy else math.pi / 2
    l_d = l_r * l_theta
    print(f'l_d: {l_d}')
    l_c = l_d / c_d
    print(f'l_c: {l_c}')

    # Calculate r_c
    r_dx = dx - (BASE_D / 2)
    r_dy = dy - (BASE_D / 2)

    r_r = (r_dx * r_dx + r_dy * r_dy) / (2 * abs(r_dx))
    r_theta = math.atan(r_dy/(r_r-r_dx)) if r_dx != r_dy else math.pi / 2
    print(f'r_r: {r_r}')
    r_d = r_r * r_theta
    print(f'r_d: {r_d}')
    r_c = r_d / c_d
    print(f'r_c: {r_c}')

move_to(0, 1000)
