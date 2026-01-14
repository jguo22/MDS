import math
BASE_D = 209

def move_to(dx, dy):
    # Make sure dx isn't 0
    if dx == 0:
        print("zero")
        return
    # Calculate distance to travel in arc for center of bot
    c_r = (dx * dx + dy * dy) / (2 * abs(dx))
    print(f'c_r: {c_r}')
    theta = math.asin(dy/c_r)
    print(f'c_theta: {theta}')
    c_d = c_r * theta

    print(f'distance: {c_d}')


    if dx > 0:
        l_r = c_r + (BASE_D / 2)
        r_r = c_r - (BASE_D / 2)
    else:
        l_r = c_r - (BASE_D / 2)
        r_r = c_r + (BASE_D / 2)

    # Calculate l_c
    l_d = l_r * theta
    print(f'l_d: {l_d}')
    l_c = l_d / c_d
    print(f'l_c: {l_c}')

    # Calculate r_c
    r_d = r_r * theta
    print(f'r_d: {r_d}')
    r_c = r_d / c_d
    print(f'r_c: {r_c}')

move_to(20, 1000)
