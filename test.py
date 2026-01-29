from thetaStar import ThetaStar


thetaStar = ThetaStar()

sx, sy = 0, 0
gx, gy = 1000, 90
thetaStar.set_start(sx, sy)
thetaStar.set_goal(gx, gy)
print("theta*")
print(sx, sy)
print(gx, gy)
print(thetaStar.ox)
rx, ry = thetaStar.path_find()

# asdfas
