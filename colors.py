# --------------------- YOLO CLASS NAMES ---------------------

CLASS_NAMES = [
    'Boundary',      # 0
    'Golden Can',    # 1
    'Golden Zone',   # 2
    'Green Can',     # 3
    'Green Zone',    # 4
    'Red Can',       # 5
    'Red Zone',      # 6
    'Robot'          # 7
]

# --------------------- COLOR VALUES ---------------------

GREEN_ZONE = 0
RED_ZONE = 1
GOLDEN_ZONE = 2
GREEN_ZONE_OPP = 3
RED_ZONE_OPP = 4
GOLDEN_ZONE_OPP = 5

ZONE_CLASS_NAMES = [
    'Green Zone',
    'Red Zone',
    'Golden Zone'
]

GREEN_CAN = 0
RED_CAN = 1
GOLDEN_CAN = 2

CAN_CLASS_NAMES = [
    'Green Can',
    'Red Can',
    'Golden Can'
]


def zoneNamesToNumbers(names: list[str]) -> list[int]:
    return [ZONE_CLASS_NAMES.index(name) for name in names]


def canNamesToNumbers(names: list[str]):
    return [CAN_CLASS_NAMES.index(name) for name in names]
