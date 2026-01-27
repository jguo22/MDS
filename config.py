FT_TO_MM = 304.8

# measurements are in mm
CENTER_BORDER_X = 8 * FT_TO_MM
BACK_BORDER_X = 0
LEFT_BORDER_Y = 4 * FT_TO_MM
RIGHT_BORDER_Y = - 4 * FT_TO_MM

CAN_DIAMETER = 76.2

CLASS_NAMES = [
    'Boundary', # TODO: remove boundary class
    'Golden Can',
    'Golden Zone',
    'Green Can',
    'Green Zone',
    'Red Can',
    'Red Zone',
    'Robot'
]

WHEEL_D = 101.6
BASE_D = 237.236
SCOOPER_LENGTH = 254
