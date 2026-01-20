from yolo.segment import getQuadrilateralsAndClasses, segmentImage
from pixelTo3D import transform_uv_to_xy
import numpy as np
import cv2 as cv


image_path = str("yolo/test.jpg")
print(f"Loading image: {image_path}")

image = cv.imread(image_path)
if image is None:
    raise Exception("no image")

print("Running segmentation...")

result = segmentImage(image)
quads, classes = getQuadrilateralsAndClasses(result, image)

# Transform each vertex from pixel coordinates to ground plane coordinates
transformed_quads = []
for quad in quads:
    # quad shape is (4, 1, 2) -> reshape to (4, 2)
    vertices = quad.reshape(4, 2)

    # Transform each vertex
    transformed_vertices = []
    for vertex in vertices:
        u, v = vertex[0], vertex[1]  # pixel coordinates
        x, y = transform_uv_to_xy(u, v)  # ground plane coordinates (mm)
        transformed_vertices.append([x, y])

    transformed_quads.append(np.array(transformed_vertices))

# Calculate center xy point of each quad
centers = []
for transformed_quad in transformed_quads:
    # Average all x and y coordinates
    center_x = np.mean(transformed_quad[:, 0])
    center_y = np.mean(transformed_quad[:, 1])
    centers.append((center_x, center_y))

# Get only the closest of each class
# Group by class and find closest (minimum distance from origin/robot)
temp_closest = {}
for i, (center, class_name) in enumerate(zip(centers, classes)):
    x, y = center
    # Distance from robot (at origin): sqrt(x^2 + y^2)
    distance = np.sqrt(x**2 + y**2)

    if class_name not in temp_closest:
        temp_closest[class_name] = {
            'index': i,
            'center': center,
            'distance': distance,
            'quad': transformed_quads[i]
        }
    else:
        # Update if this one is closer
        if distance < temp_closest[class_name]['distance']:
            temp_closest[class_name] = {
                'index': i,
                'center': center,
                'distance': distance,
                'quad': transformed_quads[i]
            }

# Convert to parallel arrays
closest_quads_xy = []  # Array of quads with xy coordinates
closest_centers = []   # Array of xy center points
closest_classes = []   # Array of class names

for class_name, data in temp_closest.items():
    closest_quads_xy.append(data['quad'])
    closest_centers.append(data['center'])
    closest_classes.append(class_name)

print(f"Found {len(closest_classes)} unique classes:")
for i, class_name in enumerate(closest_classes):
    x, y = closest_centers[i]
    distance = temp_closest[class_name]['distance']
    print(f"  {class_name}: center=({x:.1f}, {y:.1f}) mm, distance={distance:.1f} mm")
