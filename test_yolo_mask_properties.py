#!/usr/bin/env python3
"""Test what properties YOLO masks have."""

import cv2
from vision.segment import segmentImage

# Load any test image
image = cv2.imread('test.jpg')
if image is None:
    print("No test.jpg found, trying other images...")
    import glob
    images = glob.glob('*.jpg') + glob.glob('*.png')
    if images:
        image = cv2.imread(images[0])
        print(f"Using {images[0]}")

if image is None:
    print("No images found!")
    exit(1)

print(f"Image shape: {image.shape}")

# Run segmentation
result = segmentImage(image)

if result.masks is None:
    print("No masks detected!")
    exit(1)

print(f"\nFound {len(result.masks)} masks")

# Inspect first mask
mask = result.masks[0]
print(f"\nMask object type: {type(mask)}")
print(f"Mask attributes: {[attr for attr in dir(mask) if not attr.startswith('_')]}")

# Check specific properties
if hasattr(mask, 'data'):
    print(f"\nmask.data shape: {mask.data.shape}")

if hasattr(mask, 'xy'):
    print(f"\nmask.xy: {mask.xy}")
    print(f"mask.xy type: {type(mask.xy)}")
    if len(mask.xy) > 0:
        print(f"mask.xy[0] shape: {mask.xy[0].shape if hasattr(mask.xy[0], 'shape') else 'N/A'}")
        print(f"First few points: {mask.xy[0][:5] if len(mask.xy[0]) > 0 else 'empty'}")

if hasattr(mask, 'xyn'):
    print(f"\nmask.xyn: type = {type(mask.xyn)}")

if hasattr(mask, 'shape'):
    print(f"\nmask.shape: {mask.shape}")

# Check orig_shape
if hasattr(result, 'orig_shape'):
    print(f"\nresult.orig_shape: {result.orig_shape}")
