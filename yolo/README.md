train folder for training stuff 

# Run camera, press 'p' to save a frame as capture.png
  python3 yolo/yolo_detect.py --model yolo/train/runs/train/exp1/weights/best.pt --source usb0 --thresh 0.2

  # Then test the saved frame
  python3 yolo/yolo_detect.py --model yolo/train/runs/train/exp1/weights/best.pt --source capture.png --thresh 0.2


# Try different resolution
  python3 yolo/yolo_detect.py --model yolo/train/runs/train/exp1/weights/best.pt --source usb0 --thresh 0.2 --resolution 640x480

  # Or match training resolution
  python3 yolo/yolo_detect.py --model yolo/train/runs/train/exp1/weights/best.pt --source usb0 --thresh 0.2 --resolution 1280x720


