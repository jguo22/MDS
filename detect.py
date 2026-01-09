import cv2

LOWER_GREEN = (40, 100, 100)
UPPER_GREEN = (80, 255, 255)


def detect(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    thresh = cv2.inRange(img, LOWER_GREEN, UPPER_GREEN)

    cv2.imshow('Initial Threshold', thresh)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    thresh = cv2.dilate(thresh, kernel, iterations=2)
    cv2.imshow('Threshold', thresh)
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    contours = [cv2.convexHull(contour) for contour in contours]
    return contours


def draw_contours(img, contours):
    for contour in contours:
        if cv2.contourArea(contour) > 100:
            cv2.drawContours(img, [contour], -1, (0, 255, 0), 2)
    return img


if __name__ == "__main__":
    img_path = 'out.jpg'
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Could not read image at {img_path}")
    else:
        contours = detect(img)
        result = draw_contours(img.copy(), contours)
        cv2.imshow('Detected Contours', result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
