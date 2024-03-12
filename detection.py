import cv2 as cv
import numpy as np

def roi(url):
    """
    Extracts the region of interest (ROI) from the given image URL.

    Args:
    - url (str): The URL or file path of the input image.

    Returns:
    - roi (numpy.ndarray): The region of interest (ROI) extracted from the image.
    """
    # Load the image
    img = cv.imread(url)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    ker = np.ones((3,3), np.uint8)
    sharpenkernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

    # Apply Gaussian blur to reduce noise
    gray_blurred = cv.GaussianBlur(gray, (5, 5), cv.BORDER_CONSTANT)

    # Calculate gradients using Sobel operator
    Gx = cv.Sobel(gray_blurred, cv.CV_64F, 1, 0, ksize=3)
    Gy = cv.Sobel(gray_blurred, cv.CV_64F, 0, 1, ksize=3)
    GxA = cv.convertScaleAbs(Gx)
    GyA = cv.convertScaleAbs(Gy)
    resSobel = GxA + GyA

    # Apply thresholding to obtain binary image
    _, thr = cv.threshold(resSobel, 30, 255, cv.THRESH_BINARY)
    sharperes = cv.morphologyEx(thr, cv.MORPH_OPEN, ker, iterations=1)
    sharperes = cv.filter2D(sharperes, -1, sharpenkernel)

    # Detect lines using Hough Line Transformation
    lines = cv.HoughLinesP(sharperes, 2, np.pi/360, 60, minLineLength=50, maxLineGap=10)

    # Find the bounding box coordinates
    min_x = min([line[0][0] for line in lines])
    min_y = min([line[0][1] for line in lines])
    max_x = max([line[0][2] for line in lines])
    max_y = max([line[0][3] for line in lines])

    # Expand the bounding box by 100%
    roi_width = max_x - min_x
    roi_height = max_y - min_y
    expansion_width = int((max_x - min_x) * 0.3 / 2)
    expansion_height = int((max_y - min_y) * 0.05 / 2)
    min_x = max(0, min_x - expansion_width)
    min_y = max(0, min_y - expansion_height)
    max_x = min(img.shape[1], max_x + expansion_width)
    max_y = min(img.shape[0], max_y + expansion_height)

    # Calculate the width and height of the ROI
    roi_width = max_x - min_x
    roi_height = max_y - min_y

    # Check if the width-to-height ratio is less than 1:6
    if roi_height < 4 * roi_width:
        # Expand the bounding box height to achieve the desired ratio
        expansion_height = int((4 * roi_width - roi_height) / 2)
        min_y = max(0, min_y - expansion_height)
        max_y = min(img.shape[0], max_y + expansion_height)

    elif roi_height > 6 * roi_width:
        # Trim the bottom part of the ROI to keep the ratio under 1:6
        trim_height = int((roi_height - 6 * roi_width) / 2)
        max_y -= trim_height

    # Extract the ROI
    roi = rgb[min_y:max_y, min_x:max_x]

    return roi
