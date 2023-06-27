import cv2 as cv
import os
from pathlib import Path
from PIL import Image
from google_images_download import google_images_download

import numpy as np
from matplotlib import pyplot as plt


def rotate(image_to_rotate):
    # convert the image to grayscale
    gray = cv.cvtColor(result, cv.COLOR_BGR2GRAY)

    # Threshold the image to obtain a binary image
    ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY)

    non_zero_coords = cv.findNonZero(thresh)

    # Iterate over the non-zero pixels and find the pixel with the lowest grayscale intensity
    min_intensity = 255
    min_intensity_coord = None
    for coord in non_zero_coords:
        x, y = coord[0]
        intensity = gray[y, x]
        if intensity < min_intensity:
            min_intensity = intensity
            min_intensity_coord = coord

    handle_cord_x, handle_cord_y = min_intensity_coord[0]

    # Divide the binary image into four quarters
    height, width = thresh.shape
    if handle_cord_x < width / 2 and handle_cord_y < height / 2:
        quarter = 1
    elif handle_cord_x < width and handle_cord_y < height / 2 or handle_cord_y > 500:
        quarter = 2
    elif handle_cord_x < width / 2 and handle_cord_y < height:
        quarter = 3
    else:
        quarter = 4

    # apply edge detection using the Canny edge detector
    edges = cv.Canny(gray, 50, 150, apertureSize=3)

    # detect lines in the image using the Hough transform
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=100, maxLineGap=10)

    # filter out lines that do not correspond to the blade of the sword
    filtered_lines = lines

    angle = 0
    if len(filtered_lines) > 0:
        x1, y1, x2, y2 = filtered_lines[0][0]
        slope = (y2 - y1) / (x2 - x1)
        angle = np.arctan(slope) * 180 / np.pi + 270
        if quarter == 1 or quarter == 3:
            angle += 180

    (h, w) = result.shape[:2]
    expansion_factor = 1.2
    M = cv.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)

    # calculate the size of the expanded image
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    #
    # # apply the rotation matrix to the expanded image
    M[0, 2] += new_w / 2 - w // 2
    M[1, 2] += new_h / 2 - h // 2
    expanded_img = cv.warpAffine(result, M, (new_w, new_h))

    return expanded_img

if __name__ == '__main__':
    path = Path(f'{os.getcwd()}')
    file_list = list(path.glob('cab/*.jpg'))
    file_list_jpeg = list(path.glob('*/*.jpeg'))
    file_list.extend(file_list_jpeg)
    file_list_png = list(path.glob('*/*.png'))
    # file_list.extend(file_list_png)
    # print(len(file_list))

    response = google_images_download.googleimagesdownload()
    arguments = {"keywords": "sword",
                 "format": "jpg",
                 "limit": 4,
                 "print_urls": True,
                 "size": "medium",
                 "aspect_ratio":"panoramic"}
    response.download(arguments)

    output_size = (256, 256)
    i = 0

    for file in file_list:
        # load image
        image = cv.imread(str(file), cv.IMREAD_UNCHANGED)

        # Convert the image to grayscale
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        gauss = cv.GaussianBlur(gray, (5, 5), 0)

        threshold = cv.adaptiveThreshold(gauss, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 5.5)

        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
        dilated = cv.dilate(threshold, kernel, iterations=3)

        # _, threshold = cv.threshold(gauss, 230, 255, cv.THRESH_BINARY_INV)

        # Find the contours
        contours, _ = cv.findContours(dilated, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        max_contur = max(contours, key=cv.contourArea)

        # Draw the contours
        mask = np.zeros_like(gray)

        cv.drawContours(mask, [max_contur], -1, 255, cv.FILLED)

        # Apply the mask
        result = cv.bitwise_and(image, image, mask=mask)

        result = rotate(result)

        result = cv.resize(result, output_size)

        cv.imwrite(f'result/image{i}.jpg', result)
        i += 1

    for file in file_list_png:
        image = np.array(Image.open(file).convert('RGBA'))
        _, mask = cv.threshold(image[:, :, 3], 180, 255, cv.THRESH_BINARY)
        image = image[:, :, :3]
        result = cv.bitwise_and(image, image, mask=mask)
        result = cv.cvtColor(result, cv.COLOR_BGR2RGB)
        result = rotate(result)
        result = cv.resize(result, output_size)
        cv.imwrite(f'result/image{i}.jpg', result)
        i += 1

