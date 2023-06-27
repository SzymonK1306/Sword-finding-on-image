import cv2
import numpy as np
from pathlib import Path
import os

path = Path(f'{os.getcwd()}')
sword_list = list(path.glob('result/*.jpg'))

background_list = list(path.glob('downloads/*.jpg'))

print(len(background_list))

image_number = 0

for background in background_list:
    for i in range(20):
        # Load the swords and landscape images
        sword_index = np.random.randint(0, len(sword_list))
        sword = cv2.imread(str(sword_list[sword_index]))
        landscape = cv2.imread(str(background))

        sword = cv2.resize(sword, (int(landscape.shape[1] * 0.4), int(landscape.shape[0] * 0.4)))

        # Create a mask for the swords image
        gray = cv2.cvtColor(sword, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=len)

        print(contour)
        x_box, y_box, w_box, h_box = cv2.boundingRect(contour)

        # Get the dimensions of the swords image
        height, width, _ = sword.shape

        # Generate a random rotation angle for the swords
        angle = np.random.randint(0, 360)

        # Rotate the swords image with the generated angle
        M = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
        swords_rotated = cv2.warpAffine(sword, M, (width, height))

        mask_rotated = cv2.warpAffine(mask, M, (width, height))

        # Place the rotated swords image on the landscape image
        x = np.random.randint(0, landscape.shape[1] - width)
        y = np.random.randint(0, landscape.shape[0] - height)
        roi = landscape[y:y + height, x:x + width]
        masked_swords = cv2.bitwise_and(swords_rotated, swords_rotated, mask=mask_rotated)
        masked_roi = cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(mask_rotated))
        result = cv2.add(masked_swords, masked_roi)

        # Define the center of the rectangle and its dimensions
        center = (w_box//2, h_box//2)
        size = (w_box, h_box)

        # Calculate the corner points of the rectangle
        rect = cv2.boxPoints((center, size, -angle))

        # Convert the points to integers
        rect = np.int0(rect)

        # Draw the rectangle on the image
        cv2.drawContours(result, [rect], 0, (0, 0, 255), 3)

        # cv2.rectangle(result, (x_box, y_box), (x_box + w_box, y_box + h_box), (0, 255, 0), 2)
        landscape[y:y + height, x:x + width] = result

        cv2.imwrite(f'dataSet/image{image_number}.jpg', landscape)

        image_number += 1
