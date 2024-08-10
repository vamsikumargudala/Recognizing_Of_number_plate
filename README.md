
# Computer Vision and License Plate Recognition

Computer vision has evolved significantly over the decades and is now used in a variety of real-world applications, including facial recognition, disease diagnosis, and real-time object detection. This project demonstrates how computer vision techniques, particularly Convolutional Neural Networks (CNNs) and Optical Character Recognition (OCR), can be applied to license plate recognition.

## Overview

This project uses computer vision to:
1. Detect license plates from images of vehicles.
2. Segment and crop the license plate from the image.
3. Recognize and extract the alphanumeric characters on the license plate using OCR.

## Key Concepts

### Convolutional Neural Networks (CNNs)

A CNN is a type of neural network that excels at detecting patterns and features in images. It consists of multiple layers:
- **Input Layer:** Receives the raw image data.
- **Convolutional Layers:** Apply filters to detect patterns like edges, shapes, and textures.
- **Pooling Layers:** Reduce the dimensionality of the data.
- **Fully Connected Layers:** Perform the final classification.

### License Plate Recognition

The license plate recognition process involves several steps:

1. **License Plate Detection:**
   - Resize and convert the image to grayscale.
   - Apply a bilateral filter to remove noise.
   - Use edge detection to highlight the license plate.
   - Find contours and filter out the license plate based on its rectangular shape.
   - Mask the image to isolate the license plate.

2. **Character Segmentation:**
   - Crop the detected license plate from the image.

3. **Character Recognition:**
   - Use OCR (Optical Character Recognition) to read the characters from the cropped image.

## Implementation Steps

### License Plate Detection

```python
import cv2
import numpy as np
import imutils

# Load the image
img = cv2.imread('car.jpg')

# Resize and convert to grayscale
img = cv2.resize(img, (620, 480))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply bilateral filter
gray = cv2.bilateralFilter(gray, 13, 15, 15)

# Perform edge detection
edged = cv2.Canny(gray, 30, 200)

# Find contours
contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

# Filter contours to find the license plate
screenCnt = None
for c in contours:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.018 * peri, True)
    if len(approx) == 4:
        screenCnt = approx
        break

# Mask the license plate area
mask = np.zeros(gray.shape, np.uint8)
new_image = cv2.drawContours(mask, [screenCnt], 0, 255, -1)
new_image = cv2.bitwise_and(img, img, mask=mask)
```

### Character Segmentation and Recognition

```python
import pytesseract

# Segment the license plate
(x, y) = np.where(mask == 255)
(topx, topy) = (np.min(x), np.min(y))
(bottomx, bottomy) = (np.max(x), np.max(y))
cropped = gray[topx:bottomx+1, topy:bottomy+1]

# Perform OCR to read the license plate
text = pytesseract.image_to_string(cropped, config='--psm 11')
print("Detected license plate Number is:", text)
```

## Fetch Vehicle Owner Information

Once the license plate number is extracted, it can be sent to RTO (Regional Transport Office) APIs to fetch vehicle owner details such as name, address, engine number, and more.

## Conclusion

This project demonstrates how to utilize computer vision and machine learning techniques to perform license plate recognition, from detecting and isolating the plate to recognizing its characters and retrieving vehicle information.

Thank you for exploring this project!

```

Feel free to modify the content as needed for your specific use case.
