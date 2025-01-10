import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
import easyocr
import os
from datetime import datetime

try:
    # Load the image
    img_path = 'numberPlate/photos/images.jpg'
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at the path: {img_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
except FileNotFoundError as e:
    print(e)
    exit()
except Exception as e:
    print(f"Error loading or converting image to grayscale: {e}")
    exit()

try:
    # Noise reduction and edge detection
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)  # Noise reduction
    edged = cv2.Canny(bfilter, 30, 200)  # Edge detection
except Exception as e:
    print(f"Error in noise reduction or edge detection: {e}")
    exit()

try:
    # Finding contours
    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
except Exception as e:
    print(f"Error finding contours: {e}")
    exit()

# Locate the contour of the license plate
location = None
try:
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            location = approx
            break
    if location is None:
        raise ValueError("License plate contour not detected.")
except ValueError as e:
    print(e)
    exit()
except Exception as e:
    print(f"Error finding license plate contour: {e}")
    exit()

try:
    # Create a mask and extract the region of interest
    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [location], 0, 255, -1)
    new_image = cv2.bitwise_and(img, img, mask=mask)

    # Crop the image based on the mask
    (x, y) = np.where(mask == 255)
    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y))
    cropped_image = gray[x1:x2 + 1, y1:y2 + 1]
except Exception as e:
    print(f"Error in creating mask or cropping the image: {e}")
    exit()

try:
    # Save the cropped image to a different folder with a unique name using a timestamp
    output_folder = 'numberPlate/cropped'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Generate a unique filename using the current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_filename = f'cropped_license_plate_{timestamp}.png'
    output_path = os.path.join(output_folder, unique_filename)
    cv2.imwrite(output_path, cropped_image)
except Exception as e:
    print(f"Error saving the cropped image: {e}")
    exit()

try:
    # Use EasyOCR to read text from the cropped image
    reader = easyocr.Reader(['en'])
    result = reader.readtext(cropped_image)
    if not result:
        raise ValueError("No text detected from the cropped image.")
    
    # Extract detected text
    text = result[0][-2]
except ValueError as e:
    print(e)
    exit()
except Exception as e:
    print(f"Error reading text from the cropped image: {e}")
    exit()

try:
    # Draw text and rectangle around the detected license plate
    font = cv2.FONT_HERSHEY_SIMPLEX
    res = cv2.putText(img, text=text, org=(approx[0][0][0], approx[1][0][1] + 60),
                      fontFace=font, fontScale=1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    res = cv2.rectangle(img, tuple(approx[0][0]), tuple(approx[2][0]), (0, 255, 0), 3)

    # Plot the final image with detected text
    plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
except Exception as e:
    print(f"Error drawing text/rectangle or plotting the final image: {e}")
    exit()
