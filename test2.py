import cv2
import numpy as np

# Load the image
image_path = r"C:\Users\shiva\Downloads\measure_object_size\mixed.jpg"
image = cv2.imread(image_path)

# Resize the image for easier processing (optional)
image = cv2.resize(image, (800, 600))

# Convert the image to HSV color space
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define the range for clip color (modify this based on your image)
# Assuming clips are orange/yellow, adjust values as needed
lower_clip_color = np.array([15, 100, 100])  # Lower boundary for the clip color
upper_clip_color = np.array([35, 255, 255])  # Upper boundary for the clip color

# Create a mask for the clip color
clip_mask = cv2.inRange(hsv, lower_clip_color, upper_clip_color)

# Apply morphological operations to clean the mask
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
clip_mask = cv2.morphologyEx(clip_mask, cv2.MORPH_CLOSE, kernel)
clip_mask = cv2.morphologyEx(clip_mask, cv2.MORPH_OPEN, kernel)

# Find contours on the mask
contours, _ = cv2.findContours(clip_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Iterate through each contour and draw bounding boxes for clips
for contour in contours:
    # Get the bounding box for each contour
    x, y, w, h = cv2.boundingRect(contour)

    # Filter contours based on size (adjust these values based on your image)
    if 20 < w < 100 and 20 < h < 100:  # These values can be adjusted as needed
        # Draw a rectangle around the detected clip
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display the result
cv2.imshow("Clips Detected", image)

# Wait for a key press and close the windows
cv2.waitKey(0)
cv2.destroyAllWindows()
