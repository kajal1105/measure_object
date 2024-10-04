import cv2
import numpy as np

# Load the image
image_path = r"C:\Users\shiva\Downloads\measure_object_size\4.jpeg"
image = cv2.imread(image_path)

# Resize the image to make it easier to work with (optional)
image = cv2.resize(image, (800, 600))

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply GaussianBlur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Use Canny edge detection to find edges
edges = cv2.Canny(blurred, 50, 150)

# Find contours from the edges
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Iterate through each contour to find the clips on the wire
for contour in contours:
    # Get the bounding box for each contour
    x, y, w, h = cv2.boundingRect(contour)

    # Define a heuristic to filter out clips based on size
    if 20 < w < 100 and 20 < h < 100:  # You can adjust these values based on the image
        # Draw a rectangle around the detected clip
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Show the result
cv2.imshow("Clips Detected", image)

# Wait for a key press and close the windows
cv2.waitKey(0)
cv2.destroyAllWindows()
