import cv2
import numpy as np

# Load the image
image_path = r"C:\Users\shiva\Downloads\measure_object_size\1.jpeg"
image = cv2.imread(image_path)

# Convert the image to the HSV color space
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define the range of orange color in HSV
lower_orange = np.array([5, 100, 100])  # Lower boundary for orange color
upper_orange = np.array([15, 255, 255]) # Upper boundary for orange color

# Create a mask for the orange color
mask = cv2.inRange(hsv, lower_orange, upper_orange)

# Apply the mask to the original image
result = cv2.bitwise_and(image, image, mask=mask)

# Display the original image and the result
cv2.imshow("Original Image", image)
cv2.imshow("Orange Wire Detected", result)

# Wait until a key is pressed, then close the displayed images
cv2.waitKey(0)
cv2.destroyAllWindows()
