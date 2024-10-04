import cv2
import numpy as np

# Load the image
image_path = r"C:\Users\shiva\Downloads\measure_object_size\1.jpeg"
image = cv2.imread(image_path)

# Resize the image (optional)
image = cv2.resize(image, (800, 600))

# Convert the image to HSV to apply a color filter
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define the color range for the wire (adjust based on the wire's color)
lower_color = np.array([0, 0, 0])     # Lower bound of the wire color in HSV
upper_color = np.array([180, 255, 50]) # Upper bound of the wire color in HSV

# Create a mask to filter out the wire
mask = cv2.inRange(hsv, lower_color, upper_color)

# Apply some morphological operations to clean up the mask
kernel = np.ones((5, 5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

# Find contours from the mask
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Initialize a variable to store the wire bounding box
wire_bounding_box = None

# Iterate through each contour to find the wire
for contour in contours:
    # Get the bounding box for each contour
    x, y, w, h = cv2.boundingRect(contour)

    # Define a heuristic to identify the wire based on size and shape
    if w > 100 and h > 20:  # Adjust these values based on the expected size of the wire
        wire_bounding_box = (x, y, w, h)
        # Draw the detected wire
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

# If the wire is found, check for interruptions (i.e., clips)
if wire_bounding_box:
    for contour in contours:
        # Get the bounding box for each contour
        x, y, w, h = cv2.boundingRect(contour)

        # Define a heuristic to filter out clips based on size
        if 20 < w < 100 and 20 < h < 100:  # Adjust these values based on the size of clips
            # Draw a rectangle around the detected clip
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Check if the clip overlaps with the wire's bounding box
            wire_x, wire_y, wire_w, wire_h = wire_bounding_box
            if (x > wire_x and x < wire_x + wire_w) or (y > wire_y and y < wire_y + wire_h):
                cv2.putText(image, "Wire is clipped", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                break
else:
    cv2.putText(image, "Wire not found", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

# Show the result
cv2.imshow("Wire and Clips Detected", image)

# Wait for a key press and close the windows
cv2.waitKey(0)
cv2.destroyAllWindows()
