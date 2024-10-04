import cv2
import numpy as np

# Load the image
image_path = r"C:\Users\shiva\Downloads\measure_object_size\1.jpeg"
image = cv2.imread(image_path)

# Resize the image (optional)
image = cv2.resize(image, (800, 600))

# Convert the image to HSV to apply a color filter
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define the color range for the wire/rope (adjust based on the color)
lower_color = np.array([0, 0, 0])     # Lower bound of the wire/rope color in HSV
upper_color = np.array([180, 255, 50]) # Upper bound of the wire/rope color in HSV

# Create a mask to filter out the wire/rope
mask = cv2.inRange(hsv, lower_color, upper_color)

# Apply some morphological operations to clean up the mask
kernel = np.ones((5, 5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

# Find contours from the mask
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Initialize variables to store the wire bounding box and endpoints
wire_contour = None
endpoints = []

# Iterate through each contour to find the wire/rope
for contour in contours:
    # Get the bounding box for each contour
    x, y, w, h = cv2.boundingRect(contour)

    # Define a heuristic to identify the wire/rope based on size and shape
    if w > 100 and h > 20:  # Adjust these values based on the expected size of the wire/rope
        wire_contour = contour
        # Draw the detected wire/rope
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Find the endpoints of the wire/rope using the contour
        leftmost = tuple(contour[contour[:, :, 0].argmin()][0])
        rightmost = tuple(contour[contour[:, :, 0].argmax()][0])
        endpoints = [leftmost, rightmost]

        # Draw circles at the endpoints
        for point in endpoints:
            cv2.circle(image, point, 10, (0, 255, 255), -1)

# If the wire/rope is found, check for connectors at the endpoints
if wire_contour is not None and len(endpoints) == 2:
    for point in endpoints:
        px, py = point

        # Define a region of interest (ROI) around the endpoint to detect connectors
        roi_size = 30  # Adjust this value based on the expected size of the connector
        roi_x_start = max(px - roi_size, 0)
        roi_y_start = max(py - roi_size, 0)
        roi_x_end = min(px + roi_size, image.shape[1])
        roi_y_end = min(py + roi_size, image.shape[0])

        # Extract the ROI
        roi = mask[roi_y_start:roi_y_end, roi_x_start:roi_x_end]

        # Find contours within the ROI
        roi_contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Check if there is a connector within the ROI
        connector_found = False
        for roi_contour in roi_contours:
            x, y, w, h = cv2.boundingRect(roi_contour)

            # Define a heuristic to identify the connector based on size
            if w > 10 and h > 10:  # Adjust these values based on the expected connector size
                # Draw a green rectangle around the detected connector in the ROI
                cv2.rectangle(image, (roi_x_start + x, roi_y_start + y),
                              (roi_x_start + x + w, roi_y_start + y + h),
                              (0, 255, 0), 2)
                connector_found = True

        # If a connector is detected at the endpoint, draw a red rectangle around the endpoint
        if connector_found:
            cv2.rectangle(image, (px - roi_size, py - roi_size),
                          (px + roi_size, py + roi_size),
                          (0, 0, 255), 2)
            # Add text indicating the endpoint is covered
            cv2.putText(image, "Endpoint is covered", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

# If no wire/rope is found
else:
    cv2.putText(image, "Wire/Rope not found", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

# Show the result
cv2.imshow("Wire/Rope and Connectors Detected", image)

# Wait for a key press and close the windows
cv2.waitKey(0)
cv2.destroyAllWindows()
