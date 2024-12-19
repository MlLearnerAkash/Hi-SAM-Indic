import cv2
import numpy as np
import pytesseract

# Configure pytesseract
pytesseract.pytesseract.tesseract_cmd = "/home/akash/miniconda3/envs/hi_sam/bin/tesseract"

# Load the image
image_path = '/home/akash/ws/dataset/ST/BharatST/YOLODataset/images/val/D_image_625.jpg'
image = cv2.imread(image_path)

# Read annotations from the text file
annotations_file = '/home/akash/ws/dataset/ST/BharatST/YOLODataset/labels/val/D_image_625.txt'
annotations = []

with open(annotations_file, 'r') as file:
    for line in file.readlines():
        # Split the line into individual values
        parts = line.strip().split()
        coords = list(map(float, parts[1:]))  # Convert coordinates to float
        annotations.append(coords)

# Convert annotations to pixel coordinates
height, width, _ = image.shape
polygons = []
for annotation in annotations:
    # Each annotation consists of 8 values (4 points, each with x and y)
    polygon = []
    for i in range(0, len(annotation), 2):
        x = int(annotation[i] * width)
        y = int(annotation[i + 1] * height)
        polygon.append((x, y))
    polygons.append(polygon)

# Create a mask for the annotated regions
mask = np.zeros((height, width), dtype=np.uint8)
for polygon in polygons:
    polygon_np = np.array(polygon, np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [polygon_np], color=255)

# Grayscale the image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# Morphological operations to remove noise
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

# Mask the annotated regions
inverted_opening = cv2.bitwise_and(255 - opening, mask)

# Extract words in white pixels
# Retain only white pixels (words) in the annotated regions
words_only = cv2.bitwise_and(opening, mask)

# Save and visualize the result
cv2.imwrite('words_only.png', words_only)

# Optional: Perform text extraction on the result
data = pytesseract.image_to_string(words_only, lang='eng', config='--psm 6')
print("Extracted Text:\n", data)

# Visualize results (optional)
# cv2.imshow('Words Only', words_only)
# cv2.waitKey(0)
# cv2.destroyAllWindows()