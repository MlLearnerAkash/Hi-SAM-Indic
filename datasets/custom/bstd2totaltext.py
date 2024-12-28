# import cv2
# import numpy as np
# import pytesseract

# # Configure pytesseract
# pytesseract.pytesseract.tesseract_cmd = "/home/akash/miniconda3/envs/hi_sam/bin/tesseract"

# # Load the image
# image_path = '/home/akash/ws/dataset/ST/BharatST/YOLODataset/images/val/D_image_625.jpg'
# image = cv2.imread(image_path)

# # Read annotations from the text file
# annotations_file = '/home/akash/ws/dataset/ST/BharatST/YOLODataset/labels/val/D_image_625.txt'
# annotations = []

# with open(annotations_file, 'r') as file:
#     for line in file.readlines():
#         # Split the line into individual values
#         parts = line.strip().split()
#         coords = list(map(float, parts[1:]))  # Convert coordinates to float
#         annotations.append(coords)

# # Convert annotations to pixel coordinates
# height, width, _ = image.shape
# polygons = []
# for annotation in annotations:
#     # Each annotation consists of 8 values (4 points, each with x and y)
#     polygon = []
#     for i in range(0, len(annotation), 2):
#         x = int(annotation[i] * width)
#         y = int(annotation[i + 1] * height)
#         polygon.append((x, y))
#     polygons.append(polygon)

# # Create a mask for the annotated regions
# mask = np.zeros((height, width), dtype=np.uint8)
# for polygon in polygons:
#     polygon_np = np.array(polygon, np.int32).reshape((-1, 1, 2))
#     cv2.fillPoly(mask, [polygon_np], color=255)



import cv2
import numpy as np
import pytesseract
import os

# Configure pytesseract (if needed)
pytesseract.pytesseract.tesseract_cmd = "/home/akash/miniconda3/envs/hi_sam/bin/tesseract"

def create_and_save_mask(image_path, annotations_file, target_dir):
    """
    Creates a mask from image annotations and saves it.

    Args:
      image_path: Path to the image file.
      annotations_file: Path to the annotation file.
      target_dir: Directory to save the mask.
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        height, width, _ = image.shape
        annotations = []
        with open(annotations_file, 'r') as file:
            for line in file.readlines():
                parts = line.strip().split()
                coords = list(map(float, parts[1:]))
                annotations.append(coords)

        polygons = []
        for annotation in annotations:
            polygon = []
            for i in range(0, len(annotation), 2):
                x = int(annotation[i] * width)
                y = int(annotation[i + 1] * height)
                polygon.append((x, y))
            polygons.append(polygon)

        mask = np.zeros((height, width), dtype=np.uint8)
        for polygon in polygons:
            polygon_np = np.array(polygon, np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [polygon_np], color=255)

        # Save the mask
        mask_filename = os.path.splitext(os.path.basename(image_path))[0] + "_mask.jpeg"
        mask_path = os.path.join(target_dir, mask_filename)
        cv2.imwrite(mask_path, mask)
        print(f"Mask saved to: {mask_path}")

    except Exception as e:
        print(f"Error processing {image_path}: {e}")

if __name__ == "__main__":
    images_dir = '/home/akash/ws/dataset/ST/BharatST/YOLODataset/images/train/'  # Replace with your images directory
    annotations_dir = '/home/akash/ws/dataset/ST/BharatST/YOLODataset/labels/train/'  # Replace with your annotations directory
    target_dir = '/home/akash/ws/dataset/ST/BharatST/YOLODataset/Word_mask/train'  # Replace with your target directory
    bad_file_list=[]
    for filename in os.listdir(images_dir):
        if filename.endswith(('.jpg', '.png', '.jpeg')):  # Adjust extensions as needed
            image_path = os.path.join(images_dir, filename)
            annotations_file = os.path.join(annotations_dir, filename.split(".")[0] + '.txt')

            # Check if the annotation file exists
            if os.path.exists(annotations_file):
                create_and_save_mask(image_path, annotations_file, target_dir)
            else:
                print(f"Annotation file not found for {filename}")
                bad_file_list.append(filename)
    with open('bad_file_list.txt', 'w') as file:
        for item in bad_file_list:
            file.write(f"{item}\n")