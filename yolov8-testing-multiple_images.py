from ultralytics import YOLO
import cv2
import os
from tqdm import tqdm

# Load the model
model = YOLO('/content/best.pt')  # replace with the path to your best.pt file

# Define the path to your test images
test_images_path = '/content/images'  # replace with your test images directory

# Function to perform inference on an image
def perform_inference(image_path):
    # Read the image
    img = cv2.imread(image_path)
    
    # Perform inference
    results = model(img)
    
    # Visualize the results on the image
    annotated_img = results[0].plot()
    
    # Get the base name and file extension of the image
    base_name, ext = os.path.splitext(os.path.basename(image_path))
    
    # Construct the output file name with '_output'
    output_file_name = base_name + '_output' + ext
    
    # Save the annotated image
    output_path = os.path.join('output', output_file_name)
    cv2.imwrite(output_path, annotated_img)
    
    return results[0]  # Return the results for further analysis if needed

# Create output directory if it doesn't exist
os.makedirs('output', exist_ok=True)

# Get list of image files
image_files = [f for f in os.listdir(test_images_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Process images
all_results = []
for image_file in tqdm(image_files, desc="Processing images"):
    image_path = os.path.join(test_images_path, image_file)
    result = perform_inference(image_path)
    all_results.append(result)

print(f"Testing completed. Processed {len(image_files)} images. Check the 'output' directory for results.")
