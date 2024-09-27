from ultralytics import YOLO
import cv2
import os

# Load the model
model = YOLO('path/to/your/best.pt')  # replace with the path to your best.pt file

# Define the path to your test images
test_images_path = 'path/to/your/test/images'  # replace with your test images directory

# Function to perform inference on an image
def perform_inference(image_path):
    # Read the image
    img = cv2.imread(image_path)
    
    # Perform inference
    results = model(img)
    
    # Visualize the results on the image
    annotated_img = results[0].plot()
    
    # Save the annotated image
    output_path = os.path.join('output', os.path.basename(image_path))
    cv2.imwrite(output_path, annotated_img)
    
    print(f"Processed {image_path}. Output saved to {output_path}")

# Create output directory if it doesn't exist
os.makedirs('output', exist_ok=True)

# Iterate over all images in the test directory
for image_file in os.listdir(test_images_path):
    if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(test_images_path, image_file)
        perform_inference(image_path)

print("Testing completed. Check the 'output' directory for results.")

# Evaluate the model's performance
print("Evaluating model performance...")
results = model.val(data='path/to/your/data.yaml')  # replace with the path to your data YAML file
print(f"mAP50-95: {results.box.map}")  # Mean Average Precision
print(f"Precision: {results.box.p}")
print(f"Recall: {results.box.r}")
