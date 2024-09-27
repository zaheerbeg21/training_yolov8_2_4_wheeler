from ultralytics import YOLO

# Load a pretrained model (recommended for transfer learning)
model = YOLO('yolov8s.pt')  # Using 's' (small) model instead of 'n' (nano) for better performance

# Define hyperparameters
hyper_params = {
    'data': 'path/to/your/custom_data.yaml',  # Replace with path to your custom data YAML
    'epochs': 100,
    'imgsz': 640,
    'batch': 16,
    'lr0': 0.01,
    'single_cls': True,  # Since you're working with a single class
    'patience': 50,  # Early stopping patience
    'save_period': 10,  # Save checkpoint every 10 epochs
}

# Train the model
results = model.train(**hyper_params)

# Evaluate model performance on the validation set
val_results = model.val()

# Perform object detection on a test image
test_results = model('path/to/your/test_image.jpg')

# Export the model to ONNX format
success = model.export(format='onnx')
