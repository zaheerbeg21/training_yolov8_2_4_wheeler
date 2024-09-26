import os
import yaml
from ultralytics import YOLO

# Set CUDA environment variable
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Load configuration
conf_path = '/home/training_yolov8_2_4_wheeler/local/config.yaml'   # always use absolute path

with open(conf_path, 'r') as file:
    config = yaml.safe_load(file)

# Load a pretrained model
model = YOLO('/home/training_yolov8_2_4_wheeler/weight/best.pt')   # always use absolute path

# Set training parameters
number_of_epochs = config.get('epochs', 300)
batch_size = config.get('batch', 16)
image_size = config.get('imgsz', 640)

# Additional parameters
patience = config.get('patience', 50)
save_period = config.get('save_period', 10)

# Use the model
results = model.train(
    data=conf_path,
    epochs=number_of_epochs,
    imgsz=image_size,
    batch=batch_size,
    patience=patience,
    save_period=save_period,
    device='0',  # use GPU. Set to 'cpu' if you want to use CPU
    project='yolov8_2_4_wheeler_anpr',
    name='experiment_1',  # creates a new folder for each run
    exist_ok=False,  # increment run if exists
    pretrained=True,
    optimizer=config.get('optimizer', 'Adam'),
    lr0=config.get('lr0', 0.01),
    lrf=config.get('lrf', 0.01),
    momentum=config.get('momentum', 0.937),
    weight_decay=config.get('weight_decay', 0.0005),
    warmup_epochs=config.get('warmup_epochs', 3.0),
    warmup_momentum=config.get('warmup_momentum', 0.8),
    warmup_bias_lr=config.get('warmup_bias_lr', 0.1),
    box=config.get('box', 7.5),
    cls=config.get('cls', 0.5),
    dfl=config.get('dfl', 1.5),
    plots=True,  # save plots of results
    save=True,  # save trained model
    verbose=config.get('verbose', True)
)

# Evaluate the model on the validation set
val_results = model.val()

# Optionally, you can run inference on a test image
# model.predict('path/to/test/image.jpg', save=True, imgsz=640, conf=0.25)

print(f"Training complete. Results saved in {results}")
print(f"Validation results: {val_results}")