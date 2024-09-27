CUDA_LAUNCH_BLOCKING=1

from ultralytics import YOLO

# Load a model
# model = YOLO('yolov8n.yaml')  # build a new model from YAML
# model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights
# model = YOLO('weight/epc_100_dataset_165/weights/best.pt').load('weight/epc_100_dataset_165/weights/best.pt')  # build from YAML and transfer weights

# Load a model
model = YOLO('yolov8n.yaml')  # build a new model from YAML

## Resume training
#results = model.train(resume=True)

## load a pretrained model (recommended for training)
#model = YOLO('/home/training_yolov8_2_4_wheeler/runs/detect/train/weights/last.pt')   # always use absolute path

conf_path = '/home/training_yolov8_2_4_wheeler/data/config.yaml'   # always use absolute path
number_of_epochs = 100


## Use the model
results = model.train(data=conf_path, 
                        epochs=number_of_epochs,
                        lr0 = 0.01,  # initial learning rate
                        lrf= 0.01,  # final learning rate (lr0 * lrf)
                        optimizer= "AdamW",
                        imgsz = 640,
                        batch= 16,  # train the model
                        project = 'anpr_yolov8',
                        name = 'model_weight',
                        single_cls =  True,  # Assuming you're working with a single class (added symbols)
                        save_period = 25)  # Save a checkpoint every 50 epochs)  # Save a checkpoint every 50 epochs
                    

