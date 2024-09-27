CUDA_LAUNCH_BLOCKING=1

from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.yaml')  # build a new model from YAML
## load a pretrained model (recommended for training)
model = YOLO('/home/training_yolov8_2_4_wheeler/runs/detect/train/weights/last.pt')   # always use absolute path

conf_path = '/home/training_yolov8_2_4_wheeler/local/config.yaml'   # always use absolute path
number_of_epochs = 300
initial_learning_rate= 0.01  # initial learning rate
final_learning_rate = 0.01 # final learning rate (lr0 * lrf))  
patience = 100   #If you want the model to train for longer, you can increase the patience

## Use the model
results = model.train(data=conf_path, 
                        epochs=number_of_epochs, 
                        project='yolov8_license_plate_detection',
                        name='model_1', # creates a new folder for each run 
                        lr0 = initial_learning_rate,  
                        lrf = final_learning_rate, 
                        patience=patience)  


## Resume training
# results = model.train(resume=True)

## Train the model with 2 GPUs
# results = model.train(data=conf_path, epochs=number_of_epochs, device=[0,1])



# -----------with checkpoints-------------
# from ultralytics import YOLO
# from torch.utils.checkpoint import Checkpoint

# # Load a model
# model = YOLO('yolov8n.yaml')  # build a new model from YAML
# model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
# model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

# conf_path = 'config.yaml'
# number_of_epochs = 100


# optimizer = torch.optim.Adam(model.parameters())

# checkpoint = Checkpoint(model, optimizer)
# for epoch in range(100):
#     # Train your model...
#     if epoch % 10 == 0:
#         checkpoint.save(f"checkpoint_{epoch}.pt")