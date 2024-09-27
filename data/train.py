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
results = model.train(data=conf_path, epochs=number_of_epochs)  # train the model

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