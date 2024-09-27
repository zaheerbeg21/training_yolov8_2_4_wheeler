# train yolov8 on ANPR dataset

## environment 
python -m venv env
cd ~/env
source bin/activate
source ./activate

# download sort

wget git clone https://github.com/abewley/sort.git

#install in ubuntu
apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
