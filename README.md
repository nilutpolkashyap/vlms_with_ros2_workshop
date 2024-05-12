# Application of Vision Language Models with ROS 2 workshop

## Create a ROS 2 colcon workspace

```
mkdir -p ~/ros2_ws/src
```

## Create & Setup Python Virtual Environment 
```
cd ~/ros2_ws

virtualenv -p python3 ./vlm-venv                      
source ./vlm-venv/bin/activate

# Make sure that colcon doesn’t try to build the venv
touch ./vlm-venv/COLCON_IGNORE        
```

## Install Python dependencies
```
pip install timm --extra-index-url https://download.pytorch.org/whl/cpu  # is needed for torch

pip install "openvino>=2024.1" "torch>=2.1" opencv-python supervision transformers yapf pycocotools addict "gradio>=4.19" tqdm
```

## Add your Python virtual environment package path 
**Make sure to update <<YOUR_USER_NAME>> with your system username.**
```
export PYTHONPATH='/home/<<YOUR_USER_NAME>>/ros2_ws/vlm-venv/lib/python3.10/site-packages'
```

## Clone this repository inside the 'src' folder of your workspace
```
cd ~/ros2_ws/src

git clone https://github.com/nilutpolkashyap/vlms_with_ros2_workshop.git
```

## Download weights and required packages
```
cd ~/ros2_ws/src/vlms_with_ros2_workshop

python3 download_weights.py
```

## Download OpenVINO IR models from Google Drive

Download the zip file from here [link here]()

Place the contents of the zip file inside the **'openvino_irs'** directory in following path

```
~/ros2_ws/src/vlms_with_ros2_workshop/ros2_vlm/ros2_vlm/modules/openvino_irs
```

## Build and source the workspace 
```
cd ~/ros2_ws
colcon build --symlink-install

source ~/ros2_ws/install/setup.bash
```

## Run the GroundedSAM (GroundingDINO + SAM)  node
```
cd ~/ros2_ws

ros2 run ros2_vlm grounded_sam --ros-args -p video_source:=/dev/video2 -p isSegment:=False -p detectionList:="["spectacles", "person", "hair"]"
```

## Resources
- [Using Python Packages with ROS 2](https://docs.ros.org/en/humble/How-To-Guides/Using-Python-Packages.html)
- [How to use (python) virtual environments with ROS2?
](https://answers.ros.org/question/371083/how-to-use-python-virtual-environments-with-ros2/)
- [grounded-segment-anything.ipynb](https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/grounded-segment-anything/grounded-segment-anything.ipynb)
