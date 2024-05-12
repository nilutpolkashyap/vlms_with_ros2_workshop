from pathlib import Path
import os
import urllib.request
import subprocess

current_dir = os.getcwd()

grounding_dino_config_base_path = os.path.join(current_dir, 'ros2_vlm', 'ros2_vlm', 'modules', 'checkpoints')
grounding_dino_config_name = "groundingdino_swint_ogc.pth"
file_path = os.path.join(grounding_dino_config_base_path, grounding_dino_config_name)
grounding_dino_config_link = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"

# Check if file exists
if not os.path.exists(file_path):
    print(f"Downloading {grounding_dino_config_name}...")
    # Download file
    urllib.request.urlretrieve(grounding_dino_config_link, file_path)
    print("Download complete!")
else:
    print(f"File {grounding_dino_config_name} already exists.")

ground_dino_dir = os.path.join(current_dir, 'ros2_vlm', 'ros2_vlm', 'modules', 'GroundingDINO')
subprocess.run(["git", "clone", "https://github.com/wenyi5608/GroundingDINO/", ground_dino_dir])

efficient_sam_dir = os.path.join(current_dir, 'ros2_vlm', 'ros2_vlm', 'modules', 'EfficientSAM')
subprocess.run(["git", "clone", "https://github.com/yformer/EfficientSAM/", efficient_sam_dir])



