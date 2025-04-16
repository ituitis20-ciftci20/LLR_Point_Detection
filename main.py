import subprocess

# Move segmentation model
subprocess.run(['mv', './Dataset003_FemurTibia', '../bone_seg_nnunet_main'], cwd='./SEGMENTATION/models')

# Get repo dependencies for YOLO
subprocess.run(['pip3', 'install', 'scipy==1.10.0', 'typeguard', 'menpo', 'menpofit', 'tensorflow==2.12', 'tensorflow-addons==0.20.0', 'pillow', 'scikit-image', 'scikit-learn', 'numpy', 'matplotlib', 'pandas', 'pingouin', 'seaborn', 'statsmodels', 'notebook', 'configparser'])
# Get repo dependencies for segmentation
import subprocess
subprocess.run(['pip3', 'install', 'torch==2.3', 'torchaudio==2.3', 'torchvision==0.18', 'opencv-python', 'matplotlib'])


# # export all the models from drive with curl
# def download_file(file_id, filename):
#   """Downloads a file from Google Drive using its file ID.

#   Args:
#     file_id: The ID of the file to download.
#     filename: The name of the file to save the downloaded content to.
#   """
#   url = f"https://drive.google.com/uc?export=download&id={file_id}"
#   try:
#     subprocess.run(['curl', '-L', '-o', filename, url], check=True)
#     print(f"File '{filename}' downloaded successfully.")
#   except subprocess.CalledProcessError as e:
#     print(f"Error downloading file: {e}")


# # Example usage:
# # Replace with your actual FILE_ID and desired filename
# yolo1_id = "1D-2FUgt-g1LH-Pt1-dnAB8Ys2rjfJ-0S"
# yolo2_id = "1kgtJmkIPHG8uJl-4WYSW0bxObtZCtYGy"
# yolo3_id = "10uvr4BaMPLAVS6o4TT5wUbwkpJEygATg"
# segmttn_id =
# filename = "Dataset003_FemurTibia.zip"
# download_file(segmentation_id, filename)

import os
import sys

# 1- YOLO
script_dir = os.path.join(os.getcwd(), "./YOLO/demo")

# Now, you can run your code that relies on being in the script's directory
print(f"Runing YOLO in {script_dir}")

subprocess.run(['python', 'yolo_whole.py'], cwd=script_dir)

# Wait for output file
import time
while not os.path.exists("./OUTPUT/OUTPUT_YOLO"):
    print("Waiting for YOLO output...")
    time.sleep(1)

# 2- Segmentation
# COPY TEST_IMAGES
import shutil
segmentation_dir = os.path.join(os.getcwd(),"./SEGMENTATION/bone_seg_nnunet_main")
input_images_dir = "./INPUT_IMAGES"
testing_images_dir = os.path.join(segmentation_dir, "./testing_images")
print(testing_images_dir)
# Check if input images directory exists
if os.path.exists(input_images_dir):
    for file_name in os.listdir(input_images_dir):
        file_path = os.path.join(input_images_dir, file_name)
        if os.path.isfile(file_path):
            print(file_path)
            shutil.copy(file_path, testing_images_dir)
else:
    print(f"Directory {input_images_dir} does not exist.")

# Get the script's directory
script_dir = segmentation_dir

# Now, you can run your code that relies on being in the script's directory
print(f"Runing segmentation in {script_dir}")

# give only one 
subprocess.run(['python', 'main.py', '--clear', '--id', '003'], cwd=script_dir)

script_dir = "./SEGMENTATION/mask_to_notch"
# Now, you can run your code that relies on being in the script's directory
print(f"Runing point detection in {script_dir}")

subprocess.run(['python', 'calculate_for_whole_folder2.py'], cwd=script_dir)

from read_results import read_YOLO_data, read_SEG_data, tranform_points, change_labels, store_data, get_image_name
import os

seg_folder = './OUTPUT//OUTPUT_SEGMENTATION//results'
yolo_file_path = './OUTPUT//OUTPUT_YOLO//results.txt'
input_images_path = './INPUT_IMAGES'
data = read_YOLO_data(yolo_file_path)

data.update(read_SEG_data(seg_folder,data))
# print('BEFORE: LABEL MAP AND TRANSFORM')
# print(data)

mapped_data = {}
for key, value in data.items():
   is_left = False
   if key.find('left') != -1:
      data[key]['is_left'] = True
   else:
      data[key]['is_left'] = False
   img_path = os.path.join(input_images_path, get_image_name(key)[0]+'.png')
   tranform_points(image_path=img_path, data=value)
   mapped_data[key] = change_labels(is_left=data[key]['is_left'], data_dict=value)

store_data(mapped_data)

