import os
import glob
import sys
from typing import Tuple
from pathlib import Path
from PIL import Image
from image_utils import image_class as ic

import cv2
import numpy as np

project_dir = "~/DB"
project_dir = os.path.expanduser(project_dir)
dataset = "twin_falls_palletizer_vision_system"

# image_name = "Image010821113024.png"
# img_path = os.path.join(project_dir, dataset, image_name)
# image = ic.ImageClass()
# image.read_image(img_path=img_path)

dir_path = os.path.join(project_dir, dataset)
list_of_image_paths = glob.glob(dir_path + "/*.png")

new_dir_path = Path(os.path.join(project_dir, "cropped_" + dataset))
if not new_dir_path.is_dir():
    new_dir_path.mkdir(parents=True, exist_ok=True)

image = ic.ImageClass()
for image_path in list_of_image_paths:
    print("Loading image from:{}".format(image_path))
    image_name = Path(image_path).name
    image.read_image(img_path=image_path)
    img, format = image.crop_image_top_left(top_left_coord=(505, 444))
    image.set_image(image=img, image_format=format)
    new_image_path = os.path.join(new_dir_path, image_name)
    print("Writing image to:{}".format(new_image_path))
    image.write_image(img_path=new_image_path)

# image.show_image()

# pil_img = Image.open(img_path)
# cv2_img = cv2.imread(img_path)
#
# print("pil_img:{}".format(pil_img.size))
# print("cv2_img:{}".format(cv2_img.shape))
#
# top_left_cropping_coord = (505, 444)  ## (y,x)
# cv2_img = cv2_img[top_left_cropping_coord[0] :, top_left_cropping_coord[1] :]
# cv2.namedWindow("img", cv2.WINDOW_NORMAL)
# cv2.imshow("img", cv2_img)
# cv2.resizeWindow("img", 640 * 2, 480 * 2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
#
# dir_path = os.path.join(project_dir, dataset)
# list_of_image_paths = glob.glob(dir_path + "/*.png")
# n_images = len(list_of_image_paths)
# print(Path(list_of_image_paths[0]).name)
# print(Path(list_of_image_paths[0]).parents[1])
