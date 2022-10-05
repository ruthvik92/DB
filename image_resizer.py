import os
import glob
import sys
from typing import Tuple
from pathlib import Path
from PIL import Image

import cv2

project_dir = "~/DB"
project_dir = os.path.expanduser(project_dir)
dataset = "twin_falls_palletizer_vision_system"
image_name = "Image010821113024.png"

img_path = os.path.join(project_dir, dataset, image_name)
pil_img = Image.open(img_path)
cv2_img = cv2.imread(img_path)

print("pil_img:{}".format(pil_img.size))
print("cv2_img:{}".format(cv2_img.shape))

top_left_cropping_coord = (505, 444)  ## (y,x)
cv2_img = cv2_img[top_left_cropping_coord[0] :, top_left_cropping_coord[1] :]
cv2.namedWindow("img", cv2.WINDOW_NORMAL)
cv2.imshow("img", cv2_img)
cv2.resizeWindow("img", 640 * 2, 480 * 2)
cv2.waitKey(0)
cv2.destroyAllWindows()


dir_path = os.path.join(project_dir, dataset)
list_of_image_paths = glob.glob(dir_path + "/*.png")
n_images = len(list_of_image_paths)
print(Path(list_of_image_paths[0]).name)
print(Path(list_of_image_paths[0]).parents[1])

sys.exit()


def resize_image(img_path: str = "", top_left_cropping_coord: Tuple = (505, 444)):
    img_name = Path(img_path).name
    bgr_image = cv2.imread(img_path)
    bgr_image = bgr_image[top_left_cropping_coord[0] :, top_left_cropping_coord[1] :]
    cv2.namedWindow("cropped_img", cv2.WINDOW_NORMAL)
    cv2.imshow("cropped_img", bgr_image)
    cv2.resizeWindow("cropped_img", 640 * 2, 480 * 2)
    cv2.waitKey(0)
    cv2.imwrite()
