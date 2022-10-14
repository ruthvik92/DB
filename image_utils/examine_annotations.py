import cv2
import numpy as np
from image_class import ImageClass

path = "/home/ruthvik/DB/datasets/TD_TR/TD500/test_images/IMG_0059.JPG"
coords = [303, 642, 1534, 708, 1528, 819, 297, 753]
coords = np.array(coords).reshape(-1, 2)
coords1 = [220, 726, 1702, 876, 1684, 1053, 202, 903]
coords1 = np.array(coords1)
coords1 = np.array(coords1).reshape(-1, 2)

image = cv2.imread(path)

# pts = pts.reshape((-1, 1, 2))

isClosed = True
# Blue color in BGR
color = (255, 0, 0)
# Line thickness of 2 px
thickness = 2

# Using cv2.polylines() method
# Draw a Blue polygon with
# thickness of 1 px
image = cv2.polylines(image, [coords], isClosed, color, thickness)
color = (0, 255, 0)
image = cv2.polylines(image, [coords1], isClosed, color, thickness)

img = ImageClass()
img.set_image(image=image, image_format="BGR")
img.show_image()
# cv2.imshow("image", image)
# cv2.waitKey(0)
