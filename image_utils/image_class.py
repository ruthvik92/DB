from typing import Tuple

import cv2
import numpy as np


class ImageClass(object):
    """Image class
    Intended to store images in numpy arrays and associated
    image format information. If you do not enclose your image
    (numpy array) its format (RGB or BGR) then down the line
    there is no way to tell if an image is RGB or BGR
    """

    def __init__(self):
        self.__image_format = "BGR"
        self.__image = None

    def read_image(self, img_path: str, **kwargs):
        """Read an image using cv2.imread

        Parameters
        ----------
        img_path : str
            Path to the image that needs to be loaded
        """
        self.__image = cv2.imread(img_path, **kwargs)

    def get_image(self):
        """Get access to the loaded image

        Returns
        -------
        np.ndarray
            Loaded image in numpy array format.
        """
        return self.__image

    def set_image(self, image: np.ndarray, image_format: str):
        """Set the image value if you happen to modify the loaded
        image.

        Parameters
        ----------
        image : np.ndarray
            Image in numy array format.
        image_format : str
            Is the image 'RGB' or 'BGR'?
        """
        self.__image = image
        self.__image_format = image_format

    def crop_image_top_left(self, top_left_coord: Tuple = (505, 444)):
        """Crop an image given top left coordinates

        Parameters
        ----------
        top_left_coord : Tuple
            Give top left (y,x) and everything to the right and
            below will be included in the cropped image, by default (505, 444)

        Returns
        -------
        (np.ndarray, str)
            Returns a tuple of cropped image and its format.
        """
        cropped_image = self.__image[top_left_coord[0] :, top_left_coord[1] :]
        return cropped_image, self.__image_format

    def show_image(self, window_name="self.__image"):
        """Display a loaded image"""
        # cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, self.__image)
        # cv2.resizeWindow(window_name, 640 * 2, 480 * 2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # cv2.destroyWindow("image")
        cv2.waitKey(1)

    def write_image(self, img_path: str):
        """Write an image to disk given a path.

        Parameters
        ----------
        img_path : str
            Path to which image is to be written
        """
        cv2.imwrite(img_path, self.__image)
