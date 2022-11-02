from typing import Tuple

import cv2
import numpy as np
from PIL import Image


class ImageClass(object):
    """Image class
    Intended to store images in numpy arrays and associated
    image format information. If you do not enclose your image
    (numpy array) its format (RGB or BGR) then down the line
    there is no way to tell if an image is RGB or BGR
    """

    def __init__(self):
        self.loading_engine = "opencv"
        self.__image_format = "BGR"
        self.__image = None
        self.__image_height = None
        self.__image_width = None

    def read_image(self, img_path: str, loading_engine: str = "opencv", **kwargs):
        """Read an image using cv2.imread

        Parameters
        ----------
        img_path : str
            Path to the image that needs to be loaded
        """
        if loading_engine.lower() == "opencv":
            self.__image = cv2.imread(img_path, **kwargs)
            self.__image_format = "BGR"
        elif loading_engine.lower() == "pil":
            self.__image = Image.open(img_path, **kwargs)
            self.__image_format = "RGB"
        self.calc_image_dimensions()

    def calc_image_dimensions(self):
        if self.loading_engine.lower() == "opencv":
            self.__image_height, self.__image_width = (
                self.__image.shape[0],
                self.__image.shape[1],
            )
        elif self.loading_engine.lower() == "pil":
            self.__image_width, self.__image_height = self.__image.size

    @property
    def dimensions(self):
        return self.__image_height, self.__image_width

    @property
    def image(self):
        """Get access to the loaded image

        Returns
        -------
        np.ndarray
            Loaded image in numpy array format.
        """
        return self.__image, self.__image_format

    @image.setter
    def image(self, image_and_format: Tuple[np.ndarray, str]):
        """Set the image value if you happen to modify the loaded
        image.

        Parameters
        ----------
        image : np.ndarray
            Image in numy array format.
        image_format : str
            Is the image 'RGB' or 'BGR'?
        """
        image, image_format = image_and_format
        self.__image = image
        self.__image_format = image_format
        self.calc_image_dimensions()

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

    def to_grayscale(self):
        if (self.__image_format).lower() == "bgr":
            self.__image = cv2.cvtColor(self.__image, cv2.COLOR_BGR2GRAY)
        elif (self.__image_format).lower() == "rgb":
            self.__image = cv2.cvtColor(self.__image, cv2.COLOR_RGB2GRAY)
        elif (self.__image_format).lower() == "gray":
            pass
        self.__image_format = "GRAY"
        self.calc_image_dimensions()
        return self

    def show_image(
        self, window_name: str = "self.__image", window_size: Tuple = (640 * 2, 480 * 2)
    ):
        """Display a loaded image"""
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, self.__image)
        cv2.resizeWindow(window_name, window_size[0], window_size[1])
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # cv2.destroyWindow("image")
        # cv2.waitKey(1)

    def write_image(self, img_path: str):
        """Write an image to disk given a path.

        Parameters
        ----------
        img_path : str
            Path to which image is to be written
        """
        cv2.imwrite(img_path, self.__image)
