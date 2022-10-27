from typing import List, Tuple
from copy import deepcopy
from itertools import compress
import pathlib
import os
import sys

import cv2
import numpy as np

from image_utils import image_class as ic
from image_utils import cv_utils as cu
from file_utils import file_utils as fu


class FileReaderInference(object):
    def __init__(self, polygons_path: str, image_path: str):
        self.polygons_path = polygons_path
        self.image_path = image_path
        self.image = ic.ImageClass()
        self.image.read_image(img_path=image_path)
        self.polygons = []
        self.scores = []
        self.load_a_samples_polygons()
        self.polygon_sub_images = [
            ic.ImageClass() for i in range(len(self.polygons))
        ]  # full images make this option with a config system
        self.bboxes_sub_images = [
            ic.ImageClass() for i in range(len(self.polygons))
        ]  # cropped images

    def load_a_samples_polygons(self):
        polygons_scores = fu.read_text_file(self.polygons_path)
        polygons_scores = [line.split(",") for line in polygons_scores]
        polygons_scores = [
            [float(vertex) for vertex in polygon] for polygon in polygons_scores
        ]
        self.polygons = [polygon_score[0:-1] for polygon_score in polygons_scores]
        self.scores = [polygon_score[-1] for polygon_score in polygons_scores]

    def draw_a_samples_polygons(self):
        image, _ = self.image.image
        image = cu.draw_polygon_on_image(image=image, polygons=self.polygons)
        self.image.image = (image, "BGR")

    def show_polygons_on_a_sample(self):
        self.image.show_image(window_name="full_image")

    def crop_sub_images(self):
        height, width = self.image.dimensions
        masks = cu.get_masks_from_polygon(
            image_dimensions=(height, width), polygons=self.polygons
        )
        image, _ = self.image.image

        roi_polygons = list(
            map(lambda x: x[:, :, np.newaxis] * image, masks)
        )  # full images with only ROIs
        bounding_rects = [
            cv2.boundingRect(np.array(polygon, dtype=np.int32).reshape(-1, 2))
            for polygon in self.polygons
        ]  # returns [(x,y,w,h)...()] use img[y:y+h, x:x+w]
        print(bounding_rects)

        roi_cropped_bboxes = [
            image[item[1] : item[1] + item[3], item[0] : item[0] + item[2]]
            for item in bounding_rects
        ]
        cv2.namedWindow("temp", cv2.WINDOW_NORMAL)
        # cv2.imshow("temp", roi_polygons[0])
        cv2.imshow("temp", roi_cropped_bboxes[1])
        cv2.resizeWindow("temp", 640, 480)
        cv2.waitKey(0)
        pass


if __name__ == "__main__":
    inputs_path = "/home/ruthvik/DB/twin_falls_palletizer_vision_system"
    outputs_path = "/home/ruthvik/DB/demo_results"
    input_files = fu.get_list_of_files(inputs_path, file_type="*.png")
    output_files = fu.get_list_of_files(outputs_path, file_type="*.txt")
    # file_reader.load_a_samples_polygons(polygons_path=file_reader._output_files[0])
    for i in range(len(input_files)):
        file_reader = FileReaderInference(
            polygons_path=output_files[i], image_path=input_files[i]
        )
        # file_reader.draw_a_samples_polygons()
        # file_reader.show_polygons_on_a_sample()
        file_reader.crop_sub_images()
        sys.exit(0)
        cv2.waitKey(0)
        sys.exit()
