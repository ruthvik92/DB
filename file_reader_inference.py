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
    def __init__(self):
        self.image = ic.ImageClass()

    def load_a_samples_polygons(self, polygons_path: str):
        polygons_scores = fu.read_text_file(polygons_path)
        polygons_scores = [line.split(",") for line in polygons_scores]
        polygons_scores = [
            [float(vertex) for vertex in polygon] for polygon in polygons_scores
        ]
        polygons = [polygon_score[0:-1] for polygon_score in polygons_scores]
        scores = [polygon_score[-1] for polygon_score in polygons_scores]
        return polygons, scores

    def draw_a_samples_polygons(self, polygons_path: str, image_path: str):
        polygons, scores = self.load_a_samples_polygons(polygons_path=polygons_path)
        self.image.read_image(img_path=image_path)
        image, _ = self.image.get_image()
        image = cu.draw_polygon_on_image(image=image, polygons=polygons)
        self.image.set_image(image=image, image_format="BGR")

    def show_polygons_on_a_sample(self):
        self.image.show_image()
        cv2.waitKey(0)


if __name__ == "__main__":
    inputs_path = "/home/ruthvik/DB/twin_falls_palletizer_vision_system"
    outputs_path = "/home/ruthvik/DB/demo_results"
    input_files = fu.get_list_of_files(inputs_path, file_type="*.png")
    output_files = fu.get_list_of_files(outputs_path, file_type="*.txt")
    file_reader = FileReaderInference()
    # file_reader.load_a_samples_polygons(polygons_path=file_reader._output_files[0])
    for i in range(len(input_files)):
        file_reader.draw_a_samples_polygons(
            polygons_path=output_files[i],
            image_path=input_files[i],
        )
        file_reader.show_polygons_on_a_sample()
