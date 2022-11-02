import os
import sys
import time
from typing import List, Tuple, Dict

import pytesseract as pytess
import numpy as np
import cv2

from image_utils import image_class as ic
from image_utils import cv_utils as cu
from file_utils import file_utils as fu
from file_reader_inference import FileReaderInference


class TextRecognitionEngine(object):
    def __init__(self, images: List, roi_bbox_resize_fracs: Tuple):
        self.images = images
        self.roi_bbox_resize_fracs = roi_bbox_resize_fracs  # width_frac, height_frac
        self.recognized_strings = []

    def extract_text_from_images(self):
        self.resize_bbox_rois()

        t1 = time.time()
        strings = [
            pytess.image_to_string(image, config="--psm 6")  # 6works
            for image in self.images
        ]
        t2 = time.time()
        print("Time taken for recog:{}".format(t2 - t1))
        self.recognized_strings = strings
        # https://stackoverflow.com/questions/60977964/pytesseract-not-recognizing-text-as-expected
        self.put_text_on_bbox_rois(detected_strings=strings)

        return strings

    def put_text_on_bbox_rois(self, detected_strings: List):
        assert len(detected_strings) == len(
            self.images
        ), "Number of images({}) in self.images in not same({}) as detected_strings".format(
            len(self.images), len(detected_strings)
        )
        for img, string in zip(self.images, detected_strings):
            cu.draw_text_on_image(image=img, string=string)

    def apply_image_manipulations(
        self, list_of_callables: List, list_of_arguments: List[Dict]
    ):
        for i in range(len(self.images)):
            for callable, kwargs in zip(list_of_callables, list_of_arguments):
                self.images[i] = callable(self.images[i], **kwargs)
        return self


def make_text_recognition_engine(
    image_objects: List, roi_bbox_resize_fracs: Tuple, image_encoding: str
):
    if image_encoding.lower() == "gray":
        images = [img_obj.to_grayscale().image[0] for img_obj in image_objects]
    else:
        images = [img_obj.image[0] for img_obj in image_objects]

    text_recognizer_obj = TextRecognitionEngine(
        images=images,
        roi_bbox_resize_fracs=roi_bbox_resize_fracs,
    )

    return text_recognizer_obj


if __name__ == "__main__":
    inputs_path = "/home/ruthvik/DB/twin_falls_palletizer_vision_system"
    outputs_path = "/home/ruthvik/DB/demo_results"
    input_files = fu.get_list_of_files(inputs_path, file_type="*.png")
    output_files = fu.get_list_of_files(outputs_path, file_type="*.txt")
    manipulations_callables = cu.make_manipulations_sequencer(
        manipulations_sequence=["contrast_brightness", "unsharp_image"]
    )
    manipulations_arguments = [
        {"contrast_control": 1.5, "brightness_control": 10},
        {"amount": 1.25},
    ]
    # file_reader.load_a_samples_polygons(polygons_path=file_reader._output_files[0])
    for i in range(len(input_files)):
        file_reader = FileReaderInference(
            polygons_path=output_files[i],
            image_path=input_files[i],
        )
        # temp, _ = file_reader.image.image
        # print(temp.shape)
        # string = pytess.image_to_string(temp)
        # print(string)
        # sys.exit(0)
        file_reader.draw_a_samples_polygons()
        file_reader.show_polygons_on_a_sample()
        t1 = time.time()
        file_reader.crop_rois()
        # file_reader.show_rois(visualization_type="polygons")
        # file_reader.show_rois()
        text_recognizer = make_text_recognition_engine(
            image_objects=file_reader.bboxes_rois,
            roi_bbox_resize_fracs=(1.0, 1.0),
            image_encoding="gray",
        )
        text_recognizer = text_recognizer.apply_image_manipulations(
            list_of_callables=manipulations_callables,
            list_of_arguments=manipulations_arguments,
        )

        detected_text = text_recognizer.extract_text_from_images()
        t2 = time.time()
        print("Time taken to run one recognition:{}".format(t2 - t1))
        print(detected_text)
        file_reader.show_rois()
        cv2.waitKey(0)
        # sys.exit(0)
