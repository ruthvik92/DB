import os
import sys
import time
from typing import List, Tuple

import pytesseract as pytess
import numpy as np
import cv2

from image_utils import image_class as ic
from image_utils import cv_utils as cu
from file_utils import file_utils as fu
from file_reader_inference import FileReaderInference


def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened


class TextRecognitionEngine(object):
    def __init__(self, images: List):
        self.images = images
        self.roi_bbox_size = (1.0, 1.0)  # width_frac, height_frac
        self.recognized_strings = []

    def extract_text_from_images(self):
        self.resize_bbox_rois()
        rgb_images = []
        gray_images = []
        for img_obj in self.images:
            bgr_img, _ = img_obj.image
            rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
            # print(bgr_img.shape)
            gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
            rgb_images.append(rgb_img)
            gray_images.append(gray_img)
            # print(gray_img.shape)

        t1 = time.time()
        strings = [
            pytess.image_to_string(gray_image, config="--psm 6")  # 6works
            for gray_image in gray_images
        ]
        t2 = time.time()
        print("Time taken for just GRAY recog:{}".format(t2 - t1))
        # https://stackoverflow.com/questions/60977964/pytesseract-not-recognizing-text-as-expected
        self.put_text_on_bbox_rois(detected_strings=strings)

        return strings

    def put_text_on_bbox_rois(self, detected_strings: List):
        assert len(detected_strings) == len(
            self.images
        ), "Number of images({}) in self.images in not same({}) as detected_strings".format(
            len(self.images), len(detected_strings)
        )
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.5
        fontColor = (0, 255, 0)
        thickness = 1
        lineType = 1
        for img_obj, text in zip(self.images, detected_strings):
            bgr_img, _ = img_obj.image
            height, width = bgr_img.shape[0:2]
            bgr_img = cv2.putText(
                bgr_img,
                text,
                (width // 2, height // 2),
                font,
                fontScale,
                fontColor,
                thickness,
                lineType,
            )
            img_obj.image = (bgr_img, "BGR")

    def resize_bbox_rois(self):
        width, height = self.roi_bbox_size
        for img_obj in self.images:
            src_bgr_img, enc = img_obj.image
            # dst_bgr_img = cv2.resize(
            #    src_bgr_img, (width, height), interpolation=cv2.INTER_CUBIC
            # )
            src_bgr_img = cv2.resize(
                src_bgr_img, None, fx=width, fy=height, interpolation=cv2.INTER_CUBIC
            )
            # kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            # dst_bgr_img = cv2.filter2D(dst_bgr_img, -1, kernel)
            # dst_bgr_img = unsharp_mask(image=dst_bgr_img)
            alpha = 1.5  # Contrast control (1.0-3.0)
            beta = 10  # Brightness control (0-100)
            dst_bgr_img = cv2.convertScaleAbs(src_bgr_img, alpha=alpha, beta=beta)
            dst_bgr_img = unsharp_mask(image=dst_bgr_img, amount=1)
            # dst_bgr_img = cv2.threshold(dst_bgr_img, 100, 255, cv2.THRESH_BINARY)[1]
            img_obj.image = (dst_bgr_img, enc)
            # img_obj.image = (src_bgr_img, enc)


if __name__ == "__main__":
    inputs_path = "/home/ruthvik/DB/twin_falls_palletizer_vision_system"
    outputs_path = "/home/ruthvik/DB/demo_results"
    input_files = fu.get_list_of_files(inputs_path, file_type="*.png")
    output_files = fu.get_list_of_files(outputs_path, file_type="*.txt")
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
        text_recognizer = TextRecognitionEngine(image_objects=file_reader.bboxes_rois)
        detected_text = text_recognizer.extract_text_from_images()
        t2 = time.time()
        print("Time taken to run one recognition:{}".format(t2 - t1))
        print(detected_text)
        file_reader.show_rois()
        cv2.waitKey(0)
        # sys.exit(0)
