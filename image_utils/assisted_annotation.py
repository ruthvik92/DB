from typing import List, Tuple
from copy import deepcopy
from itertools import compress

import cv2
import numpy as np


hand_annot_polygon = []
model_annot_polygon = []


def on_mouse_double_clicks(event, x, y, flags, params):
    # https://stackoverflow.com/questions/28327020/opencv-detect-mouse-position-clicking-over-a-picture
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img, (x, y), 100, (255, 0, 0), -1)
        hand_annot_polygon.append((y, x))


# https://medium.com/analytics-vidhya/handling-mouse-events-in-open-cv-part-3-3dfdd59ab2f6


class AiAssistedAnnotations(object):
    # https://stackoverflow.com/questions/60587273/drawing-a-line-on-an-image-using-mouse-clicks-with-python-opencv-library
    def __init__(self, original_image, ai_annotated_polygons: List[Tuple]):
        self.original_image = original_image
        self.cloned_image = self.make_clone_of_original()

        cv2.namedWindow("image", cv2.WINDOW_NORMAL)
        # cv2.resizeWindow("image", 640 * 2, 480 * 2)
        cv2.setMouseCallback("image", self.extract_coordinates)

        # List to store start/end points
        self.hand_annot_instance_polygon = []
        self.hand_annot_image_polygons = []
        self.ai_annotated_polygons = ai_annotated_polygons
        self.nth_instance = 0
        self.isClosed = True
        # Blue color in BGR
        self.hand_color = (255, 0, 0)
        self.ai_color = (0, 0, 255)
        # Line thickness of 2 px
        self.thickness = 2

    # def load_image(self, image_path: str):
    #    image = cv2.imload(image_path)
    #    img = ImageClass()
    #    img.set_image(image=image, image_format="BGR")
    #    self._original_image = img
    #    self._cloned_image = deepcopy(self._original_image)

    def show_ai_predictions(self):
        cloned_image, _ = self.cloned_image.get_image()
        print(cloned_image.shape)
        for coords in self.ai_annotated_polygons:
            cloned_image = cv2.polylines(
                cloned_image, [coords], self.isClosed, self.ai_color, self.thickness
            )
        self.cloned_image.set_image(image=cloned_image, image_format="BGR")
        # self.cloned_image.show_image(window_name="image")
        self.show_image(image=cloned_image, window_name="image")
        # cv2.imshow("image", cloned_image)
        # cv2.resizeWindow("image", 640, 480)
        cv2.waitKey(0)

    def make_clone_of_original(self):
        return deepcopy(self.original_image)

    def show_image(self, image: np.array, window_name="image"):
        """Display a loaded image"""
        # cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, image)
        cv2.resizeWindow(window_name, 640 * 2, 480 * 2)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # cv2.destroyWindow("image")
        # cv2.waitKey(1)

    def check_if_point_in_poly(self, point: Tuple):
        point_in_ai_polygons = []
        point_in_hand_polygons = []
        for item in self.ai_annotated_polygons:
            dist = cv2.pointPolygonTest(item, point, True)
            point_in_ai_polygons.append(dist)

        for item in self.hand_annot_image_polygons:
            dist = cv2.pointPolygonTest(item, point, True)
            point_in_hand_polygons.append(dist)
        return point_in_ai_polygons, point_in_hand_polygons

    def extract_coordinates(self, event, x, y, flags, parameters):
        # Record starting (x,y) coordinates on left mouse button click
        cloned_image, _ = self.cloned_image.get_image()
        if event == cv2.EVENT_LBUTTONDOWN:
            # print(10 * "$")
            # print([x, y])
            self.hand_annot_instance_polygon.append([x, y])
            cloned_image = cv2.circle(cloned_image, (x, y), 3, (255, 0, 0), -1)
            self.cloned_image.set_image(image=cloned_image, image_format="BGR")
            # self.cloned_image.show_image(window_name="image")
            self.show_image(image=cloned_image)

        # Record ending (x,y) coordintes on left mouse bottom release
        elif event == cv2.EVENT_LBUTTONDBLCLK:
            self.hand_annot_instance_polygon.append([x, y])
            self.hand_annot_image_polygons.append(
                self.hand_annot_instance_polygon.reshape(-1, 2)
            )
            self.hand_annot_instance_polygon = []
            print(
                "Starting: {}, Ending: {}".format(
                    self.hand_annot_instance_polygon[0],
                    self.hand_annot_instance_polygon[-1],
                )
            )
            cloned_image = cv2.polylines(
                cloned_image,
                [np.array(self.hand_annot_instance_polygon).reshape(-1, 2)],
                self.isClosed,
                self.hand_color,
                self.thickness,
            )
            self.cloned_image.set_image(image=cloned_image, image_format="BGR")
            # self.cloned_image.show_image(window_name="image")
            self.show_image(image=cloned_image)

            # Draw line

        # Clear drawing boxes on right mouse button click
        elif event == cv2.EVENT_RBUTTONDOWN:
            point = x, y
            point_in_ai_poly, point_in_hand_poly = self.check_if_point_in_poly(
                point=point
            )

        def update_polygons(point_in_ai_poly: List, point_in_hand_poly: List):
            new_clone = self.make_clone_of_original()
            self.cloned_image = new_clone
            new_cloned_image, _ = new_clone.get_image()
            bool_point_in_ai_poly = [item == 1.0 for item in point_in_ai_poly]
            bool_point_in_hand_poly = [item == 1.0 for item in point_in_hand_poly]
            self.ai_annotated_polygons = compress(
                self.ai_annotated_polygons, bool_point_in_ai_poly
            )
            self.hand_annot_image_polygons = compress(
                self.hand_annot_image_polygons, bool_point_in_hand_poly
            )
            self.show_ai_predictions()

    # def show_image(self):
    #    return self.clone


if __name__ == "__main__":
    draw_line_widget = DrawLineWidget()
    while True:
        cv2.imshow("image", draw_line_widget.show_image())
        key = cv2.waitKey(1)

        # Close program with keyboard 'q'
        if key == ord("q"):
            cv2.destroyAllWindows()
            exit(1)
