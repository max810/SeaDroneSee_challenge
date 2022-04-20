import os
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import cv2

import warnings
warnings.filterwarnings("ignore")

from sahi.model import Yolov5DetectionModel
from sahi.utils.cv import read_image
from sahi.utils.file import download_from_url
from sahi.predict import get_prediction, get_sliced_prediction, predict
from sahi.model import DetectionModel

from typing import Dict, List, Optional, Union
from sahi.utils.compatibility import fix_full_shape_list, fix_shift_amount_list
from sahi.prediction import ObjectPrediction

from IPython.display import Image
import numpy as np
import sys
import cv2
import torch
from PIL import Image as Img
from IPython.display import display


class MyDetectionModel(DetectionModel):
    def load_model(self):
        """
        Detection model is initialized and set to self.model.
        """
        try:
            import yolov5
        except ImportError:
            raise ImportError('Please run "pip install -U yolov5" ' "to install YOLOv5 first for YOLOv5 inference.")

        # set model
        try:
            #model = yolov5.load(self.model_path, device=self.device)
            model = yolov5.load(self.model_path, device=self.device)
            model.conf = self.confidence_threshold
            self.model = model
        except Exception as e:
            raise TypeError("model_path is not a valid yolov5 model path: ", e)

        # set category_mapping
        if not self.category_mapping:
            category_mapping = {str(ind): category_name for ind, category_name in enumerate(self.category_names)}
            self.category_mapping = category_mapping
            print(category_mapping)

    def perform_inference(self, image: np.ndarray, image_size: int = None):
        """
        Prediction is performed using self.model and the prediction result is set to self._original_predictions.
        Args:
            image: np.ndarray
                A numpy array that contains the image to be predicted. 3 channel image should be in RGB order.
            image_size: int
                Inference input size.
        """
        try:
            import yolov5
        except ImportError:
            raise ImportError('Please run "pip install -U yolov5" ' "to install YOLOv5 first for YOLOv5 inference.")

        # Confirm model is loaded
        assert self.model is not None, "Model is not loaded, load it by calling .load_model()"

        if image_size is not None:
            warnings.warn("Set 'image_size' at DetectionModel init.", DeprecationWarning)
            prediction_result = self.model(image, size=image_size)
        elif self.image_size is not None:
            prediction_result = self.model(image, size=self.image_size)
        else:
            prediction_result = self.model(image)

        self._original_predictions = prediction_result

    @property
    def num_categories(self):
        """
        Returns number of categories
        """
        return len(self.model.names)

    @property
    def has_mask(self):
        """
        Returns if model output contains segmentation mask
        """
        has_mask = self.model.with_mask
        return has_mask

    @property
    def category_names(self):
        return self.model.names

    def _create_object_prediction_list_from_original_predictions(
        self,
        shift_amount_list: Optional[List[List[int]]] = [[0, 0]],
        full_shape_list: Optional[List[List[int]]] = None,
    ):
        """
        self._original_predictions is converted to a list of prediction.ObjectPrediction and set to
        self._object_prediction_list_per_image.
        Args:
            shift_amount_list: list of list
                To shift the box and mask predictions from sliced image to full sized image, should
                be in the form of List[[shift_x, shift_y],[shift_x, shift_y],...]
            full_shape_list: list of list
                Size of the full image after shifting, should be in the form of
                List[[height, width],[height, width],...]
        """
        original_predictions = self._original_predictions

        # compatilibty for sahi v0.8.15
        shift_amount_list = fix_shift_amount_list(shift_amount_list)
        full_shape_list = fix_full_shape_list(full_shape_list)

        # handle all predictions
        object_prediction_list_per_image = []
        for image_ind, image_predictions_in_xyxy_format in enumerate(original_predictions.xyxy):
            shift_amount = shift_amount_list[image_ind]
            full_shape = None if full_shape_list is None else full_shape_list[image_ind]
            object_prediction_list = []

            # process predictions
            for prediction in image_predictions_in_xyxy_format.cpu().detach().numpy():
                x1 = int(prediction[0])
                y1 = int(prediction[1])
                x2 = int(prediction[2])
                y2 = int(prediction[3])
                bbox = [x1, y1, x2, y2]
                score = prediction[4]
                category_id = int(prediction[5])
                category_name = self.category_mapping[str(category_id)]

                # fix negative box coords
                bbox[0] = max(0, bbox[0])
                bbox[1] = max(0, bbox[1])
                bbox[2] = max(0, bbox[2])
                bbox[3] = max(0, bbox[3])

                # fix out of image box coords
                if full_shape is not None:
                    bbox[0] = min(full_shape[1], bbox[0])
                    bbox[1] = min(full_shape[0], bbox[1])
                    bbox[2] = min(full_shape[1], bbox[2])
                    bbox[3] = min(full_shape[0], bbox[3])

                # ignore invalid predictions
                if not (bbox[0] < bbox[2]) or not (bbox[1] < bbox[3]):
                    logger.warning(f"ignoring invalid prediction with bbox: {bbox}")
                    continue

                object_prediction = ObjectPrediction(
                    bbox=bbox,
                    category_id=category_id,
                    score=score,
                    bool_mask=None,
                    category_name=category_name,
                    shift_amount=shift_amount,
                    full_shape=full_shape,
                )
                object_prediction_list.append(object_prediction)
            object_prediction_list_per_image.append(object_prediction_list)

        self._object_prediction_list_per_image = object_prediction_list_per_image

#model_type = "yolov5"
MODEL_PATH ='/home/sara.pieri/Documents/SeaDroneSee_challenge/runs/train/v5l-extra-head/weights/last.pt'
model_device = "cuda" # or 'cuda:0'
model_confidence_threshold = 0.4
slice_height = 256
slice_width = 256
overlap_height_ratio = 0.2
overlap_width_ratio = 0.2
perform_standard_pred = True
postprocess_match_threshold = 0.5
image_size = 256
verbose = 2

detection_model = MyDetectionModel(
   model_path = MODEL_PATH,
   confidence_threshold = 0.25,
   device="cuda",
)

results = predict(
    model_type='yolov5',
    model_path=MODEL_PATH,
    model_device=model_device,
    model_confidence_threshold=model_confidence_threshold,
    source= "/home/sara.pieri/Documents/datasets/SeaDroneSee/val/images",
    slice_height=slice_height,
    slice_width=slice_width,
    overlap_height_ratio=overlap_height_ratio,
    overlap_width_ratio=overlap_width_ratio,
    return_dict=True,
    export_pickle=True,
    export_crop = True, 
    dataset_json_path  = '/home/sara.pieri/Documents/datasets/SeaDroneSee/val.json',
    project = 'yolov-extra-head',
    name = 'exp',
    verbose = 2
)

#!sahi coco evaluate --dataset_json_path /home/sara.pieri/Documents/datasets/SeaDroneSee/val.json --result_json_path /home/sara.pieri/Documents/SeaDroneSee_challenge/notebooks/yolov-extra-head/exp15/result.json
