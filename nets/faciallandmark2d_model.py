import math
import cv2
import numpy as np

from nets.base_model import BaseModel
# from utils import calculate_angle_distance

class FacialLandmark2DModel(BaseModel):
    def __init__(self, model_path=None, apply_all_optim=True, device="cpu", scale=1.2):
        super().__init__(model_path=model_path, apply_all_optim=apply_all_optim)#, device=device)
        self.scale = scale
        # self.input_size = (112, 112)
        
        self.mean = np.array([123.675, 116.28, 103.53], np.float64).reshape(1, -1)
        self.std = np.array([58.395, 57.12, 57.375], np.float64).reshape(1, -1)

        self.load_model(device)

    def preprocess(self, frame, bbox):
        # Pre-process the input
        x_min = bbox[0]
        y_min = bbox[1]
        x_max = bbox[2]
        y_max = bbox[3]

        box_w = x_max - x_min
        box_h = y_max - y_min
        
        c_x = (x_min + x_max) // 2
        c_y = (y_min + y_max) // 2
        
        if box_h <= box_w:
            pad = box_w // 2
            box_h = box_w
        else:
            pad = box_h // 2
            box_w = box_h

        x_min = max(0, c_x - pad)
        x_max = min(frame.shape[1] - 1, c_x + pad)
        y_min = max(0, c_y - pad)
        y_max = min(frame.shape[0] - 1, c_y + pad)

        # # Remove a part of top area for alignment, see paper for details
        # x_min -= int(box_w * (self.scale - 1) / 2)
        # y_min += int(box_h * (self.scale - 1) / 2)
        # x_max += int(box_w * (self.scale - 1) / 2)
        # y_max += int(box_h * (self.scale - 1) / 2)

        # x_min = max(x_min, 0)
        # y_min = max(y_min, 0)
        # x_max = min(x_max, frame.shape[1] - 1)
        # y_max = min(y_max, frame.shape[0] - 1)

        # box_w = x_max - x_min + 1
        # box_h = y_max - y_min + 1
        
        # if box_

        image = frame[y_min:y_max, x_min:x_max, :]
        ###print("shape:", image.shape)
        image = cv2.resize(image, (self.input_width, self.input_height))
        image = image.astype(np.float32)

        cv2.cvtColor(image, cv2.COLOR_BGR2RGB, image)   # inplace
        cv2.subtract(image, self.mean, image)           # inplace
        cv2.multiply(image, 1 / self.std, image)        # inplace

        image = image.transpose((2, 0, 1))[np.newaxis, ...]
        image = np.ascontiguousarray(image)
        
        return image, x_min, y_min, box_w, box_h     
    
    def postprocess(self, output, x_min, y_min, box_w, box_h):
        landmarks_2D = output[0].reshape(-1, 3).astype(np.float32)
        landmarks_2D[:, 0] = landmarks_2D[:, 0] * box_w + x_min
        landmarks_2D[:, 1] = landmarks_2D[:, 1] * box_h + y_min
        # print("landmarks_2D:", landmarks_2D)
        visibility = 1 / (1 + np.exp(-landmarks_2D[:, 2]))
        landmarks_2D[:, 2] = np.round(visibility)
        return landmarks_2D
    
    def pipeline(self, frame, bbox):
        """
        Process the input frame and bounding box to extract 2D facial landmarks.
        
        Args:
            frame (numpy.ndarray): The input image frame.
            bbox (list): Bounding box coordinates [x_min, y_min, x_max, y_max].
        
        Returns:
            numpy.ndarray: Detected 2D facial landmarks as shape (N, 3), where N is the number of landmarks.
            3: x, y, visibility for each landmark.
        """
        image, x_min, y_min, box_w, box_h = self.preprocess(frame, bbox)
        output = self.predict(image)
        landmarks_2D = self.postprocess(output, x_min, y_min, box_w, box_h)
        return landmarks_2D

LEFT_EAR = 21
LEFT_EYE = 7
RIGHT_EAR = 22
RIGHT_EYE = 14
NOSE = 0

