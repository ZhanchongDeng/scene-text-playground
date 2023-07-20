import cv2
from cv2.dnn import TextDetectionModel_DB
import argparse
import numpy as np
import easyocr

class DetectionModel:
    def detect(self, image):
        raise NotImplementedError
    
class DBOpenCV(DetectionModel):
    def __init__(self, detection_model_path):
        detection_model = TextDetectionModel_DB(detection_model_path)
        # Post-processing parameters
        binThreshold = 0.3
        polyThresh = 0.5
        maxCandidates = 200
        unclipRatio = 2.0
        detection_model.setBinaryThreshold(binThreshold)
        detection_model.setPolygonThreshold(polyThresh)
        detection_model.setMaxCandidates(maxCandidates)
        detection_model.setUnclipRatio(unclipRatio)
        # Normalization parameters
        scale = 1/255
        size = (736, 736)
        mean = [122.67891434, 116.66876762, 104.00698793]
        detection_model.setInputParams(scale=scale, size = size, mean = mean)

        self.detection_model = detection_model

    def detect(self, image):
        results = self.detection_model.detect(image)
        vertices = results[0]
        confidence = results[1]
        return vertices
    
class DBEasyOCR(DetectionModel):
    def __init__(self, detection_model_path = None):
        self.detection_model = easyocr.Reader(['en'], False, "easyocr_models/", detect_network='dbnet18', recognizer = False, detector = True)

    def detect(self, image):
        '''Detection from EasyOCR.
        
        Args:
            image (numpy.ndarray): Image to be detected.
            
        Returns:
            horizontal_list:    List of bounding boxes. Each box is [xmin, xmax, ymin, ymax]
            free_list:          List of free points. Each box is [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]'''
        horizontal_list, free_list = self.detection_model.detect(image)

        # format horizontal_list into four vertices in a (4,2) ndarray
        # left up, left down, right down, right up
        vertices = []
        for box in horizontal_list[0]:
            box_vertices = [
                [box[0], box[3]],
                [box[0], box[2]],
                [box[1], box[2]],
                [box[1], box[3]]
            ]
            vertices.append(np.array(box_vertices))
        return vertices