import onnxruntime as ort
from cv2.dnn import TextRecognitionModel
import easyocr
import numpy as np
import cv2

class RecognitionModel:
    def recognize(self, images, vertices):
        raise NotImplementedError
    

class CRNNOpenCV(RecognitionModel):
    def __init__(self, model_path = 'crnn_cs.onnx', alphabet_path = 'alphabet_36.txt'):
        self.alphabet_path = alphabet_path
        self.vocabulary = []
        self.decode_type = 'CTC-greedy'
        self.scale = 1/127.5
        self.mean = [127.5, 127.5, 127.5]
        self.input_size = (100, 32)
        
        self.recognition_model = TextRecognitionModel(model_path)
        self.recognition_model.setDecodeType(self.decode_type)
        with open(self.alphabet_path, 'r') as f:
            for line in f.readlines():
                self.vocabulary.append(line.strip())
        self.recognition_model.setVocabulary(self.vocabulary)
        self.recognition_model.setInputParams(scale=self.scale, size=self.input_size, mean=self.mean)

    def recognize(self, image, vertices):
        '''Text Recognition using CRNN model
        
        Args:
            image (numpy.ndarray): Image to be recognized.
            vertices:  List of bounding boxes.

        Returns:
            rec_text:   Recognized text.
        '''
        rec_text = []
        for box in vertices:
            cropped = self.fourPointsTransform(image, box)
            text = self.recognition_model.recognize(cropped)
            rec_text.append(text)
        return rec_text
    
    @staticmethod
    def fourPointsTransform(frame, vertices):
        vertices = np.asarray(vertices)
        outputSize = (100, 32)
        targetVertices = np.array([
            [0, outputSize[1] - 1],
            [0, 0],
            [outputSize[0] - 1, 0],
            [outputSize[0] - 1, outputSize[1] - 1]], dtype="float32")

        # convert vertices and targetVertices to float32
        vertices = vertices.astype("float32")
        targetVertices = targetVertices.astype("float32")
        rotationMatrix = cv2.getPerspectiveTransform(vertices, targetVertices)
        result = cv2.warpPerspective(frame, rotationMatrix, outputSize)
        return result
    
class CRNNEasyOCR(RecognitionModel):
    def __init__(self, model_path = None, alphabet_path = None):
        # recog_network can be "standard" or https://www.jaided.ai/easyocr/modelhub/
        # these are mostly CRNN models anyways, but with different language
        self.recognition_model = easyocr.Reader(['en'], False, "easyocr_models/", recog_network = "standard", recognizer = True, detector = False)

    def recognize(self, image, vertices):
        rec_text = []
        for box in vertices:
            cropped = CRNNOpenCV.fourPointsTransform(image, box)
            text = self.recognition_model.recognize(cropped)
            rec_text.append(text)
        return rec_text


class TROCR(RecognitionModel):
    def __init__(self, model_path = '...', alphabet_path = '...'):
        ...

    def recognize(self, cropped):
        return ...
