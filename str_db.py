import cv2
import argparse
import numpy as np

from DetectionModel import DetectionModel, DBOpenCV, DBEasyOCR
from RecognitionModel import RecognitionModel, CRNNOpenCV, CRNNEasyOCR


def detect_image(detection_model:DetectionModel, image):
    '''Text Detection using model.
    
    Args:
        detection_model (DetectionModel): Detection model to be used.
        image (numpy.ndarray): Image to be detected.
        
    Returns:
        vertices:   List of bounding boxes.
    '''
    vertices = detection_model.detect(image)
    return vertices

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

def check_input_size(model_name):
    import onnxruntime as ort
    model = ort.InferenceSession(model_name, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    input_shape = model.get_inputs()[0].shape
    return input_shape


def recognition_image(recognition_model:RecognitionModel, image, vertices = None):
    '''Text Recognition using CRNN model
    
    Args:
        recogntion_model (RecognitionModel): Recognition model to be used.
        image (numpy.ndarray): Image to be recognized.
        vertices:  List of bounding boxes.

    Returns:
        rec_text:   Recognized text.
    '''
    rec_text = []
    for box in vertices:
        cropped = fourPointsTransform(image, box)
        cv2.imshow("cropped", cropped)
        cv2.waitKey(0)
        text = recognition_model.recognize(cropped)
        rec_text.append(text)
    return rec_text

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("detection_model_path", type=str)
    parser.add_argument("recognition_model_path", type=str)
    parser.add_argument("image_path", type=str)
    args = parser.parse_args()

    
    # Load image
    image = cv2.imread(args.image_path)
    # cv2.imshow("Original", image)
    # cv2.waitKey(0)

    detection_model = DBOpenCV(args.detection_model_path)
    # detection_model = DBEasyOCR()
    vertices = detect_image(detection_model, image)
    annotated = cv2.polylines(image, vertices, True, [0, 255, 0], 2)
    cv2.imshow("Text Recognition", annotated)
    cv2.waitKey(0)


    recognition_model = CRNNOpenCV(args.recognition_model_path)
    # recognition_model = CRNNEasyOCR()
    rec_text = recognition_image(recognition_model, image, vertices)

    print(rec_text)
