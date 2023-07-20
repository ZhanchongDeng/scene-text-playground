import cv2
import argparse
import easyocr

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
        cropped = CRNNOpenCV.fourPointsTransform(image, box)
        cv2.imshow("cropped", cropped)
        cv2.waitKey(0)
        text = recognition_model.recognize(cropped)
        rec_text.append(text)
    return rec_text

def e2e_easyocr(image_path):
    reader = easyocr.Reader(['en'], False, "easyocr_models/", detect_network='dbnet18')
    return reader.readtext(image_path, detail=0)

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


    # recognition_model = CRNNOpenCV(args.recognition_model_path)
    recognition_model = CRNNEasyOCR()
    rec_text = recognition_model.recognize(image, vertices)

    # rec_text = e2e_easyocr(args.image_path)

    print(rec_text)
