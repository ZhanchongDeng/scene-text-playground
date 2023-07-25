import cv2
import argparse
import easyocr
import pandas as pd
import time

from DetectionModel import DetectionModel, DBOpenCV, DBEasyOCR
from RecognitionModel import RecognitionModel, CRNNOpenCV, CRNNEasyOCR, TrOCR, Parseq

def check_input_size(model_name):
    import onnxruntime as ort
    model = ort.InferenceSession(model_name, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    input_shape = model.get_inputs()[0].shape
    return input_shape

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
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # cv2.imshow("Original", image)
    # cv2.waitKey(0)

    detection_model = DBOpenCV(args.detection_model_path)
    # detection_model = DBEasyOCR()
    vertices = detection_model.detect(image)
    annotated = cv2.polylines(image, vertices, True, [0, 255, 0], 2)
    cv2.imshow("Text Recognition", cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)


    # recognition_model = CRNNOpenCV(args.recognition_model_path)
    # recognition_model = CRNNEasyOCR()
    # recognition_model = TrOCR()
    # recognition_model = Parseq()
    rec_models = [Parseq()]
    result = []
    for model in rec_models:
        start_time = time.time()
        rec_text = model.recognize(image, vertices)
        inference_time = time.time() - start_time
        result.append([model.__class__.__name__, rec_text, inference_time])

    # format result to a csv
    result = pd.DataFrame(result, columns=['Model', 'Text', 'Inference Time'])
    result.to_csv('result.csv', index=False)

    # rec_text = e2e_easyocr(args.image_path)

    print(result)
