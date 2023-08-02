import cv2
import argparse
import easyocr
import pandas as pd
import time
import torch
import torchvision.transforms.functional as TF
from torchvision import transforms as T
import numpy as np
from pathlib import Path

from DetectionModel import DetectionModel, DBOpenCV, DBEasyOCR
from RecognitionModel import RecognitionModel, CRNNOpenCV, CRNNEasyOCR, TrOCR, Parseq

def check_input_size(model_name):
    import onnxruntime as ort
    model = ort.InferenceSession(model_name, providers=['CPUExecutionProvider'])
    input_shape = model.get_inputs()[0].shape
    return input_shape

def e2e_easyocr(image_path):
    reader = easyocr.Reader(['en'], False, "easyocr_models/", detect_network='dbnet18')
    return reader.readtext(image_path, detail=0)


def decode(preds, alphabet):
    rec_text = []
    # take preds' argmax of the last dimension
    ids = preds.argmax(-1) - 1
    for b in ids:
        b_text = []
        for id in b:
            if id == -1:
                break
            b_text.append(alphabet[id])
        rec_text.append(''.join(b_text))
    return rec_text

def visualize_transformation():
    transform = T.Compose([
            T.ToTensor(),
            T.Resize((32, 128), T.InterpolationMode.BICUBIC),
            # T.Normalize(0.5, 0.5)
    ])

    img = cv2.imread('images/3ave.jpeg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # image = TF.to_tensor(img)
    # image = TF.resize(image, [32, 128])
    image = transform(img)
    image = image.numpy()
    image = np.transpose(image, (1, 2, 0))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # show it in cv2 again
    cv2.imshow("Text Recognition", image)
    cv2.waitKey(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", type=str)
    parser.add_argument("--use-detection", help="Use Detection Model or not", default=False, action='store_true')
    parser.add_argument("-v", help="Show images step by step", default=False, action='store_true')
    parser.add_argument("--detect-path", help = "Detection Model path", type=str, default = "DB_TD500_resnet50.onnx")
    parser.add_argument("--recog-path", help = "Recognition Model path", type=str, default = "crnn_cs.onnx")
    args = parser.parse_args()

    # Load image
    image = cv2.imread(args.image_path)
    if args.v:
        cv2.imshow("Original", image)
        cv2.waitKey(0)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detection Model produce bounding boxes
    if args.use_detection:
        detection_model = DBOpenCV(args.detect_path)
        # detection_model = DBEasyOCR()
        vertices = detection_model.detect(image)
        annotated = cv2.polylines(image, vertices, True, [0, 255, 0], 2)
        if args.v:
            cv2.imshow("Text Recognition", cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
            cv2.waitKey(0)

    bounding_boxes = vertices if args.use_detection else None

    # Recognition Model produce text
    # recognition_model = CRNNOpenCV(args.recog_path), CRNNEasyOCR(), TrOCR(), Parseq()
    rec_models = [Parseq()]
    result = []
    for model in rec_models:
        start_time = time.time()
        rec_text = model.recognize(image, bounding_boxes)
        inference_time = time.time() - start_time
        result.append([model.__class__.__name__, rec_text, inference_time])

    # format result to a csv
    result = pd.DataFrame(result, columns=['Model', 'Text', 'Inference Time'])
    result.to_csv('result.csv', index=False)

    # rec_text = e2e_easyocr(args.image_path)
    print(result)


if __name__ == "__main__":
    # main()
    model_dir = Path("saved_models")
    for file in model_dir.iterdir():
        # check all onnx
        if file.suffix == ".onnx":
            print(file)
            print(check_input_size(file))
    
    # model = torch.hub.load('baudm/parseq', 'parseq', pretrained=True).eval()
    # model.load_state_dict(torch.load('saved_models/parseq.pt', map_location=torch.device('cpu')))
    # model = torch.jit.load('saved_models/parseq-small-ts.bin')

    # image = TF.normalize(image, 0.5, 0.5)
    # image = image.unsqueeze(0)
    # logits = model(image)
    # pred = logits.softmax(-1)
    # gt = model.tokenizer.decode(pred)

    # # read 94_full.txt
    # with open('94_full.txt', 'r') as f:
    #     alphabet = f.read()
    # text = decode(pred, alphabet)