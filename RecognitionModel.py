import onnxruntime as ort
from cv2.dnn import TextRecognitionModel
import easyocr
import numpy as np
import cv2
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch
from torchvision import transforms as T
import torchsummary

class RecognitionModel:
    def recognize(self, images, vertices):
        raise NotImplementedError
    
    @staticmethod
    def fourPointsTransform(frame, vertices, outputSize = [100, 32]):
        vertices = np.asarray(vertices)
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
        if vertices == None:
            text = self.recognition_model.recognize(image)
            rec_text.append(text)
        else:
            for box in vertices:
                cropped = self.fourPointsTransform(image, box)
                text = self.recognition_model.recognize(cropped)
                rec_text.append(text)
        return rec_text
    
class CRNNEasyOCR(RecognitionModel):
    def __init__(self):
        # recog_network can be "standard" or https://www.jaided.ai/easyocr/modelhub/
        # these are mostly CRNN models anyways, but with different language
        self.recognition_model = easyocr.Reader(['en'], False, "easyocr_models/", recog_network = "standard", recognizer = True, detector = False)

    def recognize(self, image, vertices):
        rec_text = []
        for box in vertices:
            cropped = self.fourPointsTransform(image, box)
            text = self.recognition_model.recognize(cropped)
            rec_text.append(text)
        return rec_text


class TrOCR(RecognitionModel):
    def __init__(self):
        self.processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-str')
        self.model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-str')
        self.input_size = [1, 3, 384, 384]
        
    def recognize(self, image, vertices):
        rec_text = []
        if vertices == None:
            pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
            generated_ids = self.model.generate(pixel_values)
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            rec_text.append(generated_text)
        else:
            for box in vertices:
                cropped = self.fourPointsTransform(image, box, self.input_size[2:4])
                pixel_values = self.processor(images=cropped, return_tensors="pt").pixel_values
                generated_ids = self.model.generate(pixel_values)
                generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                rec_text.append(generated_text)
        return rec_text
    
    
class Parseq(RecognitionModel):
    def __init__(self):
        # available models: abinet, crnn, trba, vitstr, parseq_tiny, and parseq.
        self.model = torch.hub.load('baudm/parseq', 'parseq', pretrained = True).eval()
        self.img_size = [1, 3, 32, 128]
        self.transform = T.Compose([
            T.ToTensor(),
            T.Resize((self.img_size[2:4]), T.InterpolationMode.BILINEAR),
            # T.Normalize(0.5, 0.5),
        ])
        # torchsummary.summary(self.model, [3, 224, 224])

    def recognize(self, image, vertices):
        rec_text = []
        if vertices == None:
            image = self.transform(image)
            image = image.unsqueeze(0)
            logits = self.model(image)
            pred = logits.softmax(-1)
            label, confidence = self.model.tokenizer.decode(pred)
            rec_text.append(label[0])
        else:
            for box in vertices:
                cropped = self.fourPointsTransform(image, box, self.image_size[2:4])
                cropped = self.transform(cropped)
                cropped = cropped.unsqueeze(0)
                logits = self.model(cropped)
                pred = logits.softmax(-1)
                label, confidence = self.model.tokenizer.decode(pred)
                rec_text.append(label[0])
        return rec_text