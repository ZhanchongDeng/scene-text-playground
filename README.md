# 🔎 Computer Vision with Text
zhanchong@eyepop.ai

This page briefly discuss computer vision with Text related data, and discuss in depth about Scene Text Detection and Scene Text Recogntion.

## Introduction

Computer Vision is commonly associated with objects detection or human detection, but text occupies a large chunk of what we/computers see as well. When it comes to textual data, it's important to differentiate between Optical Character Recognition (OCR) versus Scene Text Recognition (STR):

**OCR** models expects scanned documents, typically in close to horizontal orientation, has large body text, whether it's handwritten or printed. It's considered solved in Academia with close to 98 accuracy on some dataset.

**STR** models expects text within a natural environment, which can have variations in orientation, lighting, curvature, fonts, etc. As of July of 2023, there are room for improvements in both detection and recognition of these text.

**STR** fits better with our need because it is a natural extension on top of segmentation/bounding box from other computer vision task/model. For example, if we wish to detect license plate after finding a car in our image, we should use a STR right after Object Detection model.

## Task

Computer vision on scene text is typically divided into a 2-stage process.

### Scene Text Detection

Detection reads an image containing one more many text body, and produce bounding box or segmentation mask for each text body detected.

<figure><img src="../.gitbook/assets/Screenshot 2023-07-21 at 2.54.53 PM.png" alt=""><figcaption><p>Multiple detection box in green.</p></figcaption></figure>

### Scene Text Recognition

The output of detection is typically cropped and piped into recognition, and we end up with texts bounded to each bounding box.

> `['52442284', 'WILDLIFE', 'INJURED', '3km', 'NEXT']`

Note 0: STR seems to be used interchangebly with the entire task and this second stage.

Note 1: Most recognition model seem to assume 1 text body per image. This assumption exist across different models and dataset. Recognizing an image with lots of text all over the place using these models yield terrible result, as they are not expecting that type of input.

Here's a good [summary paper](https://arxiv.org/pdf/1904.01906.pdf)/[github](https://github.com/clovaai/deep-text-recognition-benchmark) talking about limitations of current STR models.

## Dataset

Scene Text, like with natural language processing, is expensive to label. Thus, these dataset are typically small (200\~500 images). Most models are actually trained on synthetic data in combination with these expensive real data. Below are representative dataset referenced by a lot of models between 2017-2023.

* [ICDAR 2013](https://paperswithcode.com/dataset/icdar-2013)
* [MSRA-TD500](https://paperswithcode.com/dataset/msra-td500)
* [Total-Text](https://paperswithcode.com/dataset/total-text)

## Models

Models should be selected based on:

1. Correctness (as evaluated on dataset)
2. Speed
3. Integration Difficulty

* PaddleOCR
  * Github: [https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc\_en/recognition\_en.md](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc\_en/recognition\_en.md)
* OpenCV DNN TextDetection & TextRecognition
  * Github: [https://github.com/opencv/opencv/blob/master/doc/tutorials/dnn/dnn\_text\_spotting/dnn\_text\_spotting.markdown](https://github.com/opencv/opencv/blob/master/doc/tutorials/dnn/dnn\_text\_spotting/dnn\_text\_spotting.markdown)
  * [https://github.com/opencv/opencv/blob/4.x/samples/dnn/text\_detection.py](https://github.com/opencv/opencv/blob/4.x/samples/dnn/text\_detection.py)
* easyocr
  * API: [https://www.jaided.ai/easyocr/documentation/](https://www.jaided.ai/easyocr/documentation/)
  * Github: [https://github.com/JaidedAI/EasyOCR](https://github.com/JaidedAI/EasyOCR)
  * PyPl: [https://pypi.org/project/easyocr/](https://pypi.org/project/easyocr/)
* TrOCR
  * Paper: [https://arxiv.org/pdf/2109.10282.pdf](https://arxiv.org/pdf/2109.10282.pdf)
  * Github: [https://github.com/microsoft/unilm/tree/master/trocr](https://github.com/microsoft/unilm/tree/master/trocr)
  * Example Notebook: [https://github.com/NielsRogge/Transformers-Tutorials/blob/master/TrOCR/Inference\_with\_TrOCR\_%2B\_Gradio\_demo.ipynb](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/TrOCR/Inference\_with\_TrOCR\_%2B\_Gradio\_demo.ipynb)
  * API Documentation: [https://huggingface.co/docs/transformers/model\_doc/trocr](https://huggingface.co/docs/transformers/model\_doc/trocr)
  * Model Hub: [https://huggingface.co/models?sort=trending\&search=microsoft%2Ftrocr](https://huggingface.co/models?sort=trending\&search=microsoft%2Ftrocr)

