from pathlib import Path
import xml.etree.ElementTree as ET
import cv2
import lmdb

cute80_base_dir = Path('CUTE80_Dataset')
imgdir = cute80_base_dir / 'CUTE80'
gt = cute80_base_dir / 'Groundtruth' / 'GroundTruth.xml'

tree = ET.parse(gt)
root = tree.getroot()
num_found = []
num_not_found = []
for child in root:
    img_file_name = child[0].text
    num_text = len(child) - 1

    if (imgdir / img_file_name).exists():
        img = cv2.imread(str(imgdir / img_file_name))
        print(f"Image size: {img.shape}")
        print(f"Number of text: {num_text}")
        for i in range(num_text):
            num_x = len(child[i + 1].attrib['x'].split(' '))
            num_y = len(child[i + 1].attrib['y'].split(' '))
            print(f"Text {i + 1}: {num_x} {num_y}")

            # if num_x > 4:
            #     print(img_file_name)

        num_found.append(img_file_name.lower())
    else:
        # print(f"Image {img_file_name} not found")
        num_not_found.append(img_file_name.lower())

print(f"Number of found images: {len(num_found)}")
print(f"Number of not found images: {len(num_not_found)}")
import collections
found_count = collections.Counter(num_found)
# print all count > 1
for key, value in found_count.items():
    if value > 1:
        print(key, value)

# img79 = cv2.imread(str(imgdir / 'image079.jpg'))
# cv2.imshow('img79', img79)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
    # print(len(child[1].attrib['x'].split(' ')))
    # print(len(child[1].attrib['y'].split(' ')))