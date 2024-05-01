import re
from typing import List
import requests
import json
import numpy as np
import traceback
import time
import cv2

def check_english(input_sent):
    english_pattern  = re.compile(r'[A-Za-z0-9]+')
    if english_pattern.search(input_sent):
        return True
    return False

def order_boxes4nom(
    boxes: List[List[int]]
    )->List[List[int]]:
    return sorted(
        boxes,
        key=lambda x: x[0][0][0],
        reverse=True
    )
    # boxes = np.asarray(boxes)
    # if len(boxes) == 0:
    #     return boxes, []

    # # Enumerate the boxes along with their indices
    # indexed_boxes = list(enumerate(boxes))
    
    # # Sort the boxes based on the minimum x-coordinate of each box
    # indexed_boxes.sort(key=lambda x: -x[1][:, 0].min())

    # threshold = 15
    # sorted_boxes = []
    # sorted_indices = []
    # row = [indexed_boxes[0]]
    # prev_y = indexed_boxes[0][1][:, 0].min()

    # for i in range(1, len(indexed_boxes)):
    #     index, box = indexed_boxes[i]
    #     y = box[:, 0].min()
    #     if abs(y - prev_y) > threshold:
    #         # Sort the row from right to left
    #         row.sort(key=lambda x: x[1][:, 1].min())
    #         sorted_boxes.extend(row)
    #         sorted_indices.extend([item[0] for item in row])
    #         row = []
    #     row.append((index, box))
    #     prev_y = y

    # if len(row) > 0:
    #     row.sort(key=lambda x: x[1][:, 1].min())
    #     sorted_boxes.extend(row)
    #     sorted_indices.extend([item[0] for item in row])

    # # Extract the sorted boxes and indices
    # sorted_boxes = np.array([item[1] for item in sorted_boxes])
    
    # return sorted_boxes, sorted_indices

def translate(text):
    url = 'https://api.clc.hcmus.edu.vn/nom_translation/90/1'
    response = requests.request('POST', url, data={'nom_text': text})
    time.sleep(0.1)     
    
    try:
        result = json.loads(response.text)['sentences']
        result = result[0][0]['pair']['modern_text']
        return result
    except Exception as e:
        print(e)
        traceback.print_exc()
        print(f'[ERR] "{text}": {response.text}')
        return 'Cannot translate this text.'
    
    
def crop_img_boxes(boxes, image_array):
    margin = 10
    crop_img =[]
    for bounding_box_coords in boxes:
        # Xác định tọa độ cắt dựa trên bounding box
        x_min = min(coord[0] for coord in bounding_box_coords) - margin
        x_max = max(coord[0] for coord in bounding_box_coords) + margin
        y_min = min(coord[1] for coord in bounding_box_coords) - margin
        y_max = max(coord[1] for coord in bounding_box_coords) + margin

        # Cắt phần bounding box từ ảnh gốc sử dụng lát cắt NumPy
        cropped_image_array = image_array[y_min:y_max, x_min:x_max, :]
        cropped_image_array = cv2.rotate(cropped_image_array, cv2.ROTATE_90_COUNTERCLOCKWISE)
        crop_img.append(cropped_image_array)
    return crop_img

if __name__ == '__main__':
    bboxes = [
        [
            [[1762.0, 424.0], [1871.0, 430.0], [1805.0, 1640.0], [1696.0, 1634.0]]
            , ('敕命緬念神奉著封德靈運翊保其申', 0.6955432891845703)
        ], 
        [[[2210.0, 505.0], [2310.0, 514.0], [2207.0, 1639.0], [2107.0, 1630.0]], ('敕神仍有水奉神凖社春事等穆', 0.7343415021896362)], 
        [[[1912.0, 539.0], [2023.0, 545.0], [2008.0, 827.0], [1897.0, 821.0]], ('今丕承', 0.9922056198120117)],
        [[[1446.0, 582.0], [1541.0, 580.0], [1547.0, 928.0], [1452.0, 930.0]],('黎民欽哉', 0.9265642762184143)],
        [[[1610.0, 574.0], [1702.0, 578.0], [1656.0, 1615.0], [1564.0, 1611.0]], ('春凖其事陸我神其相佑保哉', 0.7922740578651428)],
        [[[2061.0, 588.0], [2153.0, 596.0], [2059.0, 1633.0], [1967.0, 1625.0]], ('應欽給神護國庇民稔著靈應肆', 0.9537190794944763)],
        [[[202.0, 671.0], [305.0, 667.0], [326.0, 1268.0], [223.0, 1271.0]], ('嗣定貳年叁月拾捌日', 0.9157988429069519)]
    ]
    res = order_boxes4nom(bboxes)
    # Tạo hai mảng riêng biệt cho tọa độ và văn bản
    coordinates_list = []
    text_list = []

    for bbox in res:
        coordinates = bbox[0]
        text = bbox[1][0]
        
        coordinates_list.append(coordinates)
        text_list.append(text)

    print(f'text_list: {text_list}')
    print(f'coordinates_list: {coordinates_list}')