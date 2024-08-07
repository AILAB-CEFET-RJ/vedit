import numpy as np
import os
from datetime import datetime
import json
import cv2


def annotate_text(text, frame, pos_x, pos_y, font_size):

    shape = np.zeros_like(frame, np.uint8)

    width= int(len(text) * font_size * 17.5)
    height = int(font_size * 15)

    cv2.rectangle(shape, (max(pos_x-10, 0), max(int(pos_y-(font_size*30)), 0)), (pos_x+width, pos_y+height), (1,1,1), -1)
    mask = shape.astype(bool)
    frame[mask] = cv2.addWeighted(frame, 0.4, shape, 0.6, 1)[mask]

    cv2.putText(
    frame,
    text,
    (pos_x, pos_y),
    cv2.FONT_HERSHEY_SIMPLEX,
    font_size,
    (0, 255, 0),
    2,
    )

    return frame

def save_detection(frame, json_data):

    cur_time = (datetime.now()).strftime("%d%m%Y_%H-%M-%S")

    new_path = os.path.join('detections',cur_time)

    if not os.path.exists(new_path):
        os.makedirs(new_path)

    for index, obj in enumerate(json_data):
        obj_crop = frame[int(obj['box']['y1']):int(obj['box']['y2']), int(obj['box']['x1']):int(obj['box']['x2'])]

        img_save_path = os.path.join(new_path, str(index) + "_" + cur_time +".jpg")
        cv2.imwrite(img_save_path,obj_crop)

        data_save_path = os.path.join(new_path, str(index) + "_" + cur_time +".json")
        with open(data_save_path, 'w') as file:
            json.dump(obj, file, indent=4)


