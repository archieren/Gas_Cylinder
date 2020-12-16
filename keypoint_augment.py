# -*- coding: utf-8 -*-
import os
import json
import numpy as np
from PIL import Image

import base64
from io import BytesIO

import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.kps import Keypoint, KeypointsOnImage

DATASETNAME = 'three_point'
dataset_dir = os.path.join(os.getcwd(), 'data_coco_kp', DATASETNAME, 'temp')

ia.seed(3)
seq = iaa.Sequential([
   iaa.PerspectiveTransform(scale = (0.01, 0.05)),
   # iaa.TranslateY(percent=(-0.1, 0.1)),
   # iaa.TranslateX(percent=(-0.1, 0.1)),
   iaa.Rotate(rotate=(-179, 179)),
   # iaa.GaussianBlur(sigma = (0.0, 1.5)),
   # iaa.LinearContrast((0.5, 1)),
   # iaa.AdditiveGaussianNoise(scale = (0, 5)),
   # iaa.Multiply((0.5, 1.5))
])

def get_name_list(dir):

    name_list = []
    for file in os.listdir(dir):
        if file.endswith('.json'):
            name = file.split('.')[0]
            name_list.append(name)
    return name_list

def image_to_base64(image_array):
    img = Image.fromarray(image_array, mode='RGB')
    output_buffer = BytesIO()
    img.save(output_buffer, format='JPEG')
    byte_data = output_buffer.getvalue()
    base64_str = base64.b64encode(byte_data)
    return base64_str.decode("utf-8")

def parse_annotations(json_file_name):
    points, labels = [], []
    with open(json_file_name) as json_file:
        data = json.load(json_file)
        for p in data['shapes']:
            x = p['points'][0][0]
            y = p['points'][0][1]
            points.append([x, y])
            labels.append(p['label'])

    return points, labels

def generate_augment_dataset(dataset_dir, name_list, num_aug = 500):

    for name in name_list:

        image_name = os.path.join(dataset_dir, name + '.jpg')
        json_name = os.path.join(dataset_dir, name + '.json')

        image = Image.open(image_name).convert('RGB')
        image = np.asarray(image)
        points, labels = parse_annotations(json_name)

        kps = [Keypoint(x = int(points[i][0]), y = int(points[i][1])) for i in range(len(points))]
        kpsoi = KeypointsOnImage(kps, shape = image.shape)

        for j in range(num_aug):
            image_aug, kpsoi_aug = seq(image = image, keypoints = kpsoi)
            points_aug = kpsoi_aug.keypoints
            height, width, _ = image_aug.shape

            image_aug_name = os.path.join(dataset_dir, name + '_' + str(j) + '.jpg')
            json_aug_name = os.path.join(dataset_dir, name + '_' + str(j) + '.json')
            data = {}
            data['version'] = '4.2.10'
            data['flags'] = {}
            data['shapes'] = []
            for k in range(len(points_aug)):
                point_name = labels[k]
                p_x = int(points_aug[k].x)
                p_y = int(points_aug[k].y)
                data['shapes'].append({
                    'label': point_name,
                    'points':[[p_x, p_y]],
                    'group_id': None,
                    'shape_type': 'point',
                    'flags':{}
                    })
            data['imagePath'] = name + str(j) + '.jpg'
            data['imageData'] = image_to_base64(image_aug)
            data['imageHeight'] = height
            data['imageWidth'] = width
            with open(json_aug_name, 'w') as outfile:
                json.dump(data, outfile, indent = 4)
            image_aug = Image.fromarray(image_aug)
            image_aug.save(image_aug_name)
            print('success')



name_list = get_name_list(dataset_dir)
generate_augment_dataset(dataset_dir, name_list)









