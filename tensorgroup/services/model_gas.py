
import numpy as np
import json
import os
import tensorflow as tf

from tensorgroup.models.networks.CenterNetBuilder import CenterNetBuilder as CNB
from tensorgroup.models.networks.ResnetKeypointBuilder import ResnetKeypointBuilder as RKB

KPN_I_SIZE = 512

class model_gas():
    def __init__(self, config):
        # 初始化检测Pressure Gauge的Centernet
        datasetname = config['centernet_datasetname']
        saved_presure_gauge_model_dir = os.path.join(config['saved_model_dir'], 'centernet', datasetname, 'sm')
        self._pgn_i_size = config['pgn_i_size']
        _, self._predict_pg, _ = CNB.CenterNetOnResNet50V2(config['classes_num'], input_size=self._pgn_i_size)
        self._predict_pg.load_weights(os.path.join(saved_presure_gauge_model_dir,'{}.h5'.format(datasetname))) # 不要用 by_name=True
        self._predict_pg.make_predict_function()

        # 初始化检测指针角度的KeyPointNet
        datasetname = config['keypointnet_datasetname']
        saved_keypoint_model_dir = os.path.join(config['saved_model_dir'], 'keypointnet', datasetname, 'sm')
        self._kpn_i_size = config['kpn_i_size']
        _, self._predict_kp, _ = RKB.build_keypoint_resnet_101(input_size=self._kpn_i_size) # use default value
        self._predict_kp.load_weights(os.path.join(saved_keypoint_model_dir,'{}.h5'.format(datasetname)))
        self._predict_kp.make_predict_function()

    def inference(self, image_array):
        p_result = self.__pgn_inference(image_array)
        # 对p_result的结果一个一个的处理。
        k_result = self.__kpn_inference(image_array)
        return (p_result, k_result)

    def __pgn_inference(self, image_array):
        image_t = tf.convert_to_tensor(image_array)
        image_t = tf.image.convert_image_dtype(image_t, dtype=tf.float32)
        image_t = tf.image.resize(image_t, [self._pgn_i_size, self._pgn_i_size], method=tf.image.ResizeMethod.BILINEAR)
        image_input = tf.expand_dims(image_t, axis=0)
        predicts = self._predict_pg.predict(image_input)[0]
        scores = predicts[:, 2]
        indices = np.where(scores > 0.10)
        detections = predicts[indices].copy()
        return detections        

    def __kpn_inference(self, image_array):
        # image in RGB-mode
        # w, h, _ = image_array.shape

        image_t = tf.convert_to_tensor(image_array)
        image_t = tf.image.convert_image_dtype(image_t, dtype=tf.float32)
        image_t = tf.image.resize(image_t, [self._kpn_i_size, self._kpn_i_size], method=tf.image.ResizeMethod.BILINEAR)
        image_input = tf.expand_dims(image_t, axis=0)
        predicts = self._predict_kp.predict(image_input)[0]
        scores = predicts[:, 2]
        indices = np.where(scores > 0.50)
        detections = predicts[indices].copy()
        return detections


