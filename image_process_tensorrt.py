#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# file: image_process.py
# author: jiangqr
# data: 2017.1.3
# note: receive and send message for web
#
import tensorrt as trt
from tensorrt.parsers import uffparser
import uff
import pycuda.driver as cuda
import tensorflow as tf
import os
import numpy as np
import cv2
import sys
import config
import time
mutex = config.mutex


class ImageProcess:
    """"""
    def __init__(self):
        self.label_file = config.LABEL_FILE
        self.model_file = config.MODEL_FILE
        self.output__threshold = config.OUTPUT_THRESHOLD
        self.id2name = self.get_labels()
        tf_config = tf.ConfigProto(allow_soft_placement=True)
        tf_config.gpu_options.per_process_gpu_memory_fraction = 0.01  # 占用GPU20%的显存
        tf_config.gpu_options.allow_growth = True
        tf.Graph().as_default()
        self.sess = tf.Session(config=tf_config)
        #print('************scene gpu id:{}'.format(config.GPU_ID))
        print('*************scene gpu id: 1')
        self.create_graph()
        #with tf.device('/gpu:{}'.format(config.GPU_ID)):
        with tf.device('/gpu:1'):
            
            self.image_placeholder = tf.placeholder(tf.uint8, shape=(None,None,3))
            image_rgb = tf.reverse(self.image_placeholder, [-1])
            image = tf.image.convert_image_dtype(image_rgb, dtype=tf.float32)
            image = tf.expand_dims(image,0)
            image = tf.image.resize_bilinear(image, [512, 512],
                                                   align_corners=False) 
            image = tf.squeeze(image, [0])
            image = tf.subtract(image, 0.5)
            image = tf.multiply(image, 2.0)
            self.image = tf.transpose(image, [2,0,1])
                    
            # Loading the injected placeholder
            #self.input_placeholder = self.sess.graph.get_tensor_by_name("input_image:0")

    def __del__(self):
        if self.sess is not None:
            self.sess.close()

    def create_graph(self):
        """"""
        uff_model = uff.from_tensorflow_frozen_model(self.model_file, 
                                                     ['InceptionResnetV2/Logits/Predictions'])
        G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.ERROR)
        parser = uffparser.create_uff_parser()
        parser.register_input('input_image', (3,512,512), 0)
        parser.register_output('InceptionResnetV2/Logits/Predictions')
        
        engine = trt.utils.uff_to_trt_engine(G_LOGGER, uff_model, parser, 1, 1<<32)
        
        parser.destroy()
        
        runtime = trt.infer.create_infer_runtime(G_LOGGER)
        self.context = engine.create_execution_context()
        
        self.output = np.empty(len(self.id2name), dtype = np.float32)
        self.d_input = cuda.mem_alloc(1 * 512 * 512 * 3 * 4)
        self.d_output = cuda.mem_alloc(1 * len(self.id2name) * 4)
        
        self.bindings = [int(self.d_input), int(self.d_output)]
        self.stream = cuda.Stream()

    def get_labels(self):
        """
           get labels from label
           return dict {"id":"label name class2_name, class2_name",.....}
        """
        id2label = {}
        assert (os.path.exists(self.label_file))
        file_object = open(self.label_file)

        while 1:
            line = file_object.readline()
            if not line or line == '':
                break

            line = line.strip('\n').split(":")
            index, name= line[0], line[1]
            bool_valid = 0 if len(line)>4 else 1
            id2label[index] =  [name, bool_valid]
        return id2label

    def run(self, img):
        """detect image"""
        if img is None:
            return 0, []
        
        processed_image = self.sess.run(self.image, {self.image_placeholder: img})
        processed_image = processed_image.ravel()
        cuda.memcpy_htod_async(self.d_input, processed_image, self.stream)
        self.context.enqueue(1, self.bindings, self.stream.handle, None)
        cuda.memcpy_dtoh_async(self.output, self.d_output, self.stream)
        self.stream.synchronize()
        
        index = np.argmax(self.output)
        probabilities = self.output[index]
        name = self.id2name[str(index)]
        #probabilities = self.sess.run(self.softmax, {self.input_placeholder: self.sess.run(self.image, {self.image_placeholder: img})})
        name, bool_valid = self.id2name[str(index)]

        if probabilities < self.output__threshold or not bool_valid:
            return 0, [name, probabilities]
        else:
            return 1, [name, probabilities]


if __name__ == '__main__':
    img_handle = ImageProcess()
    cap = cv2.VideoCapture('/data/huang/behaviour_test/test_video_huanlesong_1/huanlesong_1.mp4')
    if cap.isOpened():
        ret, img = cap.read()
        frame_id = 0
        while ret:
            t0 = time.time()
            ret2, result = img_handle.run(img)
            t1 = time.time()
            ret, img = cap.read()
            frame_id += 1
            print(frame_id, t1-t0, result)
            # print 'the scene name is %s, the score is %f' % (result[0], result[1])
        cap.release()
