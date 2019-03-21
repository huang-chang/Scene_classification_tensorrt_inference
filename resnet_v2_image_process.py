#!/usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf
import os
import numpy as np
import cv2
import sys
import time
import tensorrt as trt
from tensorrt.parsers import uffparser
import uff
import pycuda.driver as cuda
from PIL import Image


class ImageProcess:
    def __init__(self, label_file, model_file, threshold=0.5):
        self.label_file = label_file
        self.model_file = model_file
        self.output__threshold = threshold
        self.id2name = self.get_labels()
        #tf_config = tf.ConfigProto()
        #tf_config.gpu_options.per_process_gpu_memory_fraction = 0.2  # 占用GPU20%的显存
        #tf_config.gpu_options.allow_growth = True
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.1)
        #tf.Graph().as_default()
        self.sess = tf.Session(config = tf.ConfigProto(gpu_options = gpu_options))
        self.create_graph()
        
        with tf.device('/gpu:1'):
            self.image_placeholder = tf.placeholder(tf.uint8, shape=(None, None, 3))
            image_rgb = tf.reverse(self.image_placeholder, [-1])
            image = tf.image.convert_image_dtype(image_rgb, dtype=tf.float32)
            image = tf.expand_dims(image,0)
            image = tf.image.resize_bilinear(image, [512,512], align_corners=False)
            image = tf.squeeze(image, [0])
            image = tf.subtract(image, 0.5)
            image = tf.multiply(image, 2.0)
            #self.image = tf.expand_dims(image, 0)
            self.image = tf.transpose(image, [2,0,1])
        

    def __del__(self):
        if self.sess is not None:
            self.sess.close()

    def create_graph(self):
        uff_model = uff.from_tensorflow_frozen_model(self.model_file,
                                                     ['InceptionResnetV2/Logits/Predictions'], list_nodes = False)
                                                     
        G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.ERROR)
        parser = uffparser.create_uff_parser()
        parser.register_input('input_image', (3,512,512),0)
        parser.register_output('InceptionResnetV2/Logits/Predictions')
        
        engine = trt.utils.uff_to_trt_engine(G_LOGGER,uff_model,parser,1,1<<32)
        
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
           return dict {"id":"label"}
        """
        id2label = {}
        assert (os.path.exists(self.label_file))
        file_object = open(self.label_file)

        while 1:
            line = file_object.readline()
            if not line or line == '':
                break

            l = line.strip('\n').split(":")
            index, name = l[0], l[1]
            id2label[index] = name
        return id2label

    def run(self, img):
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

        if probabilities < self.output__threshold:
            return 0, [name, probabilities]
        else:
            return 1, [name, probabilities]
            

if __name__ == '__main__':
    img_handle = ImageProcess('/data/huang/behaviour_test/model/labels_363_9_4.txt', '/data/huang/behaviour_test/model/inception_resnet_v2_behaviour_363_9_13_518k_split.pb')
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
           # print('process this frame: %d completely' % frame_id)
            if ret2:
                if frame_id % 20 == 0:
                    print('{}:{}:{}'.format(t1-t0,result[1], result[0]))
        cap.release()
