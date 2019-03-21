#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# file: image_process.py
# author: jiangqr
# data: 2017.1.3
# note: receive and send message for web
#
import tensorflow as tf
import os
import numpy as np
import cv2
import sys
import config
mutex = config.mutex


class ImageProcess:
    """"""
    def __init__(self):
        self.label_file = config.LABEL_FILE
        self.model_file = config.MODEL_FILE
        self.output__threshold = config.OUTPUT_THRESHOLD
        self.id2name = self.get_labels()
        tf_config = tf.ConfigProto(allow_soft_placement=True)
        tf_config.gpu_options.per_process_gpu_memory_fraction = 0.1  # 占用GPU20%的显存
        tf_config.gpu_options.allow_growth = True
        tf.Graph().as_default()
        self.sess = tf.Session(config=tf_config)
        print('************scene gpu id:{}'.format(1))
        with tf.device('/gpu:{}'.format(1)):
            self.create_graph()
            if config.BOOL_V2_MODEL:
                #self.softmax = self.sess.graph.get_tensor_by_name("InceptionResnetV2/Logits/Predictions:0")
                self.image_placeholder = tf.placeholder(tf.uint8, shape=(None,None,3))
                image_rgb = tf.reverse(self.image_placeholder, [-1])
                image = tf.image.convert_image_dtype(image_rgb, dtype=tf.float32)
                image = tf.expand_dims(image,0)
                image = tf.image.resize_bilinear(image, [512, 512],
                                                       align_corners=False) 
                image = tf.squeeze(image, [0])
                image = tf.subtract(image, 0.5)
                image = tf.multiply(image, 2.0)
                self.image = tf.expand_dims(image,0)
            else:
                self.softmax = self.sess.graph.get_tensor_by_name("InceptionV4/Logits/Predictions:0")
                    
            # Loading the injected placeholder
            #self.input_placeholder = self.sess.graph.get_tensor_by_name("input_image:0")

    def __del__(self):
        if self.sess is not None:
            self.sess.close()

    def create_graph(self):
        """"""
        with tf.gfile.FastGFile(self.model_file, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            #_ = tf.import_graph_def(graph_def, name='')
            out = tf.import_graph_def(graph_def, return_elements = ['ExpandDims', 'InceptionResnetV2/Logits/Predictions'])
            self.input_placeholder = out[0].outputs[0]
            self.softmax = out[1].outputs[0]

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

        #ret, buf = cv2.imencode('.jpg', img)
        #if not ret:
        #    return 0, []

        #img_string = buf.tostring()
        probabilities = self.sess.run(self.softmax, {self.input_placeholder: self.sess.run(self.image, {self.image_placeholder: img})})
        #probabilities = self.sess.run(self.softmax, {self.input_placeholder: img_string})
        probabilities = np.squeeze(probabilities)
        (index,) = np.where(probabilities == probabilities.max())
        name, bool_valid = self.id2name[str(index[0])]

        if probabilities.max() < self.output__threshold or not bool_valid:
            return 0, []
        else:
            return 1, [name, np.max(probabilities)]


if __name__ == '__main__':
    img_handle = ImageProcess()
    cap = cv2.VideoCapture('../data/video/wylp.mp4')
    if cap.isOpened():
        ret, img = cap.read()
        frame_id = 0
        while ret:
            ret2, result = img_handle.run(img)
            ret, img = cap.read()
            frame_id += 1
            print 'process this frame: %d completely' % frame_id
            # print 'the scene name is %s, the score is %f' % (result[0], result[1])
        cap.release()
