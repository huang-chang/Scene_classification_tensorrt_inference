#from tensorflow.python.platform import gfile
#import tensorflow as tf
#import cv2
#import time
#import numpy as np
#import tensorflow.contrib.tensorrt as trt
#
#image_to_tensor = np.zeros((1,224,224,3))
#
#def getResnet():
#    with gfile.FastGFile('resnetV150_frozen.pb', 'rb') as f:
#        graph_def = tf.GraphDef()
#        graph_def.ParseFromString(f.read())
#    return graph_def
#
#def getFP32():
#    trt_graph = trt.create_inference_graph(getResnet(),['resnet_v1_50/predictions/Reshape_1'],
#                                           max_batch_size = 1, max_workspace_size_bytes = 1<<32,
#                                           precision_mode = 'FP32')
#    return trt_graph
#print('----')    
#graph = getFP32()
#print('****')
#tf.reset_default_graph()
#    
#out = tf.import_graph_def(graph, return_elements = ['input', 'resnet_v1_50/predictions/Reshape_1'])
#input_placeholder = out[0].outputs[0]
#reshape = out[1].outputs[0]
#
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.1)
#sess = tf.Session(config = tf.ConfigProto(gpu_options=gpu_options))
#
#def image_run(image):
#    pro = sess.run(reshape, {input_placeholder: image})
#    return pro
#    
#
#image = np.ones((1,224,224,3))
#for i in range(1000):
#    t0 = time.time()
#    pro = image_run(image)
#    t1 = time.time()
#    if i % 100 == 0:  
#        print(str(i).zfill(6),t1-t0)
        
        
#cap = cv2.VideoCapture('/data/huang/behaviour_test/test_video_huanlesong_1/huanlesong_1.mp4')
#if cap.isOpened():
#    ret, img = cap.read()
#    frame_id = 0
#    while ret:
#        img = cv2.resize(img,(224,224))
#        image[0,:,:,:] = img
#        t0 = time.time()
#        pro = image_run(image)
#        t1 = time.time()
#        if frame_id % 100 == 0:  
#            print(str(frame_id).zfill(6),t1-t0)
#        frame_id += 1
#        ret,img = cap.read()
#    cap.release()
        
#from tensorflow.python.platform import gfile
#import tensorflow as tf
#import tensorflow.contrib.tensorrt as trt
##import tensorrt as trt
#import os
##os.environ["CUDA_VISIBLE_DEVICES"]="0"
#        
#def getInceptionResnetV2():
#    with gfile.FastGFile('model/inception_resnet_v2_behaviour_337_5_22_411k_split.pb', 'rb') as f:
#        graph_def = tf.GraphDef()
#        graph_def.ParseFromString(f.read())
#    return graph_def
##graph_def = getInceptionResnetV2()
#
#def getFP32(batch_size=1, workspace_size=1<<32):
#    
#    trt_graph = trt.create_inference_graph(getInceptionResnetV2(),
#                                           ['InceptionResnetV2/Logits/Predictions'],
#                                           max_batch_size=batch_size,
#                                           max_workspace_size_bytes=workspace_size,
#                                           precision_mode='FP32')
#    print('OK')
#
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
#
#getFP32()
import tensorrt as trt
import uff
from tensorrt.parsers import uffparser
import numpy as np
import pycuda.driver as cuda
import time
import tensorflow as tf
import cv2
from tensorflow.python.platform import gfile


uff_model = uff.from_tensorflow_frozen_model('model/inception_resnet_v2_behaviour_337_5_22_411k_split.pb',
                                             ['InceptionResnetV2/Logits/Predictions'], list_nodes = False)
                                             
G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.ERROR)
parser = uffparser.create_uff_parser()
parser.register_input('input_image', (3,512,512),0)
parser.register_output('InceptionResnetV2/Logits/Predictions')

engine = trt.utils.uff_to_trt_engine(G_LOGGER,uff_model,parser,1,1<<31)

parser.destroy()

#image_path = '/home/ss/Desktop/00002464.jpg'
#im = Image.open(image_path)
#arr = np.array(im)
#img = arr.ravel()
#img = img.astype(np.float32)

runtime = trt.infer.create_infer_runtime(G_LOGGER)
context = engine.create_execution_context()

output = np.empty(337, dtype = np.float32)
#d_input = cuda.mem_alloc(1 * 512 * 512 * 3 * 4)

d_input = cuda.mem_alloc(1 * 512 * 512 * 3 * 4)
d_output = cuda.mem_alloc(1 * 337 * 4)

bindings = [int(d_input), int(d_output)]
stream = cuda.Stream()

image_pixel = []

def image_processing(image):
    image = image.astype(np.float32)
    image = np.multiply(image, 1.0 / 255.0)
    print(image.shape)
    image = cv2.resize(image, (512,512), interpolation=cv2.INTER_LINEAR)
    image = 2 * (image - 0.5)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_pixel.append(image[220][220].tolist())
    #print('before transpose: {}'.format(image.shape))
    image = image.transpose((2,0,1))
    #print('after transpose: {}'.format(image.shape))
    image_onedim = image.ravel()
    return image_onedim
    
label = []
result = []

def get_label():
    with open('model/labels_337_4_10.txt', 'rb') as f:
        for i in f.readlines():
            label.append(i.strip().split(':')[1])
    return label
get_label()
cap = cv2.VideoCapture('/data/huang/behaviour_test/test_video_huanlesong_1/huanlesong_1.mp4')
frame = 0
if cap.isOpened():
    ret, photo = cap.read()
    while ret and photo is not None:
        t0 = time.time()
        photo_number = image_processing(photo)
        t1 = time.time()
        #print(photo_number.dtype,type(photo_number), photo_number.shape)
        cuda.memcpy_htod_async(d_input, photo_number, stream)
        context.enqueue(1, bindings, stream.handle, None)
        cuda.memcpy_dtoh_async(output, d_output, stream)
        stream.synchronize()
        t2 = time.time()
        index = np.argmax(output)
        print(frame,t1-t0,t2-t1,label[index],output[index])
        result.append([frame,label[index],output[index]])
        ret, photo = cap.read()
        frame += 1
        if frame == 1000:
            break
    
cap.release()
context.destroy()
engine.destroy()
runtime.destroy()

#with tf.device('/gpu:1'):
image_placeholder = tf.placeholder(tf.uint8, shape=(None, None, 3))
image_rgb = tf.reverse(image_placeholder, [-1])
image = tf.image.convert_image_dtype(image_rgb, dtype=tf.float32)
image = tf.expand_dims(image,0)
image = tf.image.resize_bilinear(image, [512,512], align_corners=False)
image = tf.squeeze(image, [0])
image = tf.subtract(image, 0.5)
image = tf.multiply(image, 2.0)
image = tf.expand_dims(image, 0)
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.05)
#tf.Graph().as_default()
#sess = tf.Session(config = tf.ConfigProto(gpu_options = gpu_options))
#sess = tf.Session(config=config)
sess = tf.Session()

with gfile.FastGFile('model/inception_resnet_v2_behaviour_337_5_22_411k_split.pb', 'rb') as f:
    graph = tf.GraphDef()
    graph.ParseFromString(f.read())
    out = tf.import_graph_def(graph, return_elements = ['input_image', 'InceptionResnetV2/Logits/Predictions'])
    input_placeholder = out[0].outputs[0]
    predictions = out[1].outputs[0]

cap = cv2.VideoCapture('/data/huang/behaviour_test/test_video_huanlesong_1/huanlesong_1.mp4')
frame = 0
if cap.isOpened():
    ret, photo = cap.read()
    while ret and photo is not None:
        t0 = time.time()
        processed_image = sess.run(image, {image_placeholder: photo})
        print(processed_image.dtype)
        image_pixel[frame].extend(processed_image[0][220][220].tolist())
        result_image = sess.run(predictions, {input_placeholder: processed_image})
        t1 = time.time()
        result_image = np.squeeze(result_image)
        index = np.argmax(result_image)
        print(frame,t1-t0,label[index],result_image[index])
        result[frame].extend([label[index],result_image[index]])
        ret, photo = cap.read()
        frame += 1
        if frame == 1000:
            break
sess.close()
cap.release()
with open('result.txt', 'w') as f:
    for i in result:
        f.write('{}\n'.format(i))
with open('pixel.txt', 'w') as f:
    for i in image_pixel:
        m = i[0] - i[3]
        l = i[1] - i[4]
        k = i[2] - i[5]
        i.extend([m,l,k])
        f.write('{}\n'.format(i[6:]))

