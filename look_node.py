import tensorflow as tf

model_path = 'model/inception_resnet_v2_behaviour_309_4_10_434k.pb'

def getInceptionResnetV2(model_path):
    with tf.gfile.FastGFile(model_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def)
getInceptionResnetV2(model_path)
sess = tf.Session()
for i in tf.get_default_graph():
    print(i)