import tensorflow as tf
from tensorflow.python.framework import graph_io
from tensorflow.python.tools import inspect_checkpoint as chkp
from darknet_tf import _fixed_padding
from darknet_tf import _IMG_SIZE, yolov3, parse_cfg

CHECKPOINTS = "checkpoints/yolov3"
OUTPUT_MODEL_PB = "yolov3-coco-nchw.pb"


def print_info():
    res = chkp.print_tensors_in_checkpoint_file(CHECKPOINTS, tensor_name='', all_tensors=True)
    print(res)
    for n in tf.get_default_graph().as_graph_def().node:
        print(n.name)


graph = tf.Graph()

with graph.as_default():
    _blocks = parse_cfg("cfg/yolov3.cfg")
    data_format = 'NCHW'

    inputs = tf.placeholder(tf.float32, [None, _IMG_SIZE[0], _IMG_SIZE[1], 3])
    boxes = yolov3(_blocks, inputs, data_format)

    OUTPUT = ["concat_scaled_boxes"]
    saver = tf.train.Saver()

    with tf.Session(graph=graph) as sess:
        saver.restore(sess, CHECKPOINTS)
        print_info()
        graphdef_inf = tf.graph_util.remove_training_nodes(graph.as_graph_def())
        graphdef_frozen = tf.graph_util.convert_variables_to_constants(sess, graphdef_inf, OUTPUT)
        graph_io.write_graph(graphdef_frozen, "./pb_model", OUTPUT_MODEL_PB, as_text=False)
        for node in graphdef_frozen.node:
            print(node.name)
