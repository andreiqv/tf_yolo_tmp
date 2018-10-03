import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import time
from darknet_tf import non_maxima_suppression

image_shape = (608, 608)
INPUT_NODE = "Placeholder"
OUTPUT_NODE = "concat_scaled_boxes"

graph_def = tf.GraphDef()

with tf.gfile.GFile("pb_model/yolov3-coco-nhwc.pb", "rb") as f:
    graph_def.ParseFromString(f.read())

graph = tf.Graph()


def prepare_image(img_path):
    # im = Image.open(img_path).resize(image_shape, Image.BICUBIC)
    im = cv2.imread(img_path, cv2.IMREAD_COLOR)
    im = cv2.resize(im, image_shape, interpolation=cv2.INTER_CUBIC)

    im = im[..., ::-1]
    result = np.array(im)  # / 255.0
    # result = result[None, ...]
    return result


with graph.as_default():
    net_inp, net_out = tf.import_graph_def(
        graph_def, return_elements=[INPUT_NODE, OUTPUT_NODE]
    )

    with tf.Session(graph=graph) as sess:
        i = 0
        images = ["images/kite.jpg", "images/kite.jpg", "images/img1.jpg",
                  "images/vkuswill.jpg", "images/vkuswill.jpg", "images/vkuswill.jpg",
                  "images/vkuswill.jpg", "images/vkuswill.jpg", "images/vkuswill.jpg",
                  "images/vkuswill.jpg", "images/vkuswill.jpg", "images/vkuswill.jpg",
                  "images/vkuswill.jpg", "images/vkuswill.jpg",
                  "images/vkuswill.jpg",
                  ]
        images = ['images/5.jpg', 'images/30.jpg']                   
        while True:
            i += 1
            # im = prepare_image(images[i % 2], image_shape)
            ims = list(map(prepare_image, images))
            arry = np.array(ims)

            start = time.time()
            out = sess.run(net_out.outputs[0], feed_dict={net_inp.outputs[0]: ims})

            # out = non_maxima_suppression(out, 0.2, 0.4)
            # print(out)

            end = time.time()
            print("fps:", len(images) / (end - start))
