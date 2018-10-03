import tensorflow as tf
import imageio
import cv2
import numpy as np
from PIL import Image
import random

_BATCH_NORM_MOMENTUM = 0.9
_BATCH_NORM_EPSILON = 1e-05
_LEAKY_RELU = 0.1
_NUM_CLASSES = 80
_ANCHORS = [(10, 13), (16, 30), (33, 23), (30, 61), (62, 45), (59, 119), (116, 90), (156, 198), (373, 326)]
_IMG_SIZE = (608, 608)


def _fixed_padding(inputs, kernel_size, data_format):
    mode = 'CONSTANT'
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    if data_format == 'NCHW':
        padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                        [pad_beg, pad_end], [pad_beg, pad_end]], mode=mode)
    else:
        padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                        [pad_beg, pad_end], [0, 0]], mode=mode)
    return padded_inputs


def parse_cfg(cfgfile):
    """
    Takes a configuration file

    Returns a list of blocks. Each blocks describes a block in the neural
    network to be built. Block is represented as a dictionary in the list

    """
    file = open(cfgfile, 'r')
    lines = file.read().split('\n')  # store the lines in a list
    lines = [x for x in lines if len(x) > 0]  # get read of the empty lines
    lines = [x for x in lines if x[0] != '#']  # get rid of comments
    lines = [x.rstrip().lstrip() for x in lines]  # get rid of fringe whitespaces

    block = {}
    blocks = []

    for line in lines:
        if line[0] == "[":  # This marks the start of a new block
            if len(block) != 0:  # If block is not empty, implies it is storing values of previous block.
                blocks.append(block)  # add it the blocks list
                block = {}  # re-init the block
            block["type"] = line[1:-1].rstrip()
        else:
            key, value = line.split("=")
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)

    return blocks


def _get_size(shape, data_format):
    if len(shape) == 4:
        shape = shape[1:]
    return shape[1:3] if data_format == 'NCHW' else shape[0:2]


def build_network(blocks, inputs, data_format="NHWC"):
    img_size = inputs.get_shape().as_list()[1:3]

    net_info = blocks[0]  # Captures the information about the input and pre-processing
    prev_filters = 3

    tf_data_format = 'channels_last'
    # transpose the inputs to NCHW
    if data_format == 'NCHW':
        inputs = tf.transpose(inputs, [0, 3, 1, 2])
        tf_data_format = 'channels_first'

    # normalize values to range [0..1]
    inputs = inputs / 255.0

    prev_output = None
    outputs_array = []
    yolo_outputs = []

    with tf.variable_scope('Conv'):
        for i, block in enumerate(blocks[1:]):
            if not outputs_array:
                prev_output = inputs
            else:
                prev_output = outputs_array[-1]

            if block["type"] == "convolutional":
                # with tf.variable_scope('conv' + str(i)) as scope:
                #     pass
                activation = block["activation"]
                try:
                    batch_normalize = int(block["batch_normalize"])
                    bias = False
                except:
                    batch_normalize = 0
                    bias = True

                filters = int(block["filters"])
                padding = int(block["pad"])
                kernel_size = int(block["size"])
                stride = int(block["stride"])

                if padding:
                    prev_output = _fixed_padding(prev_output, kernel_size, data_format)

                if bias:
                    prev_output = tf.layers.conv2d(
                        inputs=prev_output,
                        filters=filters,
                        strides=stride,
                        kernel_size=kernel_size,
                        padding='VALID',
                        # activation=tf.nn.relu,
                        use_bias=True,
                        data_format=tf_data_format
                    )
                else:
                    prev_output = tf.layers.conv2d(
                        inputs=prev_output,
                        filters=filters,
                        strides=stride,
                        kernel_size=kernel_size,
                        padding='VALID',
                        # activation=tf.nn.relu,
                        use_bias=False,
                        bias_initializer=None,
                        data_format=tf_data_format
                    )

                if batch_normalize:
                    prev_output = tf.layers.batch_normalization(
                        inputs=prev_output,
                        momentum=_BATCH_NORM_MOMENTUM,
                        epsilon=_BATCH_NORM_EPSILON,
                        scale=True,
                        training=False,
                        fused=None,
                        axis=-1 if data_format == 'NHWC' else 1
                    )
                if activation == "leaky":
                    prev_output = tf.nn.leaky_relu(prev_output, alpha=_LEAKY_RELU)

                outputs_array.append(prev_output)

            elif block["type"] == "upsample":
                stride = int(block["stride"])

                if data_format == 'NCHW':
                    prev_output = tf.transpose(prev_output, [0, 2, 3, 1])

                # if data_format == 'NCHW':
                #     new_height = prev_output.shape[2]
                #     new_width = prev_output.shape[3]
                # else:  # NHWC
                new_height = prev_output.shape[1]
                new_width = prev_output.shape[2]

                prev_output = tf.image.resize_nearest_neighbor(prev_output, (new_height * 2, new_width * 2))

                # revert back to NCHW if needed
                if data_format == 'NCHW':
                    prev_output = tf.transpose(prev_output, [0, 3, 1, 2])

                outputs_array.append(prev_output)

            elif block["type"] == "route":
                start = int(block["layers"].split(',')[0])
                try:
                    end = int(block["layers"].split(',')[1])
                except:
                    end = 0

                if start > 0:
                    start = start - i
                if end > 0:
                    end = end - i

                if end < 0:
                    if data_format == 'NHWC':
                        axis = 3
                    elif data_format == 'NCHW':
                        axis = 1
                    prev_output = tf.concat([outputs_array[i + start], outputs_array[i + end]], axis=axis)
                else:
                    prev_output = outputs_array[i + start]

                outputs_array.append(prev_output)

            elif block["type"] == "shortcut":
                _from = int(block["from"])
                activation = block["activation"]

                prev_output = tf.add(prev_output, outputs_array[i + _from])
                outputs_array.append(prev_output)

            elif block["type"] == "yolo":
                mask = list(map(int, block["mask"].split(",")))
                anchors = [_ANCHORS[m_i] for m_i in mask]
                predictions = prev_output
                num_anchors = len(anchors)
                shape = predictions.get_shape().as_list()
                grid_size = _get_size(shape, data_format)
                dim = grid_size[0] * grid_size[1]
                bbox_attrs = 5 + _NUM_CLASSES
                if data_format == 'NCHW':
                    predictions = tf.reshape(predictions, [-1, num_anchors * bbox_attrs, dim])
                    predictions = tf.transpose(predictions, [0, 2, 1])

                predictions = tf.reshape(predictions, [-1, num_anchors * dim, bbox_attrs])
                stride = (img_size[0] // grid_size[0], img_size[1] // grid_size[1])
                anchors = [(a[0] / stride[0], a[1] / stride[1]) for a in anchors]
                box_centers, box_sizes, confidence, classes = tf.split(predictions, [2, 2, 1, _NUM_CLASSES], axis=-1)
                box_centers = tf.nn.sigmoid(box_centers)
                confidence = tf.nn.sigmoid(confidence)

                grid_x = tf.range(grid_size[0], dtype=tf.float32)
                grid_y = tf.range(grid_size[1], dtype=tf.float32)
                a, b = tf.meshgrid(grid_x, grid_y)

                x_offset = tf.reshape(a, (-1, 1))
                y_offset = tf.reshape(b, (-1, 1))

                x_y_offset = tf.concat([x_offset, y_offset], axis=-1)
                x_y_offset = tf.reshape(tf.tile(x_y_offset, [1, num_anchors]), [1, -1, 2])

                box_centers = box_centers + x_y_offset
                box_centers = box_centers * stride

                anchors = tf.tile(anchors, [dim, 1])
                box_sizes = tf.exp(box_sizes) * anchors
                box_sizes = box_sizes * stride

                detections = tf.concat([box_centers, box_sizes, confidence], axis=-1)

                classes = tf.nn.sigmoid(classes)
                predictions = tf.concat([detections, classes], axis=-1)

                outputs_array.append(predictions)
                prev_output = predictions
                yolo_outputs.append(predictions)

            print("layer {:>15}  {:>3d} ".format(block["type"], i), prev_output.shape)

    return prev_output, outputs_array, yolo_outputs


def load_weights(var_list, weights_file):
    with open(weights_file, "rb") as fp:
        _ = np.fromfile(fp, dtype=np.int32, count=5)

        weights = np.fromfile(fp, dtype=np.float32)

    ptr = 0
    i = 0
    assign_ops = []

    while i < len(var_list) - 1:
        var1 = var_list[i]
        var2 = var_list[i + 1]
        if 'Conv' in var1.name.split('/'):
            # check type of next layer
            if 'batch_normalization' in var2.name:
                gamma, beta, mean, moving_variance = var_list[i + 1: i + 5]
                batch_norm_vars = [beta, gamma, mean, moving_variance]
                for var in batch_norm_vars:
                    shape = var.shape.as_list()
                    num_params = np.prod(shape)
                    var_weights = weights[ptr:ptr + num_params]
                    ptr += num_params
                    assign_ops.append(tf.assign(var, var_weights, validate_shape=True))
                # we move pointer on 4 points right, because we loaded 4 variables
                i += 4
            elif 'conv' in var2.name:
                # load biases
                bias = var2
                bias_shape = bias.shape.as_list()
                bias_params = np.prod(bias_shape)
                bias_weights = weights[ptr:ptr + bias_params].reshape(bias_shape)
                ptr += bias_params
                assign_ops.append(tf.assign(bias, bias_weights, validate_shape=True))
                i += 1
            # we can load weights of conv layer
            shape = var1.shape.as_list()
            num_params = np.prod(shape)

            var_weights = weights[ptr:ptr + num_params].reshape((shape[3], shape[2], shape[0], shape[1]))
            var_weights = np.transpose(var_weights, (2, 3, 1, 0))
            ptr += num_params
            assign_ops.append(tf.assign(var1, var_weights, validate_shape=True))
            i += 1

    return assign_ops


def detection_boxes(detections):
    c_x, c_y, w, h, attrs = tf.split(detections, [1, 1, 1, 1, -1], axis=-1)

    w_2 = w / 2.0
    h_2 = h / 2.0
    x1 = c_x - w_2
    y1 = c_y - h_2
    x2 = c_x + w_2
    y2 = c_y + h_2

    x1 = tf.nn.relu(x1)
    y1 = tf.nn.relu(y1)
    x2 = tf.nn.relu(x2)
    y2 = tf.nn.relu(y2)

    boxes = tf.concat([x1, y1, x2, y2], axis=-1)
    detections = tf.concat([boxes, attrs], axis=-1, name='concat_scaled_boxes')

    return detections


def _iou(box1, box2):
    b1_x1, b1_y1, b1_x2, b1_y2 = box1
    b2_x1, b2_y1, b2_x2, b2_y2 = box2
    x1 = max(b1_x1, b2_x1)
    x2 = min(b1_x2, b2_x2)
    y1 = max(b1_y1, b2_y1)
    y2 = min(b1_y2, b2_y2)

    intersecion_area = (x2 - x1) * (y2 - y1)
    intersecion_area = max(intersecion_area, 0)
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    # we add small epsilon of 1e-05 to avoid division by 0
    union_area = (b1_area + b2_area - intersecion_area + 1e-06)

    return intersecion_area / union_area


def non_maxima_suppression(predictions_with_boxes, confidence_threshold, iou_threshold=0.4):
    conf_mask = np.expand_dims((predictions_with_boxes[:, :, 4] > confidence_threshold), -1)
    predictions = predictions_with_boxes * conf_mask
    result = {}
    for i, image_pred in enumerate(predictions):
        shape = image_pred.shape
        non_zero_idxs = np.nonzero(image_pred)
        image_pred = image_pred[non_zero_idxs]
        image_pred = image_pred.reshape(-1, shape[-1])

        bbox_attrs = image_pred[:, :5]
        classes = image_pred[:, 5:]
        classes = np.argmax(classes, axis=-1)

        unique_classes = list(set(classes.reshape(-1)))

        for cls in unique_classes:
            cls_mask = classes == cls
            cls_boxes = bbox_attrs[np.nonzero(cls_mask)]
            cls_boxes = cls_boxes[cls_boxes[:, 4].argsort()[::-1]]

            cls_scores = cls_boxes[:, -1]
            cls_boxes = cls_boxes[:, :-1]

            while len(cls_boxes) > 0:
                box = cls_boxes[0]
                score = cls_scores[0]
                if not cls in result:
                    result[cls] = []
                result[cls].append((box, score))
                cls_boxes = cls_boxes[1:]
                ious = np.array([_iou(box, x) for x in cls_boxes])
                iou_mask = ious < iou_threshold
                cls_boxes = cls_boxes[np.nonzero(iou_mask)]
                cls_scores = cls_scores[np.nonzero(iou_mask)]
    return result


def convert_to_original_size(box, size, original_size):
    ratio = original_size / size
    box = box.reshape(2, 2) * ratio
    return list(box.reshape(-1))


def draw_boxes(boxes, img, cls_names, detection_size):
    x_ratio = img.shape[1] / detection_size[1]
    y_ratio = img.shape[0] / detection_size[0]

    res = img
    for cls, bboxs in boxes.items():
        color = (random.randint(0, 255),
                 random.randint(0, 255),
                 random.randint(0, 255))

        for box, score in bboxs:
            p1 = (int(box[0] * x_ratio), int(box[1] * y_ratio))
            p2 = (int(box[2] * x_ratio), int(box[3] * y_ratio))
            res = cv2.rectangle(res, p1, p2, color)
            # draw.text(box[:2], '{} {:.2f}%'.format(cls_names[cls], score * 100), fill=color)

    return res


def yolov3(_blocks, inputs, data_format):
    output, outputs_array, yolo_outputs = build_network(_blocks, inputs, data_format)
    d1, d2, d3 = yolo_outputs
    detections = tf.concat([d1, d2, d3], axis=1)
    boxes = detection_boxes(detections)

    return boxes


def print_detections(detections):
    for cls, boxes in detections.items():
        for box in boxes:
            print(cls, box)


if __name__ == '__main__':
    im_op = Image.open("images/kite.jpg")
    img_resized = im_op.resize(size=_IMG_SIZE)

    original_img = cv2.imread("images/kite.jpg")  # imageio.imread("images/kite.jpg")

    img = cv2.resize(original_img, dsize=(_IMG_SIZE[0], _IMG_SIZE[1]), interpolation=cv2.INTER_CUBIC)
    img = img[..., ::-1]

    data_format = 'NCHW'

    _blocks = parse_cfg("cfg/yolov3.cfg")

    inputs = tf.placeholder(tf.float32, [1, _IMG_SIZE[0], _IMG_SIZE[1], 3])

    boxes = yolov3(_blocks, inputs, data_format)

    for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
        print(var)

    global_variables = tf.global_variables()
    assign_ops = load_weights(global_variables, "yolov3-coco-wiout-spp.weights")

    saver = tf.train.Saver()

    with tf.Session() as sess:
        # saver.restore(sess, 'checkpoints/yolov3')

        r = sess.run(assign_ops)
        # sess.run(tf.global_variables_initializer())
        # uninit_vars = sess.run(tf.report_uninitialized_variables())
        # print("uninit_vars", uninit_vars)

        _boxes = sess.run(boxes, feed_dict={inputs: [img]})
        # _boxes = sess.run(boxes, feed_dict={inputs: [np.array(img_resized, dtype=np.float32)]})
        results = non_maxima_suppression(_boxes, confidence_threshold=0.05, iou_threshold=0.4)
        original_img = draw_boxes(results, original_img, [], (_IMG_SIZE[0], _IMG_SIZE[1]))
        cv2.imwrite("my-result-cv.jpg", original_img)

        saver.save(sess, 'checkpoints/yolov3')

        # for n in tf.get_default_graph().as_graph_def().node:
        #     print(n.name)

    print_detections(results)
