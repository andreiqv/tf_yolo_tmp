#!/usr/bin/env python
# -*- coding: utf-8 -*-
# According to video Нейронные сети_ быстрый инференс на GPU с помощью TensorRT _ Дмитрий Коробченко (NVIDIA) [720p]

import tensorflow as tf
import tensorrt as trt
import numpy as np
import scipy.misc
import utils
import timer


NUM_CLASSES = 112
CLASSES = [str(i) for i in range(NUM_CLASSES)] # ADJUST

#import uff
#from tensorrt.parsers import uffparser

use_hub_model = False

if use_hub_model:
	#FROZEN_FPATH = '/home/andrei/Data/Datasets/Scales/pb/output_graph.pb'
	ENGINE_FPATH = '/home/andrei/Data/Datasets/Scales/pb/hub_model_engine.plan'
	#INPUT_NODE = 'Placeholder-x'
	#OUTPUT_NODE = 'final_result'
	INPUT_SIZE = [3, 299, 299]

else:
	#FROZEN_FPATH = '/root/tmp/saved_model_inception_resnet.pb'
	#ENGINE_FPATH = '/root/tmp/engine.plan'
	#FROZEN_FPATH = 'saved_model_full.pb'
	ENGINE_FPATH = 'saved_model_full.plan'
	#INPUT_NODE = 'Placeholder-x'
	#OUTPUT_NODE = 'sigmoid_out'
	INPUT_SIZE = [3, 299, 299]

#MAX_BATCH_SIZE = 1
#MAX_WORKSPACE = 1 << 20

CROP_SIZE = tuple(INPUT_SIZE[1:])


def prepare_image(image_path):

	img_in = scipy.misc.imread(image_path, mode='RGB')	
	img = img_in.astype(np.float32)
	img = utils.resize_and_crop(img, CROP_SIZE)
	img = img.transpose(2, 0, 1)
	return img


#def inference(engine, img):
#	output = engine.infer(img)


if __name__ == '__main__':

	engine = trt.lite.Engine(PLAN=ENGINE_FPATH)

	image_path = 'images/1.jpg'	
	img = prepare_image(image_path)

	timer.timer('start')
	for _ in range(5):
		output = engine.infer(img)
		print(output)
		timer.timer()

	#inference(engine, img)
	#