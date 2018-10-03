import tensorflow as tf
import uff
import tensorrt as trt
from tensorrt.parsers import uffparser

use_hub_model = False

if use_hub_model:
	FROZEN_FPATH = '/home/andrei/Data/Datasets/Scales/pb/output_graph.pb'
	ENGINE_FPATH = '/home/andrei/Data/Datasets/Scales/pb/hub_model_engine.plan'
	INPUT_NODE = 'Placeholder-x'
	OUTPUT_NODE = 'final_result'
	INPUT_SIZE = [3, 299, 299]
else:
	#FROZEN_FPATH = '/root/tmp/saved_model_inception_resnet.pb'
	#ENGINE_FPATH = '/root/tmp/engine.plan'
	FROZEN_FPATH = 'pb_model/yolov3-coco-nhwc.pb'
	ENGINE_FPATH = 'pb_model/yolov3-coco-nhwc.plan'
	INPUT_NODE = 'Placeholder'
	OUTPUT_NODE = 'concat_scaled_boxes'
	INPUT_SIZE = [3, 608, 608]

MAX_BATCH_SIZE = 1
MAX_WORKSPACE = 1 << 20

# convert TF frozen graph to UFF graph
uff_model = uff.from_tensorflow_frozen_model(FROZEN_FPATH, [OUTPUT_NODE])

# create UFF parser and logger
parser = uffparser.create_uff_parser()
parser.register_input(INPUT_NODE, INPUT_SIZE, 0)
parser.register_output(OUTPUT_NODE)
G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.INFO)

# Build optimized inference engine
engine = trt.utils.uff_to_trt_engine(G_LOGGER, uff_model, parser, MAX_BATCH_SIZE, MAX_WORKSPACE)

# Save inference engine
trt.utils.write_engine_to_file(ENGINE_FPATH, engine.serialize())

# Cleaning Up
parser.destroy()
engine.destroy()


"""

engine = trt.utils.uff_to_trt_engine(G_LOGGER, uff_model, parser, MAX_BATCH_SIZE, MAX_WORKSPACE)
[TensorRT] ERROR: UFFParser: Validator error: input/BottleneckInputPlaceholder: Unsupported operation _PlaceholderWithDefault
[TensorRT] ERROR: Failed to parse UFF model stream
  File "/usr/lib/python3.5/dist-packages/tensorrt/utils/_utils.py", line 255, in uff_to_trt_engine
    assert(parser.parse(stream, network, model_datatype))
Traceback (most recent call last):
  File "/usr/lib/python3.5/dist-packages/tensorrt/utils/_utils.py", line 255, in uff_to_trt_engine
    assert(parser.parse(stream, network, model_datatype))
AssertionError

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/usr/lib/python3.5/dist-packages/tensorrt/utils/_utils.py", line 263, in uff_to_trt_engine
    raise AssertionError('UFF parsing failed on line {} in statement {}'.format(line, text))
AssertionError: UFF parsing failed on line 255 in statement assert(parser.parse(stream, network, model_datatype))
>>> 

----------


[TensorRT] INFO: Tactic 131 time 0.556384
[TensorRT] INFO: Tactic 132 time 0.8152
[TensorRT] INFO: Tactic 138 time 0.921984
[TensorRT] INFO: Tactic 145 time 1.10986
[TensorRT] INFO: Tactic 150 time 1.47024
[TensorRT] INFO: Tactic 155 time 1.11302
[TensorRT] INFO: Tactic 157 time 0.52672
[TensorRT] INFO: Tactic 160 time 0.870144
[TensorRT] INFO: Tactic 163 time 0.7656
[TensorRT] INFO: Tactic 166 time 0.775744
[TensorRT] INFO: 
[TensorRT] INFO: --------------- Timing module_1_apply_default/InceptionResnetV2/InceptionResnetV2/Conv2d_7b_1x1/Conv2D + module_1_apply_default/InceptionResnetV2/InceptionResnetV2/Conv2d_7b_1x1/Relu(14)
[TensorRT] INFO: Tactic 1363534230700867617 time 0.632384
[TensorRT] INFO: Tactic 1642270411037877776 time 0.65536
[TensorRT] INFO: Tactic 5443600094180187792 time 0.777696
[TensorRT] INFO: Tactic 5552354567368947361 time 0.735296
[TensorRT] INFO: Tactic 5824828673459742858 time 0.641568
[TensorRT] INFO: Tactic -6618588952828687390 time 0.841504
[TensorRT] INFO: Tactic -2701242286872672544 time 0.675104
[TensorRT] INFO: Tactic -2535759802710599445 time 0.676384
[TensorRT] INFO: Tactic -675401754313066228 time 0.696672
[TensorRT] INFO: 
[TensorRT] INFO: --------------- Timing module_1_apply_default/InceptionResnetV2/InceptionResnetV2/Conv2d_7b_1x1/Conv2D + module_1_apply_default/InceptionResnetV2/InceptionResnetV2/Conv2d_7b_1x1/Relu(1)
[TensorRT] INFO: Tactic 0 time 1.05878
[TensorRT] INFO: Tactic 1 time 0.698592
[TensorRT] INFO: Tactic 2 time 0.732448
[TensorRT] INFO: Tactic 4 scratch requested: 7378575360, available: 1048576
[TensorRT] INFO: Tactic 5 scratch requested: 438438304, available: 1048576
[TensorRT] INFO: --------------- Chose 2 (107)
[TensorRT] INFO: 
[TensorRT] INFO: --------------- Timing module_1_apply_default/hub_output/feature_vector/SpatialSqueeze(8)
[TensorRT] INFO: Tactic -1 is the only option, timing skipped
[TensorRT] INFO: 
[TensorRT] INFO: --------------- Timing MatMul(6)
[TensorRT] INFO: Tactic 0 time 0.025344
[TensorRT] INFO: Tactic 4 time 0.025728
[TensorRT] INFO: Tactic 1 time 0.023904
[TensorRT] INFO: Tactic 5 time 0.0232
[TensorRT] INFO: 
[TensorRT] INFO: --------------- Timing MatMul(15)
[TensorRT] INFO: --------------- Chose 6 (5)
[TensorRT] INFO: 
[TensorRT] INFO: --------------- Timing add + sigmoid_out(10)
[TensorRT] INFO: Tactic 0 is the only option, timing skipped
[TensorRT] INFO: Formats and tactics selection completed in 62.0438 seconds.
[TensorRT] INFO: After reformat layers: 243 layers
[TensorRT] INFO: Block size 5531904
[TensorRT] INFO: Block size 2766080
[TensorRT] INFO: Block size 1568000
[TensorRT] INFO: Block size 1254400
[TensorRT] INFO: Block size 1048576
[TensorRT] INFO: Total Activation Memory: 12168960
[TensorRT] INFO: Data initialization and engine generation completed in 0.380873 seconds.

"""