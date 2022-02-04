from PIL import Image
import numpy as np

import re
import math
import random

from rknn.api import rknn

if __name__ == '__main__':
	rknn = RKNN()
	
	print('--> Loading model')
	rknn.load_darknet(model = './model/yolov3.cfg', weight = './model/yolov3.weights')

	print('done')

	rknn.config(channel_mean_value = '0 0 0 255', reorder_channel = '0 1 2', batch_size = 1)

	print('--> building model')
	rknn.build(do_quantization = True, dataset = './dataset_416x416.txt', pre_compile = True)
	
	print('done')

	rknn.export_rknn('./model_out/yolov3.rknn')

	exit(0)
