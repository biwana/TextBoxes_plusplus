# This script only includes detection.
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# %matplotlib inline
import time
import math
from nms import nms
from crop_image import crop_image

import caffe
# Make sure that caffe is on the python path:
caffe_root = './'  # this file is expected to be in {caffe_root}/examples
import os
os.chdir(caffe_root)
import sys
sys.path.insert(0, 'python')

caffe.set_device(0)
caffe.set_mode_gpu()

config = {
	'model_def' : './models/deploy.prototxt',
	'model_weights' : './models/model_icdar15.caffemodel',
	'img_dir' : './test_images/',
	'det_visu_path' : './tbpp_results/',
	'det_save_dir' : './tbpp_results/',
	'crop_dir' : './tbpp_crop/',
	'input_height' : 768,
	'input_width' : 768,
	'overlap_threshold' : 0.2,
	'det_score_threshold' : 0.2,
	'visu_detection' : True,
}

def get_images():
    '''
    find image files in test data path
    :return: list of files found
    '''
    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    for parent, dirnames, filenames in os.walk(config['img_dir']):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    print('Find {} images'.format(len(files)))
    return files

def prepare_network(config):
	# modified to not rely on image
	net = caffe.Net(config['model_def'],	 # defines the structure of the model
                config['model_weights'],  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

	transformer = caffe.io.Transformer({'data': (1,3,config['input_height'], config['input_width'])})
	transformer.set_transpose('data', (2, 0, 1))
	transformer.set_mean('data', np.array([104,117,123])) # mean pixel
	transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
	transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

	net.blobs['data'].reshape(1,3,config['input_height'], config['input_width'])

	return net, transformer


def extract_detections(detections, det_score_threshold, image_height, image_width):
	det_conf = detections[0,0,:,2]
	det_x1 = detections[0,0,:,7]
	det_y1 = detections[0,0,:,8]
	det_x2 = detections[0,0,:,9]
	det_y2 = detections[0,0,:,10]
	det_x3 = detections[0,0,:,11]
	det_y3 = detections[0,0,:,12]
	det_x4 = detections[0,0,:,13]
	det_y4 = detections[0,0,:,14]
	# Get detections with confidence higher than 0.6.
	top_indices = [i for i, conf in enumerate(det_conf) if conf >= det_score_threshold]
	top_conf = det_conf[top_indices]
	top_x1 = det_x1[top_indices]
	top_y1 = det_y1[top_indices]
	top_x2 = det_x2[top_indices]
	top_y2 = det_y2[top_indices]
	top_x3 = det_x3[top_indices]
	top_y3 = det_y3[top_indices]
	top_x4 = det_x4[top_indices]
	top_y4 = det_y4[top_indices]

	bboxes=[]
	for i in xrange(top_conf.shape[0]):
		x1 = int(round(top_x1[i] * image_width))
		y1 = int(round(top_y1[i] * image_height))
		x2 = int(round(top_x2[i] * image_width))
		y2 = int(round(top_y2[i] * image_height))
		x3 = int(round(top_x3[i] * image_width))
		y3 = int(round(top_y3[i] * image_height))
		x4 = int(round(top_x4[i] * image_width))
		y4 = int(round(top_y4[i] * image_height))
		x1 = max(1, min(x1, image_width - 1))
		x2 = max(1, min(x2, image_width - 1))
		x3 = max(1, min(x3, image_width - 1))
		x4 = max(1, min(x4, image_width - 1))
		y1 = max(1, min(y1, image_height - 1))
		y2 = max(1, min(y2, image_height - 1))
		y3 = max(1, min(y3, image_height - 1))
		y4 = max(1, min(y4, image_height - 1))
		score = top_conf[i]
		bbox=[x1,y1,x2,y2,x3,y3,x4,y4,score]
		bboxes.append(bbox)
	return bboxes

def apply_quad_nms(bboxes, overlap_threshold):
	dt_lines = sorted(bboxes, key=lambda x:-float(x[8]))
	nms_flag = nms(dt_lines, overlap_threshold)
	results=[]
	for k,dt in enumerate(dt_lines):
		if nms_flag[k]:
			if dt not in results:
				results.append(dt)
	return results

def save_and_visu(image, image_name, results, config):
	# modified for multiple
	det_save_path=os.path.join(config['det_save_dir'], image_name.split('.')[0]+'.txt')
	det_fid = open(det_save_path, 'wt')
	if config['visu_detection']:
		# visulization
		plt.clf()
		plt.imshow(image)
		currentAxis = plt.gca()
	for result in results:
		score = result[-1]
		x1 = result[0]
		y1 = result[1]
		x2 = result[2]
		y2 = result[3]
		x3 = result[4]
		y3 = result[5]
		x4 = result[6]
		y4 = result[7]
		result_str=str(x1)+','+str(y1)+','+str(x2)+','+str(y2)+','+str(x3)+','+str(y3)+','+str(x4)+','+str(y4)+','+str(score)+'\r\n'
		det_fid.write(result_str)
		if config['visu_detection']:
			quad = np.array([[x1,y1],[x2,y2],[x3,y3],[x4,y4]])
			color_quad='r'
			currentAxis.add_patch(plt.Polygon(quad, fill=False, edgecolor=color_quad, linewidth=2))

	det_fid.close()
	if config['visu_detection']:
		plt.axis('off')
		plt.savefig(os.path.join(config['det_visu_path'], image_name), dpi=300)

# detection
net, transformer = prepare_network(config)

img_list = get_images()
for img_file in img_list:
	image=caffe.io.load_image(os.path.join(config['img_dir'], img_file))
	transformed_image = transformer.preprocess('data', image)
	net.blobs['data'].data[...] = transformed_image

	image_height, image_width, channels=image.shape
	detections = net.forward()['detection_out']
	# Parse the outputs.
	bboxes = extract_detections(detections, config['det_score_threshold'], image_height, image_width)
	# apply non-maximum suppression
	results = apply_quad_nms(bboxes, config['overlap_threshold'])
	save_and_visu(image, img_file, results, config)
	crop_image(img_file, results, os.path.join(config['crop_dir']))
print('detection finished')

	
