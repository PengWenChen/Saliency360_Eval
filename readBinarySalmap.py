#! /usr/bin/env python3
'''
Implemented by Erwan DAVID (IPI, LS2N, Nantes, France), 2018

E. DAVID, J. Guttiérez, A Coutrot, M. Perreira Da Silva, P. Le Callet (2018). A Dataset of Head and Eye Movements for 360° Videos. ACM MMSys18, dataset and toolbox track
'''

import matplotlib.pyplot as plt
import numpy as np
import re, os, sys
import pdb

"""
Usage:
	python3 readBinarySalmap.py [Frame index] [Path to binary]
"""

# Select frame to display (default: 4)
try: Nframe = int(sys.argv[1])
except: Nframe = 0

try: path2file = sys.argv[2]
except:
	# Example files, Choose at random between type of data
	choice = np.random.randint(2)
	path2file = ["../H/SalMaps/Hsalmap_50_2048x1024_32b.bin", 
				 "../HE/SalMaps/HEsalmap_50_2048x1024_32b.bin"][choice]

# Possible float precision of bin files
dtypes = {16: np.float16,
		  32: np.float32,
		  64: np.float64}

# Videos Saliency Maps
#    Extract info from file name
get_file_info = re.compile("(\d+_\w+)_(\d+)x(\d+)x(\d+)_(\d+)b")
info = get_file_info.findall(path2file.split(os.sep)[-1])
if len(info) > 0:
	name, width, height, Frames, dtype = info[0]
	width, height, Frames, dtype = int(width), int(height), int(Frames), int(dtype)
# Image Saliency Maps
else:
	get_file_info = re.compile("(\w+_\d{1,2})_(\d+)x(\d+)_(\d+)b")
	info = get_file_info.findall(path2file.split(os.sep)[-1])

	name, width, height, dtype = info[0]
	width, height, dtype = int(width), int(height), int(dtype)
	Frames = 1

# Open file to read as binary
with open(path2file, "rb") as f:
	# Position read pointer right before target frame
	f.seek(width*height * Nframe * (dtype//8))

	# Read from file the content of one frame
	data = np.fromfile(f, count=width*height, dtype=dtypes[dtype])
	# Reshape flattened data to 2D image
	data = data.reshape([height, width])
	# data = 255 * (data - np.min(data))/(np.max(data)-np.min(data)) #normal
	# cv2.imrite('tmp.jpg', data)
	# Display image as a heatmap
	plt.title("{}, frame #{}".format(name, Nframe))
	plt.imshow(data)
	plt.show()