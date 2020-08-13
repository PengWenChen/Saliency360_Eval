#! /usr/bin/env python3
# ---------------------------------
# Author: Erwan DAVID
# Year: 2018
# Lab: IPI, LS2N, Nantes, France
# Comment: Scanpath maps/videos comparison tools and example as main
# Cite: E. DAVID, J. Guttiérez, A Coutrot, M. Perreira Da Silva, P. Le Callet (2018). A Dataset of Head and Eye Movements for 360° Videos. ACM MMSys18, dataset and toolbox track
# ---------------------------------

import numpy as np
import time

def getStartPositions(fixationList):
	"""Return positions of first fixation in list of scanpaths.
	Get starting indices of individual fixation sequences.
	"""
	return np.where(fixationList[:, 0] == 0)[0]

def getScanpath(fixationList, startPositions, scanpathIdx=0):
	"""Return a scanpath in a list of scanpaths
	"""
	if scanpathIdx >= startPositions.shape[0]-1:
		range_ = np.arange(startPositions[scanpathIdx], fixationList.shape[0])
	else:
		range_ = np.arange(startPositions[scanpathIdx], startPositions[scanpathIdx+1])
	# print(range_, startPositions[scanpathIdx])
	return fixationList[range_, :].copy()

def dist_angle(vec1, vec2):
	"""Angle between two vectors - same result as orthodromic distance
	"""
	return np.arccos( np.dot(vec1, vec2) / (np.linalg.norm(vec1)*np.linalg.norm(vec2)) )

def dist_starttime(t1, t2):
	"""Different between fixation starting timestamp fitted with exponential
	"""
	return 1- np.exp(-.15 * np.abs(t1-t2))

def getValues(VEC1, VEC2, i1, i2):
	"""Measure distance and angle between all fixations that happened during a frame.
	Return scores normalized (0, 1). Lower is better.
	"""
	values = []

	UVec1 = VEC1[i1, :3]
	UVec2 = VEC2[i2, :3]

	# Distance between starting/fixations points
	dist = dist_angle(UVec1, UVec2)

	angle = .5
	if  i1 > 0 and i2 > 0:
		Uvec1 = VEC1[i1-1, :3] - VEC1[i1, :3]
		Uvec2 = VEC2[i2-1, :3] - VEC2[i2, :3]

		# Angle between saccade vectors
		angle = dist_angle(Uvec1, Uvec2)

	# Amounts to penalize fixations that happened more than 50ms (replace -.03 with -.15) appart
	# Rai, Y., & LeCallet P. & Guiterriez J. (2017). A dataset of head and eye movements for 360 degree images. MMSys, Taiwan
	# Modified with value -.03 so that the exp reaches asymptote at approx. 200ms or difference
	# timeDiff = 1-np.exp(-.03 * np.abs( VEC1[i1, 3]-VEC2[i2, 3] ))
	# timeDiff = np.min( [1, np.abs(VEC1[i1, 3]-VEC2[i2, 3])/200])
	# if timeDiff < .8:
	# 	print(np.abs( VEC1[i1, 3]-VEC2[i2, 3] ), timeDiff)

	# return np.array([dist, angle, timeDiff]) / [np.pi, np.pi, 1]

	return np.array([dist, angle]) / [np.pi, np.pi]

def computeWeightMatrix(VEC1, VEC2, weight):
	"""Return weight matrix for alignment in Jarodzka's algorithm (MultiMatch).
	"""
	WMat = np.zeros([VEC1.shape[0], VEC2.shape[0]])
	Vals = np.zeros([VEC1.shape[0], VEC2.shape[0], weight.shape[0]])

	for i1 in range(VEC1.shape[0]):
		for i2 in range(VEC2.shape[0]):
			Vals[i1, i2, :] = getValues(VEC1, VEC2, i1, i2)
			WMat[i1, i2] = (Vals[i1, i2, :] * weight).sum()/weight.sum()

	return WMat, Vals

def alignScanpaths(WMat):
	"""Compute shortest path in weight matrix from first elements to last elements of both scanpaths as part of the alignment procedure in Jarodzka's algorithm (MultiMatch).
	Dijkstra's shortest path algo
	"""
	I = WMat.shape[0]
	J = WMat.shape[1]

	dist = np.ones([I*J]) * np.inf 
	prev = np.ones([I*J], dtype=int) - 1

	dist[0] = 0
	considered = []

	while len(considered) < I*J:
		d = dist.copy()

		d[considered] = np.inf
		u = np.argmin(d)
		considered.append(u)
		
		i = u // J
		j = u %  J

		trans = []
		if (i+1)*(j+1) == I*J:
			pass
		elif j == J-1:
			trans.append([WMat[i+1, j], [i+1, j]])
		elif i == I-1:
			trans.append([WMat[i, j+1], [i, j+1]])
		else:
			trans.append([WMat[i,   j+1], [i,   j+1]])
			trans.append([WMat[i+1, j  ], [i+1, j]])
			trans.append([WMat[i+1, j+1], [i+1, j+1]])

		for w in trans:
			alt = dist[u] + w[0]
			v = w[1][0]*J + w[1][1]
			if alt < dist[v]:
				dist[v] = alt
				prev[v] = u

	i = I-1
	j = J-1
	path = [[i, j]]
	while i+j != 0:
		idx = i*J + j

		u = prev[idx]

		i = u // J
		j = u %  J

		path.append([i, j])
	path.reverse()

	path = np.array(path)

	# print("\nTime: {:.2f} sec".format(time.time()-t1))

	return path

def sphre2UnitVector(sequence):
	"""Convert from longitude/latitude to 3D unit vectors
	"""
	UVec = np.zeros([sequence.shape[0], 3])
	UVec[:, 0] = np.cos(sequence[:,1]) * np.cos(sequence[:,0])
	UVec[:, 1] = np.cos(sequence[:,1]) * np.sin(sequence[:,0])
	UVec[:, 2] = np.sin(sequence[:,1])
	return UVec

def compareScanpath(fixations1, fixations2, starts1, starts2, iSC1, iSC2, weight=[1, 1]):
	"""Return comparison scores between two scanpaths.
	Option to grapically display weight matrix, scanpath aligment, scanpaths vectors and final measures with matplotlib.
	"""
	# print("Comparing scanpath #{} and #{}".format(idx1, idx2))
	weight = np.array(weight)
	# print("Comparing scanpath #{} and #{}".format(idx1, idx2))

	# Get individual experiment trials
	scanpath1 = getScanpath(fixations1, starts1, iSC1)
	scanpath2 = getScanpath(fixations2, starts2, iSC2)
	
	# 0,1,2: 3D unit vector; 3: starting timestamp
	VEC1 = np.zeros([scanpath1.shape[0], 4])
	VEC2 = np.zeros([scanpath2.shape[0], 4])

	# Store starting timestamp
	VEC1[:, 3] = scanpath1[:, 3]
	VEC2[:, 3] = scanpath2[:, 3]

	# Convert latitudes/longitudes to unit vectors
	VEC1[:, :3] = sphre2UnitVector(scanpath1[:, 1:3])
	VEC2[:, :3] = sphre2UnitVector(scanpath2[:, 1:3])

	# Get weight matrix and individual score values per cell in the weight matrix
	WMat, Vals = computeWeightMatrix(VEC1, VEC2, weight)
	# Find the shortest path through the weight matrix
	path = alignScanpaths(WMat)

	# Individual and final score: lower is better
	scores = np.mean(Vals[path[:, 0], path[:, 1], :], axis=0)
	score = (scores * weight).sum()/weight.sum()

	# Set to True to display information about the previous comparison process
	if False:
	# if True:
		# Display:
		#	Weight matix with shortest path in red
		import matplotlib.pyplot as plt
		# plt.rcParams["figure.figsize"] = [19.2, 10.8]
		plt.subplot(221)
		plt.plot(path[:, 1], path[:, 0], color='r')
		plt.imshow(WMat, cmap='gray')
		plt.xlabel("Scanpath {} (N={})".format(iSC2, scanpath2.shape[0]))
		plt.ylabel("Scanpath {} (N={})".format(iSC1, scanpath1.shape[0]))

		# Position of both scanpaths in equirectangular
		plt.subplot(222)
		plt.plot(scanpath1[:, 1], scanpath1[:, 2], c='r') # Plot saccade lines
		plt.scatter(scanpath1[:, 1], scanpath1[:, 2], # Plot timestamp at fixation positions
				 marker='o', c='r', alpha=.5, s=scanpath1[:, 3]/100)
		plt.plot(scanpath2[:, 1], scanpath2[:, 2], c='b') # Plot saccade lines
		plt.scatter(scanpath2[:, 1], scanpath2[:, 2], # Plot timestamp at fixation positions
				 marker='o', c='b', alpha=.5, s=scanpath2[:, 3]/100)
		# Aesthetics
		plt.xlim([0, 2*np.pi])
		plt.ylim([0, np.pi])
		plt.xlabel("Latitude")
		plt.ylabel("Longitude")
		plt.gca().invert_yaxis()

		# Display total average score and individual scores
		ax = plt.subplot(212)
		text_style = dict(horizontalalignment='right', verticalalignment='center',
						  fontsize=20)#, fontdict={'family': 'monospace'})
		ax.set_axis_off()

		ax.annotate("Total score: {:.2f} (lower is better)".format(score),
			xy=(.5, .5), xycoords="axes fraction",
			xytext=(100, 20), textcoords="offset points",
			**text_style)
		for idx, val in enumerate(["Position", "Direction"]):#, "Starting time"]):
			# ax.text(3.5, 8 - idx, '{:}: {:<5.2}'.format(val, scores[:, idx].mean()), **text_style)
			# ax.annotate('{:}: {:<5.2}'.format(val, scores[:, idx].mean()),
			ax.annotate('{:}: {:<5.4} (x{:.2f})'.format(val, scores[idx], weight[idx]),
				xy=(.5, .5), xycoords="axes fraction",
				xytext=(100, -20*idx), textcoords="offset points",
				**text_style)

		plt.xlim([0, 10])
		plt.ylim([0, 10])

		plt.tight_layout()
		plt.show(); exit()

	return score
	# return scores, score

if __name__ == "__main__":
	print("Broke main for modelComparison_scanpath script"); exit()
	import os
	# np.random.seed(1)

	# Head-and-Eye data
	SP_PATH = "../HE/Scanpaths/"
	# Because Head-only data are trajectories sampled the same ways, timestamp difference will be very low and will have a great influence on the alignment. Almost guaranteeing a straight path through the weight matrix.
	SP_PATH = "../H/Scanpaths/"
	
	# Each scanpath file contains all observers fixation sequences in a row.
	#	Fixations are reported sequentially with an index entry, index 0 begins a sequence

	import glob
	scanpath_path = glob.glob(SP_PATH+"*.txt")[np.random.randint(170)]

	name = "_".join(scanpath_path.split(os.sep)[-1].split("_")[:2])

	# Load fixation lists
	# 		0_Idx, 1_longitude, 2_latitude, 3_start timestamp, 4_duration
	fixations = np.loadtxt(scanpath_path, delimiter=",", skiprows=1, usecols=(0, 1,2, 3))
	fixations = fixations * [1, 2*np.pi, np.pi, 1]

	# Get start/end indices of trials
	starts = getStartPositions(fixations)

	# Score value weighting
	#	Give more importance to temporal difference compared to spatial ones
	weight = [.5, .5, 1]
	
	print("Let's compare together all {} scanpaths in this file.".format(len(starts)))

	with open("example_ScanpathComparisons.csv", "w") as saveFile:
		saveFile.write("stimName, iSC1_iSC2, distance, angle, timeDelta, score\n")
		for iSC1 in range(1, len(starts)):
			for iSC2 in range(1, len(starts)):
				if iSC1 != iSC2:
					print(" "*20, "\r{}/{}".format((iSC1-1) * len(starts) + (iSC2-1), len(starts)**2), end="")
					scores, score = compareScanpath(fixations, iSC1, iSC2, weight)
					saveFile.write("{}, {}, {}, {}, {}, {}\n".format(
						name, "{}_{}".format(iSC1, iSC2), *scores, score
						)
					)
	print("\nDone")