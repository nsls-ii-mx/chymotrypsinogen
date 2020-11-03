from __future__ import division
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg
import re
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import scipy.cluster.hierarchy as sch
import math
import sys

class Ellipsoid:
	def __init__(self): pass

	def getMinElp(self, array_inp=None, err=0.01):
		(N, dim) = np.shape(array_inp)
		array_inp_trans = array_inp.T
		dim = float(dim)
		array_stack = np.vstack([np.copy(array_inp_trans), np.ones(N)])
		array_stack_trans = array_stack.T
		error = 1.0 + err
		u = (1.0 / N) * np.ones(N)
		while error > err:
			v = np.dot(array_stack, np.dot(np.diag(u), array_stack_trans))
			diag_vector = np.diag(np.dot(array_stack_trans , np.dot(linalg.inv(v), array_stack)))
			max_ind = np.argmax(diag_vector)
			max_val = diag_vector[max_ind]
			step_size = (max_val - dim - 1.0) / ((dim + 1.0) * (max_val - 1.0))
			new_u = (1.0 - step_size) * u
			new_u[max_ind] += step_size
			error = np.linalg.norm(new_u - u)
			u = new_u
		center_elp = np.dot(array_inp_trans, u)
		center_mul = np.array([[a * b for b in center_elp] for a in center_elp])
		A = linalg.inv(np.dot(array_inp_trans, np.dot(np.diag(u), array_inp)) - center_mul) / dim
		U, s, rotation = linalg.svd(A)
		r = 1.0/np.sqrt(s)
		return (center_elp, r, rotation)

	def plot(self, center_elp, r, rotation, ax, plotAxes=False, color='b', alpha=0.2):
		u = np.linspace(0.0, 2.0 * np.pi, 100)
		v = np.linspace(0.0, np.pi, 100)
		x = r[0] * np.outer(np.cos(u), np.sin(v))
		y = r[1] * np.outer(np.sin(u), np.sin(v))
		z = r[2] * np.outer(np.ones_like(u), np.cos(v))
		for i in range(len(x)):
			for j in range(len(x)):
				[x[i,j],y[i,j],z[i,j]] = np.dot([x[i,j],y[i,j],z[i,j]], rotation) + center_elp
		if plotAxes:
			axes = np.array([[r[0],0.0,0.0],
                             [0.0,r[1],0.0],
                             [0.0,0.0,r[2]]])
			for i in range(len(axes)):
				axes[i] = np.dot(axes[i], rotation)
			for p in axes:
				x_tmp = np.linspace(-p[0], p[0], 100) + center_elp[0]
				y_tmp = np.linspace(-p[1], p[1], 100) + center_elp[1]
				z_tmp = np.linspace(-p[2], p[2], 100) + center_elp[2]
				ax.plot(x_tmp, y_tmp, z_tmp, color=color)
		ax.plot_wireframe(x, y, z,  rstride=4, cstride=4, color=color, alpha=alpha)

def extract_coordinate(filename):
	with open(filename) as file:
		coor_list = []
		lines = file.readlines()
		for l in lines:
			l = re.split('\s+', l)
			coor_list.append(l[6:9])
	return coor_list

def extract_cluster_files(cluster_out_file):
	with open(cluster_out_file) as file:
		cluster_dict = {}
		lines = file.readlines()
		for l in lines[1:len(lines)-4]:
			l = re.split('\s+', l.strip())
			cluster_nb = int(re.findall('[1-9][0-9]*', l[0])[0])
			cluster_files = l[3:]
			cluster_dict.update({cluster_nb: cluster_files})
	return cluster_dict

def extract_matrix(matrix_file):
	with open(matrix_file) as file:
		matrix = []
		lines = file.readlines()
		for i in range(len(lines)):
			l = re.split('\s+', lines[i].strip())
			if(i%3==0):
				matrix.append(float(l[1]))
			elif (i%3==1):
				matrix.append(float(l[2]))
			else:
				matrix.append(float(l[3]))
		return matrix

def get_cc_matrix(cc_file):
	cc_matrix=[]
	with open(cc_file) as f:
		lines = f.readlines()
	for line in lines[1:]:
		line = re.split('\s+',line.strip())
		cc = float(line[2])
		distance = math.sqrt(1-cc)
		cc_matrix.append(distance)
	return cc_matrix
	
if __name__ == '__main__':
	#argv[1]: cctable.dat from Kamo outputs
	#argv[2]: file contains coordinates of a specific residue of all structures
	#argv[3]: CLUSTERS.txt from Kamo outputs
	#argv[4]: height cutoff
	
	matrix = get_cc_matrix(sys.argv[1])
	z = sch.linkage(matrix, method='ward')

	rs = extract_coordinate(sys.argv[2])

	fig_dendro = plt.figure(figsize=(80, 50))
	plt.rc('ytick', labelsize=20)
	plt.ylabel('Height', fontsize=20)
	sch.set_link_color_palette(['g', 'r', 'c', 'm', 'y'])
	d = sch.dendrogram(z, color_threshold=float(sys.argv[4]))
	color_cluster=d['color_list']
	ivl = d['ivl']
	print(ivl)
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	for i in range(len(rs)):
		rs[i][0] = float(rs[i][0])
		rs[i][1] = float(rs[i][1])
		rs[i][2] = float(rs[i][2])
	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Z')
	kmeans = KMeans(n_clusters = 5)
	kmeans.fit(rs)
	range_0, range_1, range_2, range_3, range_4 = 1, 1, 1, 1, 1
	for i in range (0, len(color_cluster)):
		if(color_cluster[i]== 'g'):
			range_0 += 1
		elif(color_cluster[i]=='r'):
			range_1 += 1
		elif(color_cluster[i]=='c'):
			range_2 += 1
		elif(color_cluster[i]=='m'):
			range_3 += 1
		elif(color_cluster[i]=='y'):
			range_4 += 1
		else:
			continue
	s0=[int(ivl[i])+1 for i in range(0,range_0)]
	s1=[int(ivl[i])+1 for i in range(range_0,range_0+range_1)]
	s2=[int(ivl[i])+1 for i in range(range_0+range_1,range_0+range_1+range_2)]
	s3=[int(ivl[i])+1 for i in range(range_0+range_1+range_2,range_0+range_1+range_2+range_3)]
	s4=[int(ivl[i])+1 for i in range(range_0+range_1+range_2+range_3,range_0+range_1+range_2+range_3+range_4)]
	s0_cluster=[]
	s1_cluster=[]
	s2_cluster=[]
	s3_cluster=[]
	s4_cluster=[]
	cluster_dict = extract_cluster_files(sys.argv[3])
	clusters = cluster_dict.keys()
	for c0 in s0:
		for cluster in clusters:
			if str(c0) in cluster_dict[cluster] and (cluster not in s0_cluster):
				s0_cluster.append(cluster)
	for c1 in s1:
		for cluster in clusters:
			if str(c1) in cluster_dict[cluster] and (cluster not in s1_cluster):
				s1_cluster.append(cluster)
	for c2 in s2:
		for cluster in clusters:
			if str(c2) in cluster_dict[cluster] and (cluster not in s2_cluster):
				s2_cluster.append(cluster)
	for c3 in s3:
		for cluster in clusters:
			if str(c3) in cluster_dict[cluster] and (cluster not in s3_cluster):
				s3_cluster.append(cluster)
	for c4 in s4:
		for cluster in clusters:
			if str(c4) in cluster_dict[cluster] and (cluster not in s4_cluster):
				s4_cluster.append(cluster)
	s0_centroid=[]
	s1_centroid=[]
	s2_centroid=[]
	s3_centroid=[]
	s4_centroid=[]
	for s0 in s0_cluster:
		s0_centroid.append([rs[s0-1][0], rs[s0-1][1], rs[s0-1][2]])
	for s1 in s1_cluster:
		s1_centroid.append([rs[s1-1][0], rs[s1-1][1], rs[s1-1][2]])
	for s2 in s2_cluster:
		s2_centroid.append([rs[s2-1][0], rs[s2-1][1], rs[s2-1][2]])
	for s3 in s3_cluster:
		s3_centroid.append([rs[s3-1][0], rs[s3-1][1], rs[s3-1][2]])
	for s4 in s4_cluster:
		s4_centroid.append([rs[s4-1][0], rs[s4-1][1], rs[s4-1][2]])
	s0_centroid = np.asarray(s0_centroid)
	s1_centroid = np.asarray(s1_centroid)
	s2_centroid = np.asarray(s2_centroid)
	s3_centroid = np.asarray(s3_centroid)
	s4_centroid = np.asarray(s4_centroid)
	ellipsoid = Ellipsoid()
	(center_0, radii_0, rotation_0) = ellipsoid.getMinElp(s0_centroid)
	(center_1, radii_1, rotation_1) = ellipsoid.getMinElp(s1_centroid)
	(center_2, radii_2, rotation_2) = ellipsoid.getMinElp(s2_centroid)
	(center_3, radii_3, rotation_3) = ellipsoid.getMinElp(s3_centroid)
	(center_4, radii_4, rotation_4) = ellipsoid.getMinElp(s4_centroid)
	ax.scatter(s0_centroid[:,0], s0_centroid[:,1], s0_centroid[:,2], color='g', s=10)
	ax.scatter(s1_centroid[:,0], s1_centroid[:,1], s1_centroid[:,2], color='r', s=10)
	ax.scatter(s2_centroid[:,0], s2_centroid[:,1], s2_centroid[:,2], color='c', s=10)
	ax.scatter(s3_centroid[:,0], s3_centroid[:,1], s3_centroid[:,2], color='m', s=10)
	ax.scatter(s4_centroid[:,0], s4_centroid[:,1], s4_centroid[:,2], color='y', s=10)
	ellipsoid.plot(center_0, radii_0, rotation_0, ax, plotAxes=True, color='g')
	ellipsoid.plot(center_1, radii_1, rotation_1, ax, plotAxes=True, color='r')
	ellipsoid.plot(center_2, radii_2, rotation_2, ax, plotAxes=True, color='c')
	ellipsoid.plot(center_3, radii_3, rotation_3, ax, plotAxes=True, color='m')
	ellipsoid.plot(center_4, radii_4, rotation_4, ax, plotAxes=True, color='y')
	plt.title("C-alpha coordinate variances at residue 146")
	plt.show()
#plt.savefig(sys.argv[4])
	#plt.close(fig)

