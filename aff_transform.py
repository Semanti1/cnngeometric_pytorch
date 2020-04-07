import cv2
#import tkinter as tk
import numpy as np
#from mayavi import mlab
import plotly.graph_objects as go
import plotly.express as px
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use('tk')
img1 = cv2.imread("/home/semanti/Documents/cnngeometric_pytorch/source.png")
img2 = cv2.imread("/home/semanti/Documents/cnngeometric_pytorch/target.png")
#img1 = cv2.imread("/home/semanti/Downloads/dino/dino0024.png")
#img2 = cv2.imread("/home/semanti/Downloads/dino/dino0025.png")
img1 = np.dot(img1[:,:,:3],[0.299,0.587,0.144])
img2 = np.dot(img2[:,:,:3],[0.299,0.587,0.144])
x = img1.shape[0]
y = img1.shape[1]
x1 = img2.shape[0]
y1 = img1.shape[1]
#print(img1.shape, img1_g.shape)
#print(img1_g)
#print(x,y,x1,y1)
#[1.1149, -0.1842,  0.1999],[ 0.2404,  0.8349,  0.1677],[ 0, 0, 1]
#aff = np.array([[1.1149, -0.1842,  0.1999],[ 0.2404,  0.8349,  0.1677],[ 0, 0, 1]]) 
#aff = np.array([[1.2622, -0.0023, -0.0100],[0.0249,  0.9497,  0.0071],[ 0, 0, 1]])
#aff = np.array([[1.0185, -0.0026,  0.0186],[-0.0370,  1.2994,  0.0196],[ 0, 0, 1]])
#aff = np.array([[ 1.0293, -0.0442, -0.0107],[0.0387,  0.9974, -0.0047],[0,0,1]])
aff = np.array([[1.0240e+00,  1.2685e-03, -2.7685e-03],[6.2905e-02,  9.7840e-01,-6.1642e-04],[ 0, 0, 1]])
d = dict()
for i in range(x):
	for j in range(y):
		pt = np.array([i,j,1])
		intensity1=img1[i,j]
		#print(intensity1, "IIIIIIIIIIIINNNNNNNNNNNNNNNNNNN")
		#print("pt", pt, pt[:-1])
		#val = np.matmul(aff,pt.T)
		#d[pt[:-1]]= np.rint(val)[:-1].T
		#grayint = np.dot(intensity1,[0.299,0.587,0.144])
		#print(grayint, "gray")
		if (intensity1 >= 75):
			#print(grayint, "abv thresh")
			#print(np.matmul(aff,pt.T), "KKKKKKKKKKKKKKKKKKKKK")
			trPt = np.rint(np.matmul(aff,pt.T))
			#print(np.rint(trPt[0]),np.rint(trPt[1]))
			#print(trPt, "trpttt")
			#print(round(trPt[0]),round(trPt[1]), "rrrrrrrrrrrrroooooooooouu")
			if(trPt[0] in range(0,x1) and trPt[1] in range(y1)):
				intensity2 = img2[int(trPt[0]),int(trPt[1])]
				if(np.abs(intensity1-intensity2)==0):
					d[(i,j)] = np.matmul(aff,pt.T)
					d[(i,j)]= tuple(np.rint(d[(i,j)][:-1].T))
		#print("orig",(i,j),"3d",d[(i,j)].T,"2d",np.rint(d[(i,j)][:-1].T))

#print(d)
rel_coord= dict()
for key, val in d.items():
	x = val[0]
	y = val[1]
	rel_coord[key] = (x,y)
	'''if x<=240 and y<=240:
		#rel_coord[key] = (x-240,y)
		rel_coord[key] = (x,y)'''

[print(k[0],k[1],v[0],v[1]) for k,v in rel_coord.items()]
'''z_coord = []
x_coord = []
coord = []
dist = round(np.sqrt((0.1999)*(0.1999) + (0.1677)*(0.1677)),2)
[z_coord.append(dist/(key[0]-val[0])) for key,val in rel_coord.items()]
ind = 0
for k in rel_coord:
	X = k[0] * z_coord[ind]
	Y = k[1] * z_coord[ind]
	#X = k[0]
	#Y = k[1]
	coord.append((X,Y,z_coord[ind]))
	ind = ind +1
[print(z) for z in coord]
x = np.array([i[0] for i in coord])
y = np.array([i[1] for i in coord])
z = np.array([i[2] for i in coord])'''
'''pts = mlab.points3d(x,y,z,z)
mesh = mlab.pipeline.delaunay2d(pts)
pts.remove()

# Draw a surface based on the triangulation
surf = mlab.pipeline.surface(mesh)

# Simple plot.
mlab.xlabel("x")
mlab.ylabel("y")
mlab.zlabel("z")
mlab.show()'''
'''coord = []
A = np.array([[0,-1,0,0],[-1,0,0,0],[0,-1,0,0],[-1,0,0,0]])
for k,val in d.items():
	pt1 = np.array([k[1],k[0]])
	pt2 = np.flip(val)
	col = np.concatenate((pt1,pt2))
	D = A
	D[:,2] = col
	#print("D",D)
	P, diag, Q = np.linalg.svd(D, full_matrices = False)
	print("Q", Q)
	coord.append(Q[:,-1])
	print(Q[:,-1])'''
'''fig = go.Figure(data=[go.Mesh3d(x=x, y=y, z=z, color='lightpink', opacity=0.50)])
fig.show()'''
		
'''fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z,
                                   mode='markers')])
fig.show()'''
'''fig = plt.figure()
ax = plt.axes(projection="3d")
ax.scatter3D(x, y, z)
#plt.show()
plt.savefig('myfig')'''
