import numpy as np
import os
import sys
import matplotlib.pyplot as plt 
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors as NN 
from sklearn.neighbors import RadiusNeighborsClassifier as RN 
from astropy.utils.console import ProgressBar

plt.style.use('ggplot')
vecinos = 16	#1 + vecinos
radio   = False
output  = 'local_15all14155.pdf'

se,el,yr = np.genfromtxt('../zinfo_img',unpack=True,usecols=(4,5,6),skiprows=6)
yrs = (yr-yr[0])/365.25

def linear(coords,a,b,c):
	x,y = coords
	return a*x + b*y + c

def quadratic(coords,a,b,c,d,e,f):
	x,y = coords
	return a + b*x + c*y + d*np.square(x) + e*np.multiply(x,y) + f*np.square(y)

def linear2(coords,a,b,c,d):
	M = np.matrix([[a,b],[c,d]])
	return np.dot(M,coords)



bid,bx,by = np.genfromtxt('all14155.dat',unpack=True,usecols=(0,3,4))

fig, ax = plt.subplots(nrows=23,ncols=3,figsize=[3.5*3,3.5*23])
fig.tight_layout()

for i,a in enumerate(np.ravel(ax)):
	#r1,d1,m,r2,d2 = np.genfromtxt('%s.dat'%(70*j/8+(i+1)),unpack=True,usecols=(1,2,5,7,8))
	id,x1,y1,m,x2,y2 = np.genfromtxt('%s.dat'%(i+1),unpack=True,usecols=(0,3,4,5,9,10))
	mask			 = (m<15)*(m>11)
	id,x1,y1,m,x2,y2   = np.transpose([id,x1,y1,m,x2,y2])[mask].T

	epinbu = np.in1d(id,bid)
	buinep = np.in1d(bid,id)

	print (id[epinbu]==bid[buinep]).sum(),id.size

	epxy = np.transpose([x2,y2])[epinbu]
	epm  = m[epinbu]

	nbrs = NN(n_neighbors=vecinos, algorithm='auto').fit(epxy)

	if radio:
		dist, idx = nbrs.radius_neighbors(np.transpose([x2,y2]),radius=400)
		nbors 	  = np.array([len(d) for d in dist])

		mednbors, minnbors = np.median(nbors),nbors.min()

		for j in range(len(dist)):
			#Elimina la misma estrella
			msk	    = dist[j]>300
			dist[j] = dist[j][msk]
			idx[j]  = idx[j][msk]


			#Toma las 50 mas brillantes
			'''
			if len(dist[j])>15:
				midx = np.argsort(epm[idx[j]])[:15]
				dist[j] = dist[j][midx]
				idx[j]  = idx[j][midx]
			'''

			if len(dist[j])>15:
				midx = np.argsort(dist[j])[:15]
				dist[j] = dist[j][midx]
				idx[j]  = idx[j][midx]
			
	else:	
		dist,idx = nbrs.kneighbors(np.transpose([x2,y2]))
		idx		 = idx[:,1:]
		dist 	 = dist[:,1:]

	means = np.array([np.mean(d) for d in dist])
	print means.size, means.shape
	nbors 	  = np.array([len(d) for d in dist])
	mednbors, minnbors = np.median(nbors),nbors.min()



	ctx = np.zeros(x1.size)
	cty = np.zeros(y1.size)
	
	with ProgressBar(x1.size) as bar:
		for k in range(x1.size):
			coords = np.transpose([x2,y2])[idx[k]].T
			poptx, pcovx = curve_fit(linear,coords,x1[idx[k]])
			popty, pcovy = curve_fit(linear,coords,y1[idx[k]])
			
			the_x = x2[k]
			the_y = y2[k]

			ctx[k] += linear([the_x,the_y],*poptx)
			cty[k] += linear([the_x,the_y],*popty)

			bar.update()

	x0,y0 = 1352.,554.
	#x0,y0 = 1788.,-6359.
	clust = (np.sqrt((x1-x0)**2 + (y1-y0)**2) < 280.)
	#clust = np.sqrt((x1-x0)**2 + (y1-y0)**2) < 350.
	#clust = ~epinbu
	#a.plot((r1-r2)[mask],(d1-d2)[mask],'o',ms=.5,alpha=.5)
	a.scatter((x1-ctx)[~clust],(y1-cty)[~clust],s=1,rasterized=True,edgecolor='',color='#0055FF',lw=.5)
	a.scatter((x1-ctx)[clust*(m<14)],(y1-cty)[clust*(m<14)],s=1.25,rasterized=True,edgecolor='',color='#FF5500',lw=.5)
	a.text(.1,.1,u'$S = %f$\n$E = %s$\n$N = %d/%d$\n$Med_d, Max_d = %.3f, %.3f$\n$Med_n,Min_n = %d,%d$'%(se[i+1],el[i+1],(clust*(m<14)).sum(),clust.size,np.median(means),np.max(means),mednbors, minnbors),transform = a.transAxes,alpha=.66,fontsize=10)
	a.text(.1,.9,u'$%d$'%(i+2),transform = a.transAxes,alpha=.66,fontsize=14)
	#a.set_xlim(-6e-5,6e-5)
	#a.set_ylim(-6e-5,6e-5)
	a.set_xlim(-10,10)
	a.set_ylim(-10,10)
	a.set_aspect('equal')

	dx = x1-ctx
	dy = x1-cty

	#np.savetxt()

	#a.set_xticklabels([])
	#a.set_yticklabels([])

		#plt.show()
	if (i%5)==0:
		print i
		#fig.suptitle('Neigh = %d'%vecinos)
		fig.savefig(output,dpi=200)
fig.savefig(output,dpi=200)