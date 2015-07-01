#!/usr/bin/env python
# -*- coding: utf-8 -*-
#FR

import os
import sys 
import numpy as np
import matplotlib.pyplot as plt 
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree
from astropy.utils.console import ProgressBar,color_print
from astropy.io import fits

folder = './%s' % sys.argv[1]

##
##	FUNCIONES
##

#Crea carpeta si es que no existe
def makedir(directory):
	if not os.path.exists(directory):
		os.makedirs(directory)

def linear(coords,a,b,c):
	x,y = coords
	return a + b*x + c*y

def linearPM(dates,PM,zero_point):
	return dates*PM + zero_point

##
##	PARAMETROS
##

local_transformation = True
neighbor_search      = 50
vc_pix 		     = 0.34

star_motion_calc = True

#Carpetas
results    = folder.replace('/data','').split('_')[0] + '/'
sm_path    = 'stars_motion/'
match_path = 'matched_epochs/'
zero 	   = results + sys.argv[2]

makedir(results+sm_path)

##
##	PIPELINE
##

#Obtiene las epocas
color_print('Buscando archivos de epocas...','cyan')
k_catalog = np.sort([f for f in os.listdir(folder) if 'k' in f and (f.endswith('.dao') or f.endswith('.cxc'))])
k_fits    = np.sort([f for f in os.listdir(folder) if 'k' in f and f.endswith('.fits')])

#Obtiene los anios correspondientes
color_print('Leyendo BJD de cada epoca...','cyan')
dates = np.zeros(k_fits.size)
with ProgressBar(k_fits.size) as bar:
	for i in range(k_fits.size):
		hdu  = fits.open(folder+k_fits[i])
		date = hdu[0].header['MJD-OBS']

		dates[i] += date
		bar.update()

dates = (dates-dates[0])/365.25
print '\tIntervalo de tiempo: %.2f años' % dates.max()

#Carga catalogo
mid,mx,my,mk,mj = np.genfromtxt(results+'Master.dat',unpack=True)
ssid = []
#Funcion para los PM
def proper_motion(j):
	fn = results+match_path+k_fits[j].replace('.fits','.match')
	
	eid,ex,ey = np.genfromtxt(fn,usecols=(0,16,17),unpack=True)
	eid       = eid.astype(int)
	einma	  = np.in1d(eid,mid)
	mainep    = np.in1d(mid,eid)

	sdata = np.transpose([mid,mx,my,mk,mj])[mainep].T
	sid,sx,sy,sk,sj = sdata
	ssid = sid
	ex,ey = np.transpose([ex,ey])[einma].T
	#Tengo las mismas estrellas para la MF y la epoca
	#Ahora a aplifcar transformacion local

	#Lee las estrellas de referencia (zero)
	#Y ve cuales estan en las estrellas ya filtradas
	rid = np.genfromtxt(zero,usecols=(0,),unpack=True)
	idx = np.in1d(sid,rid)
	
	#Crea el cKDTree con las coordenadas de 'zero'
	coords = np.empty((idx.sum(),2))
	coords[:,0] = sx[idx]
	coords[:,1] = sy[idx]
	tree = cKDTree(coords)

	#Busca los vecinos mas cercanos en 'zero'
	scoords = np.empty((sid.size,2))
	scoords[:,0] = sx
	scoords[:,1] = sy
	
	kdd,kdi = tree.query(scoords,k=neighbor_search)

	#Busca la transformacion local usando los vecinos encontrados
	exs,eys = np.transpose([ex,ey])[idx].T
	#Si hay menos estrellas que los vecinos cercanos, next
	if exs.size<neighbor_search:
		return
	ctrans = np.zeros((sid.size,2))
	for i in range(sid.size):
		nix = kdi[i]
		#Transformaciones de la estrella i
		input1 = np.transpose([exs,eys])[nix].T

		poptx, pcovx = curve_fit(linear,input1,coords.T[0][nix])
		popty, pcovy = curve_fit(linear,input1,coords.T[1][nix])
		ctrans[i,0] = linear(np.transpose([ex,ey])[i],*poptx)
		ctrans[i,1] = linear(np.transpose([ex,ey])[i],*popty)
	
	#Calcula el shift en esta epoca
	motion_x = ctrans.T[0] - sx
	motion_y = ctrans.T[1] - sy
	motion   = np.sqrt(motion_x**2+motion_y**2)

	#Guarda el shift de la epoca
	#fmt = '%d %.5f %.5f'
	#np.savetxt(results+sm_path+fn.replace('.match','.sm').split('/')[-1],np.transpose([sid,motion_x,motion_y]),fmt=fmt)
	if j==0:
		for i in range(k_fits.size):
			star = open(results+sm_path+'%d.sm'%sid[i],'w')
			star.write('#YEAR MOTION_X MOTION_Y\n')
			star.close()	

	for i in range(sid.size):
		star = open(results+sm_path+'%d.sm'%sid[i], 'a')
		star.write('%.3f %.5f %.5f\n'%(dates[j],motion_x[i],motion_y[i]))
		star.close()

color_print('Calculando diferencias por epoca...','cyan')
#ProgressBar.map(proper_motion,k_fits,multiprocess=True)
#print '\tInicializando archivos auxiliares...'
if star_motion_calc:
	with ProgressBar(k_fits.size) as bar:
		for i in range(k_fits.size):
		#for i in range(61,63):
			proper_motion(i)
			bar.update()

color_print('Recopilando informacion de cada estrella','cyan')
sm_files = np.sort([f for f in os.listdir(results+sm_path)])

ProperMotion = [[],[],[],[]]
star_mask    = []

with ProgressBar(sm_files.size) as bar:
	for i in range(sm_files.size):
		yr,motx,moty = np.genfromtxt(results+sm_path+sm_files[i],unpack=True)

		if yr.size <=10:
			continue

		poptx,pcovx = curve_fit(linearPM,yr,motx)
		popty,pcovy = curve_fit(linearPM,yr,moty)

		if type(pcovx)!=np.ndarray or type(pcovy)!= np.ndarray:
			continue

		if np.sqrt(pcovx[0,0])>=1 or np.sqrt(pcovy[0,0])>=1:
			continue

		ProperMotion[0].append(poptx[0]*vc_pix*1000)
		ProperMotion[1].append(popty[0]*vc_pix*1000)
		ProperMotion[2].append(np.sqrt(pcovx[0,0])*vc_pix*1000)
		ProperMotion[3].append(np.sqrt(pcovy[0,0])*vc_pix*1000)

		star_mask.append(i)
		bar.update()

color_print('Calculando PM...','lightcyan')
for i in range(len(ProperMotion)):
	ProperMotion[i] = np.array(ProperMotion[i])

PM = np.sqrt(ProperMotion[0]**2+ProperMotion[1]**2)

star_mask = np.array(star_mask).astype(int)
print star_mask
print np.shape(star_mask),mid.shape,mx.shape,mk.shape

sid = mid[star_mask].astype(int)
col = (mj-mk)[star_mask]
mag = mk[star_mask]
px  = mx[star_mask]
py  = my[star_mask]

fmt = '%d %.3f %.3f %.3f %.3f %.5f %.5f %.5f %.5f'
hdr = 'ID X Y COLOR_JK MAG_K PM_X PM_Y PM_X_ERR PM_Y_ERR'
np.savetxt(results+'PM_%s.dat'%sys.argv[2].split('.')[0],np.transpose([sid,px,py,col,mag,ProperMotion[0],ProperMotion[1],ProperMotion[2],ProperMotion[3]]),fmt=fmt,header=hdr)
	
		
