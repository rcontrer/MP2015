from astropy.io import fits
from astropy import wcs
from astropy.utils.console import ProgressBar, color_print
from scipy.optimize import curve_fit
import numpy as np
import sys
import os

folder = './%s' % sys.argv[1]

###
### FUNCIONES
###

#Obtiene el menor seeing de un arreglo de fits
def lowest_seeing(files):
	n = files.size #Nro epocas
	s = np.zeros(n)

	for i,f in enumerate(files):
		hdu  = fits.open(folder+f)
		see  = hdu[0].header['SEEING']
		s[i] += see
	return s.min(),np.argmin(s)

#Verifica que la carpeta exista. Si no, la crea
def makedir(directory):
	if not os.path.exists(directory):
		os.makedirs(directory)

#Transforma de XY a RADEC
def XYtoRADEC(ep):
	ffn,cfn = ep

	id,x,y,mag = np.loadtxt(folder+cfn,usecols=range(4),skiprows=3,unpack=True)
	id = id.astype(int)
	
	hdr    = fits.open(folder+ffn)[0].header
	w      = wcs.WCS(hdr)
	ra,dec = np.transpose(w.wcs_pix2world(np.transpose([x,y]),1))
	
	head = 'ID RA DEC X Y MAG'
	fmt  = '%d %.7f %.7f %.3f %.3f %.3f'
	np.savetxt(folder+ffn.replace('fits','dat'),np.transpose([id,ra,dec,x,y,mag]),header=head,fmt=fmt)

#Transformacion cuadratica
def quadratic(coords,a,b,c,d,e,f):
	x,y = coords
	#a,b,c,d,e,f = params

	return a + b*x + c*y + d*np.square(x) + e*np.multiply(x,y) + f*np.square(y)


##
## VARIABLES
##

#Carpetas
match_path = 'matched_epochs/'
results    = folder.replace('/data','').split('_')[0] + '/'

makedir(results)
makedir(results+match_path)

#Parametros
master_iterations = 4
star_treshold	  = 50
match_tolerance   = 0.3

#Control
get_RADEC, match_CMD, match_epo = [False for i in range(3)]

if 'r' in sys.argv[2] or 'a' in sys.argv[2]: get_RADEC = True
if 'c' in sys.argv[2] or 'a' in sys.argv[2]: match_CMD = True
if 'e' in sys.argv[2] or 'a' in sys.argv[2]: match_epo = True

color_print('Iniciando "Master Frame" con argumentos %s'%sys.argv[2],'lightcyan')

###
### PIPELINE
###

#Lee todos los archivos
color_print('-Leyendo archivos...','cyan')
archivos  = np.sort([f for f in os.listdir(folder)])
fits_all  = np.sort([f for f in archivos if f.endswith('.fits')])
fits_k    = np.sort([f for f in fits_all if "k" in f])
fits_j    = np.sort([f for f in fits_all if "j" in f])
catalog   = np.sort([f for f in archivos if f.endswith('.dao') or f.endswith('.cxc')])
catalog_k = np.sort([f for f in catalog if "k" in f])
catalog_j = np.sort([f for f in catalog if "j" in f])

#Obtiene el seeing
color_print('-Obteniendo menor seeing...','cyan')
ks,ksi = lowest_seeing(fits_k)
js,jsi = lowest_seeing(fits_j)
cmd_fits = fits_j[jsi], fits_k[ksi]
cmd_cata = catalog_j[jsi], catalog_k[ksi]
print '\tMenor seeing en K: %s (%f)' % (cmd_fits[1],ks)
print '\tMenor seeing en J: %s (%f)' % (cmd_fits[0],js)

#Obtiene los RA y DEC con WCS
color_print('-Obteniendo RADEC...','cyan')
if get_RADEC: 
	ProgressBar.map(XYtoRADEC,np.transpose([fits_k,catalog_k]),multiprocess=True)
	ProgressBar.map(XYtoRADEC,np.transpose([fits_j,catalog_j]),multiprocess=True)
cmd_dat = cmd_fits[0].replace('fits','dat'), cmd_fits[1].replace('fits','dat')

#CMD (match y creacion)
color_print('-Match CMD...','cyan')
execute  = 'sh stilts tmatch2 ifmt1=ascii ifmt2=ascii matcher=sky ofmt=ascii values1="RA DEC" values2="RA DEC" '
if match_CMD:
	exec_CMD = 'in1=%s in2=%s out=%s params=%.1f progress=none join=all1' % (folder+cmd_dat[1],folder+cmd_dat[0],results+'CMD.dat',match_tolerance)

	os.system(execute+exec_CMD)

#Match de todas las epocas con la de referencia (menor seeing)
color_print('-Match epocas...','cyan')
execute = execute.replace('values1="RA DEC"','values1="RA_1 DEC_1"')
if match_epo:
	def ep_match(fn):
		exec_ep = 'in1=%s in2=%s out=%s params=%.1f progress=none' % (results+'CMD.dat',folder+fn.replace('fits','dat'),results+match_path+fn.replace('fits','match'),match_tolerance)
		os.system(execute+exec_ep)

	ProgressBar.map(ep_match,fits_k,multiprocess=True)

#Crea el Master Frame!
color_print('-Creando Master Frame...','lightcyan')

mid,mx,my,mk,mj = np.genfromtxt(results+'CMD.dat',usecols=(0,3,4,5,11),unpack=True) #Toma el CMD (epoca de referencia + J)
#mid,mx,my = mid.astype(np.int32),mx.astype(np.float32),my.astype(np.float32)

def masterframe(fn):
	fn = results+match_path+fn.replace('fits','match')
	
	eid,ex,ey = np.genfromtxt(fn,usecols=(0,16,17),unpack=True)
	eid = eid.astype(int)
	epinma	  = np.in1d(eid,mid)
	mainep	  = np.in1d(mid,eid)

	coords = np.transpose([ex,ey])[epinma].T

	poptX, pcovX = curve_fit(quadratic,coords,mx[mainep])
	poptY, pcovY = curve_fit(quadratic,coords,my[mainep])	

	ptx = quadratic(coords,*poptX)
	pty = quadratic(coords,*poptY)

	#Anade todo
	counts[k][mainep]  += 1
	added_x[k][mainep] += ptx
	added_y[k][mainep] += pty

	del epinma,eid,mainep,coords,ptx,pty

print '\tNumero de iteraciones: %d' % master_iterations
for i in range(master_iterations):
	counts    = np.zeros((fits_k.size,mid.size)) 	#Contar cuantos matches hay para luego dividir
	added_x   = np.zeros((fits_k.size,mid.size))	#Suma de las posiciones en x
	added_y   = np.zeros((fits_k.size,mid.size))	#Suma de las posiciones en y

	#ProgressBar.map(masterframe,fits_k,multiprocess=True)
	#for item in ProgressBar.iterate(fits_k):
	with ProgressBar(fits_k.size) as bar:
		for k in range(fits_k.size):
			masterframe(fits_k[k])
			bar.update()
	
	id_counts = np.sum(counts,axis=0)	#Nro de veces que se encontro el ID en la epoca de referencia
	ep_counts = np.sum(counts,axis=1)	#Nro de estrellas en el match de cada epoca

	mx = np.divide(np.sum(added_x,axis=0),id_counts)
	my = np.divide(np.sum(added_y,axis=0),id_counts)

	del counts,added_x,added_y

#Graficos
print '\tGuardando graficos...'
epocas = np.arange(fits_k.size)
ep_stars = np.zeros(fits_k.size)

import subprocess as sp
for i in range(fits_k.size):
	ep_stars[i] += (int(sp.check_output('wc -l %s'%(folder+fits_k[i].replace('fits','dao')), stderr=sp.STDOUT, shell=True).split()[0])-3)

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=[6*2,4.5*3],nrows=3)

ax[0].fill_between(epocas,0,ep_stars,color='black',alpha=.66,zorder=-2)
ax[0].fill_between(epocas,0,ep_counts,color='orange',alpha=.66,zorder=0)
ax[0].plot(epocas,ep_stars,'-ok',alpha=.75,zorder=-1)
ax[0].plot(epocas,ep_counts,'-o',color='orange',alpha=.75,zorder=1)
ax[0].set_xlim(epocas.min()-1,epocas.max()+1)
ax[0].set_xlabel(r'Epocas')
ax[0].set_ylabel(r'Nro Estrellas')

ax[1].scatter(mx,id_counts,s=5,c=mk,edgecolor='',alpha=.5)
ax[1].set_xlabel(r'x')
ax[1].set_ylabel(r'Nro Epocas')
ax[1].set_xlim(-1,mx.max()+1)

ax[2].scatter(my,id_counts,s=5,c=mk,edgecolor='',alpha=.5)
ax[2].set_xlabel(r'y')
ax[2].set_ylabel(r'Nro Epocas')
ax[2].set_xlim(-1,mx.max()+1)

fig.savefig(results+'Master.png',dpi=300,bbox_inches='tight')

#Guarda el Master Frame
print '\tGuardando Master Frame...'
tidx = id_counts >= star_treshold	#Para descartar las estrellas que se encontraron en menos de st epocas
masterframe = np.transpose([mid,mx,my,mk,mj])[tidx]

fmt = '%d %.3f %.3f %.3f %.3f'
hdr = '#ID X Y MAG_K MAG_J'
np.savetxt(results+'Master.dat',masterframe,fmt=fmt,header=hdr)


#import pandas as pd
#wuo = pd.read_table(results+'CMD.dat',sep=' ',skipinitialspace=True,usecols=[0,3,4],comment='#',header=None)
