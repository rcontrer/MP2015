import os
import numpy as np
from astropy.io import ascii
from scipy.optimize import curve_fit
from astropy.io import fits as pyfits
from astropy import wcs



############################################################################################################
#
#		VIRCAM TRANSFORMATION TO RA/DEC
#
############################################################################################################

#I added this class because it was not included initially and is needed for the transformation
class Cts:
	def __init__(self):
		self.deg2rad=0.017453
		self.rad2deg=57.295779

def vircam_xy2standard(x,y,hdr,output='equatorial'):
        """vircam radial distortion, convert from x,y to xi,xn or alpha,delta depending on the output
	desired, return the distortcor to fix the photometry, this function uses the header values
	determined by CASU.
	requires x,y, and the header of the image chip, and the output, default is equatorial
	which return alpha,delta and the distortion correction, 
	the other option is standard which return xi,xn and the distortion correction"""
	cts= Cts();	

	a = hdr['CD1_1']
	b = hdr['CD1_2']
	c = hdr['CRPIX1']
	d = hdr['CD2_1']
	e = hdr['CD2_2']
	f = hdr['CRPIX2']
	tpa=hdr['CRVAL1'] 
	tpd=hdr['CRVAL2']
	projp1 = hdr['PV2_1']
	projp3 = hdr['PV2_3']
	projp5 = hdr['PV2_5']
	a = a*cts.deg2rad
	b = b*cts.deg2rad
	d = d*cts.deg2rad
	e = e*cts.deg2rad
	tpa = tpa*cts.deg2rad
	tpd = tpd*cts.deg2rad
	tand= np.tan(tpd)
	secd= 1./np.cos(tpd)
	
	x   = x-c
	y   = y-f
	xi  = a*x+b*y
	xn  = d*x+e*y
	r   =  np.sqrt(xi**2.+xn**2.)
	rfac = projp1+projp3*r**2.+projp5*r**4. 
	r    = r/rfac
	rfac = projp1+projp3*r**2.+projp5*r**4.
	xi = xi/rfac 
	xn = xn/rfac
	distortcor = 1.0 + 3.0*projp3*r**2./projp1 + 5.0*projp5*r**4./projp1
	distortcor = distortcor*(1.0+(projp3*r**2.+projp5*r**4.)/projp1)
	distortcor = 1.0/distortcor
	
	if output=='equatorial':
		aa = np.arctan(xi*secd/(1.0-xn*tand))
		alpha = aa+tpa
		delta = np.arctan((xn+tand)*np.sin(aa)/(xi*secd))
		x = alpha
		y = delta
		tpi = 2.*np.pi
		for i,j in enumerate(x):
		       	if j>tpi: x[i]   = j - tpi
		       	if j<0.000: x[i] = j + tpi
		x = x*cts.rad2deg
		y = y*cts.rad2deg
		return x,y,distortcor	
	elif output=='standard':
		return xi,xn,distorcor
	else: 
		raise NameError('the type of coordinate should be: "equatorial" or "standard","{}" was given and is not accepted'.format(output))
	pass

################################################################################################
#
#		CLASSES AND FUNCTIONS
#
#################################################################################################
	
	

class MasterFrame:
	'''	
	The MasterFrame offers a reference system to compute the relative motion of the stars.
	Init with some reference frame, the one with lowest seeing.
	Match and transform every epoch to this one.
	This MasterFrame will keep the reference system of the initial frame and their ID's but with the stars positions averaged.
	Iterate.
	'''
	def __init__(self,name):
		self.ID=[]					#ID of the stars (based on a reference frame).
		self.pos=[[],[]]				#Positions of the stars after each iteration.
		#Load the positions of the stars in the reference frame and their Color-Magnitude if available.
		self.ID,self.pos[0],self.pos[1],self.mag_A,self.col_BA=np.loadtxt(name,skiprows=1,usecols=(0,3,4,5,6),unpack=True)

		self.added_pos=[[],[]]				#Sum the position XY of a star in each frame. Then average it.
		self.matchedID=np.zeros(self.ID.size)		#number of times that the star(ID) is found
		self.added_pos[0]=np.zeros(self.ID.size)
		self.added_pos[1]=np.zeros(self.ID.size)

	
	def constrain_region(self,pos_object,region_range):
		#Pick only stars in a square region around certain point. (For cutting the Chip).
		X_Index=Index_in_Region(self.pos[0],pos_object[0]-region_range,pos_object[0]+region_range)
		Y_Index=Index_in_Region(self.pos[1],pos_object[1]-region_range,pos_object[1]+region_range)
		constrain_index=np.intersect1d(X_Index,Y_Index)
		
		#Mask every property using the constrain index.
		self.ID=self.ID[constrain_index]
		self.pos[0]=self.pos[0][constrain_index]
		self.pos[1]=self.pos[1][constrain_index]
		self.mag_A=self.mag_A[constrain_index]
		self.col_BA=self.col_BA[constrain_index]

		self.matchedID=self.matchedID[constrain_index]
		self.added_pos[0]=self.added_pos[0][constrain_index]
		self.added_pos[1]=self.added_pos[1][constrain_index]
	
	
	def save(self,output):
		ascii.write([self.ID.astype(int),self.pos[0],self.pos[1],self.mag_A,self.col_BA],output,names=["#ID","X","Y","A","B-A"])

	def get_Masks(self,ID):
		mask_master=np.in1d(self.ID,ID) #Received ID's that are present on the masterframe ID's
		mask_epoch=np.in1d(ID,self.ID)  #Masterframe ID's that are present on received ID's.
		return mask_master,mask_epoch


	def add_frame(self,ID,pos):		
		#Add a frame, is must be alredy transformed to the reference frame system.
		#Receives the ID's found and the positions to be added according those ID's
		mask=np.in1d(self.ID,ID)				#Wich stars(ID's) must be added		
		self.added_pos[0][mask]=self.added_pos[0][mask]+pos[0]	#Add new position in X
		self.added_pos[1][mask]=self.added_pos[1][mask]+pos[1]	#Add new position in Y
		self.matchedID[mask]=self.matchedID[mask]+1		#These stars were added

	def compute_master(self,star_threshold):
		#Compute the final masterframe. Discard useless stars. Keep the objects of interest.
		
		#Average the position according the number of repeats to update Reference Positions.
		self.pos[0]=np.divide(self.added_pos[0],self.matchedID)
		self.pos[1]=np.divide(self.added_pos[1],self.matchedID)
		
		#Forget the stars that were found less times that star_threshold.
		#You could choose to keep some stars using: mask= np.union1d(threshold_index,another_index)
		threshold_index=np.where(self.matchedID>=star_threshold)[0]

		self.ID=self.ID[threshold_index]
		self.pos[0]=self.pos[0][threshold_index]
		self.pos[1]=self.pos[1][threshold_index]
		self.mag_A=self.mag_A[threshold_index]
		self.col_BA=self.col_BA[threshold_index]
		
		#Reset the added_pos for a following iteration.
		self.matchedID=np.zeros(self.ID.size)
		self.added_pos[0]=np.zeros(self.ID.size)
		self.added_pos[1]=np.zeros(self.ID.size)



def XYtoRADEC(name,extension):
	#Generate new catalogues using the RADEC information for ulterior matching with stilts.
	ID_image,x_image,y_image,mag_image=np.loadtxt(name+extension, usecols=(0,1,2,3),skiprows=3, unpack=True)
	ID_image=ID_image.astype(int)
	hdulist = pyfits.open(name+".fits")
	#RA_image,DEC_image,distcoord=vircam_xy2standard(x_image,y_image,hdulist[0].header)
	
	w = wcs.WCS(name+'.fits')
	coords = np.transpose((x_image,y_image))
	RA_image,DEC_image = np.transpose(w.wcs_pix2world(coords,1))

	ascii.write([ID_image,RA_image, DEC_image,x_image,y_image,mag_image], name+".dat", names=['#ID','RA', 'DEC','X','Y','mag'])
	
#Busca el seeing mas bajo
def lowest_seeing(files):
	nro_epocas = len(files)
	seeing = np.zeros(nro_epocas)
	
	for i,a in enumerate(files):
		hdu = pyfits.open(epoch_path +a+'.fits')
		s   = hdu[0].header['SEEING']
		seeing[i] += s

	index = seeing==seeing.min()
	archivo = np.array(files)[index]
	return index[0],archivo[0]

#Transformations of the original Match code. Includes Rotations and Scalings.
#Quadratic used to generate the masterframe.
def linear(coord,a,b,c):
	x=coord[0]
	y=coord[1]
	return a + b*x + c*y

def quadratic(coord,a,b,c,d,e,f):
	x=coord[0]
	y=coord[1]
	return a + b*x + c*y + d*np.square(x) + e*np.multiply(x,y) + f*np.square(y)

def cubic(coord,a,b,c,d,e,f,g,h):
	x=coord[0]
	y=coord[1]
	x2=np.square(x)
	y2=np.square(y)
	return a + b*x + c*y + d*x2 + e*np.multiply(x,y) + f*y2 + g*np.multiply(x,x2+y2) + h*np.multiply(y,x2+y2)




#Constrain the search in different ways. Obtain the Indexes of the objects in the constrain.
#The user can put these at different sections of the Code.
def Index_in_Radius(pos,pos0,rad):
	dist2=np.square(pos[0]-pos0[0]) +np.square(pos[1]-pos0[1])
	return np.where(dist2<=np.square(rad))[0]

def Index_out_Radius(pos,pos0,rad):
	dist2=np.square(pos[0]-pos0[0]) +np.square(pos[1]-pos0[1])
	return np.where(dist2>np.square(rad))[0]

def Index_in_Region(pos,inf,sup):
	return np.where((pos>=inf) & (pos<=sup))[0]

def Index_in_CMD(color,magnitude,color_range,magnitude_range):
	index_col=np.where((color>=color_range[0]) & (color<=color_range[1]))[0]
	index_mag=np.where((magnitude>=magnitude_range[0]) & (magnitude<=magnitude_range[1]))[0]
	return np.intersect1d(index_col,index_mag)

def Index_out_CMD(color,magnitude,color_range,magnitude_range):
	index_col=np.where((color<=color_range[0]) & (color>=color_range[1]))[0]
	index_mag=np.where((magnitude<=magnitude_range[0]) & (magnitude>=magnitude_range[1]))[0]
	return np.intersect1d(index_col,index_mag)



#Directory Handling.
def make_directory(folder):
	if not os.path.exists(folder):
	    	os.makedirs(folder)
	else:
		empty_directory(folder)

def empty_directory(folder):
	for the_file in os.listdir(folder):
		file_path = os.path.join(folder, the_file)
		try:
			os.unlink(file_path)
		except Exception, e:
			print e

##############################################################################################
#
#			USER VARIABLES
#
###############################################################################################


catalogue_extension=".dao"		#Original Catalogues extension.
epoch_path="epoch/"			#Original Files Folder. Must include Catalogue and fits image.
cmd_path="cmd/"			#CMD (B-A vs. A) folder. Must include two catalogues of the same epoch with different filters and their fits images.
#filter_A="b242_3_k_11-001"		#Name of the catalogue for filter A (No extension, no path)
filter_B="b249_5_j_13-001"		#Name of the catalogue for filter B (No extension, no path)


match_path="matched_epochs/"		#Output Folder where the Stilts match between epochs and the reference frame (with cmd information) are stored.
seeing_out="Seeing.dat"			#Seeing output File. For user usage only.
date_out="Dates.dat"			#MJD output. Needed for the proper motion Code.

cmd_filename="CMD.dat"			#Output: Reference frame with information of the CMD included.
master_filename="Master.dat"		#Masterframe Output.

master_iterations=4			#Number of iterations to compute the masterframe
master_star_threshold=10		#Minimum number of times that a star must be found in the masterframe to be considered. 
match_tolerance=0.1			#Tolerance for stilts match separation in arcsec
ref_index=36	   			#IMAGE with good seeing. (Index Position in the array at epoch_path)


#Control Booleans. To don't go over innecesary parts again.
get_RADEC=True
get_Seeing=True
get_Date=True

match_cmd=True
match_epochs=True



#############################################################################################
#
#			PIPELINE
#
#############################################################################################



#GET FILE NAMES
epoch_names=[]
for file in os.listdir(epoch_path):
    if file.endswith(catalogue_extension):
        epoch_names.append(file.split(".")[0])
epoch_names.sort()
print epoch_names



if get_Seeing:
	seeing_file=open(seeing_out,'w')
	for name in epoch_names:
		hdulist = pyfits.open(epoch_path+name+".fits")
		seeing_file.write(str(hdulist[0].header["SEEING"])+"\n")
	seeing_file.close()

ref_index, ref_file = lowest_seeing(epoch_names)
print "Seeing minimo en %s" % ref_file
filter_A = ref_file

cmd_names=[filter_B,filter_A]
print cmd_names





#TRANSFORM to RA DEC
#(only for stilts match)
if get_RADEC:
	for name in epoch_names:
		print "RA-DEC CONVERTING: "+name
		XYtoRADEC(epoch_path+name,catalogue_extension)
	for name in cmd_names:
		print "RA-DEC CONVERTING: "+name
		XYtoRADEC(cmd_path+name,catalogue_extension)


cmd_names=[filter_B,filter_A]

if get_Date:
	date_file=open(date_out,'w')
	for name in epoch_names:
		hdulist = pyfits.open(epoch_path+name+".fits")
		date_file.write(str(hdulist[0].header["MJD-OBS"])+"\n")
	date_file.close()


#Generate the CMD.dat file. Use information of the CMD epoch + Reference epoch.
if match_cmd:
	#Match the two Images of the same epoch for the CMD
	in_match1="in1="+cmd_path+cmd_names[0]+".dat "	#Filter B#
	in_match2="in2="+cmd_path+cmd_names[1]+".dat "	#Filter A#
	out_match="out="+cmd_path+"CMD"+".match "
	params="params=%.1f"%match_tolerance
	os.system("sh stilts tmatch2 ifmt1=ascii ifmt2=ascii matcher=sky ofmt=ascii values1=\"RA DEC\" values2=\"RA DEC\" "+in_match1+in_match2+out_match+params)
	
	#Match that CMD with the Best Image to use as reference.
	in_match1="in1="+epoch_path+epoch_names[ref_index]+".dat "
	in_match2="in2="+cmd_path+"CMD"+".match "
	out_match="out="+cmd_path+"CMD_Ref"+".match "
	os.system("sh stilts tmatch2 ifmt1=ascii ifmt2=ascii matcher=sky ofmt=ascii values1=\"RA DEC\" values2=\"RA_2 DEC_2\" "+in_match1+in_match2+out_match+params)
	
	#Save the Reference Image with all the original stars AND the Color-Magnitude found in the match..
	#If CMD was not found for a star mark it as False
	cmd_match=cmd_path+"CMD_Ref"+".match"
	ref_name=epoch_path+epoch_names[ref_index]+".dat"

	ID,RA,DEC,X,Y= np.loadtxt(ref_name,skiprows=1, usecols=(0,1,2,3,4),unpack=True)	#read reference image
	ID_matched,B,A= np.loadtxt(cmd_match,skiprows=1, usecols=(0,11,17),unpack=True)	#read match of reference image & cmd
	
	#Set an absurd value for the cmd. Mark those without cmd as false using a mask based on the match.
	mag_A=np.zeros(ID.size)-10
	col_BA=np.zeros(ID.size)-10
	cmd_mask=np.in1d(ID,ID_matched)	
	
	#Set the real value for stars with cmd using a mask.
	mag_A[cmd_mask]=A
	col_BA[cmd_mask]=B-A

	#Output the CMD file with information about positions.
	ascii.write([ID.astype(int),RA, DEC,X,Y,mag_A,col_BA,cmd_mask], cmd_filename, names=['#ID','RA', 'DEC','X','Y','A','B-A','bool_CMD'])



#MATCH EACH EPOCH WITH THE REFERENCE EPOCH(Including CMD Information)(STILTS)
if match_epochs:
	make_directory(match_path)
	for name in epoch_names:
		print "STILTS MATCHING "+name
		in_ref="in1="+cmd_filename+" "
		in_epoch="in2="+epoch_path+name+".dat "
		out_match="out="+match_path+name+".match "
		params="params=%.1f"%match_tolerance
		os.system("sh stilts tmatch2 ifmt1=ascii ifmt2=ascii matcher=sky ofmt=ascii values1=\"RA DEC\" values2=\"RA DEC\" "+in_ref+in_epoch+out_match+params)
	     	


#CREATE MASTER FRAME (based on the better looking epoch) (CMD incorporated)
masterframe=MasterFrame(cmd_filename)
#You can call here masterframe.constrain_region() to use only a region of the chip for the masterframe.
#masterframe.constrain_region([0,0],500)
for i in range(master_iterations):
	for name in epoch_names:		
		print "MASTER FRAMING: "+name+"\nIteration: "+str(i+1)
		in_match=match_path+name+".match"
		pos_ref=[[],[]]
		pos_epoch=[[],[]]	
		pos_trans=[[],[]]		

		#read matched_file.
		#Note that since they were matched, the index corresponds to a matched pair
		#Remember we keep using the ID's of the original reference epoch.
		matchedID,pos_epoch[0],pos_epoch[1],separation=np.loadtxt(in_match,skiprows=1, usecols=(0,11,12,14),unpack=True)

		#Which stars from the epoch are present in the masterframe
		#Wich stars from the masterframe are present in the epoch
		mask_master,mask_epoch=masterframe.get_Masks(matchedID)
		pos_ref=[masterframe.pos[0][mask_master],masterframe.pos[1][mask_master]]
		pos_epoch=[pos_epoch[0][mask_epoch],pos_epoch[1][mask_epoch]]
		mag=masterframe.mag_A[mask_master]
		col=masterframe.col_BA[mask_master]



		#GLOBAL QUADRATIC TRANSFORM TO THE INITIAL REFERENCE FRAME
		#The transformation is applied to ALL matched stars

		popt_X,pcov_X= curve_fit(quadratic,[pos_epoch[0],pos_epoch[1]],pos_ref[0])
		popt_Y,pcov_Y= curve_fit(quadratic,[pos_epoch[0],pos_epoch[1]],pos_ref[1])
		
		pos_trans[0]=quadratic([pos_epoch[0],pos_epoch[1]],popt_X[0],popt_X[1],popt_X[2],popt_X[3],popt_X[4],popt_X[5])
		pos_trans[1]=quadratic([pos_epoch[0],pos_epoch[1]],popt_Y[0],popt_Y[1],popt_Y[2],popt_Y[3],popt_Y[4],popt_Y[5])
		
		#Add the matched stars to the frame.
		masterframe.add_frame(matchedID,pos_trans)
	
	#Update the masterframe using the transformed positions to the reference system.
	masterframe.compute_master(master_star_threshold)
masterframe.save(master_filename)
#Got the Masterframe. Includes ID's + Positions in the reference epoch system and Color-Magnitude of the CMD epoch.



