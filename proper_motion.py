import os
import numpy as np
from astropy.io import ascii
from scipy.optimize import curve_fit
from scipy.spatial import KDTree
import matplotlib.pyplot as plt



################################################################################################
#
#		CLASSES AND FUNCTIONS
#
#################################################################################################

	
class MasterFrame:
	#READ ONLY MASTERFRAME VERSION.	
	def __init__(self,name):
		self.ID=[]		
		self.pos=[[],[]]
		self.ID,self.pos[0],self.pos[1],self.magnitude,self.color=np.loadtxt(name,skiprows=1,usecols=(0,1,2,3,4),unpack=True)
	
	def get_Masks(self,ID):
		mask_master=np.in1d(self.ID,ID)
		mask_epoch=np.in1d(ID,self.ID)
		return mask_master,mask_epoch



class StarsPM:
	def __init__(self,output_stars,objectID, pos_x,pos_y, color,magnitude):
		
		self.output=output_stars
		self.starsID=objectID
		self.pos_x=pos_x
		self.pos_y=pos_y
		self.color=color
		self.magnitude=magnitude
		
		for i in range(self.starsID.size):
			starfile=open(self.output+"Motion"+str(int(self.starsID[i]))+".dat", 'a')
			starfile.write("#epoch(yr)  Motion_X(pix)  Motion_Y(pix)\n")			
			starfile.close()	


	def Add_Motion(self,Added_ID,epoch,Motion_X,Motion_Y):
		#ADD THE POSITION OF THE STAR AT EPOCH AFTER TRANSFORMING IT TO THE REFERENCE SYSTEM 		
		for i in range(Added_ID.size):
			starfile=open(self.output+"Motion"+str(int(Added_ID[i]))+".dat", 'a')
			starfile.write(str(epoch)+" "+str(Motion_X[i])+" "+str(Motion_Y[i])+"\n")
			starfile.close()

	def Calculate_ProperMotion(self, output_PM_Stars):
		
		ProperMotion=[[],[],[],[]]	#PM_X,PM_Y,PM_X_err,PM_Y_err.
		star_mask=[]			#Stars with valid proper_motion.
		for i in range(self.starsID.size):
			name=self.output+"Motion"+str(int(self.starsID[i]))
			dates,Motion_X,Motion_Y=np.loadtxt(name+".dat",skiprows=1,unpack=True)
			
			#Cannot fit with few epochs. 10 epochs to have trusty results.
			if dates.size <=10:
				continue
			
			popt_X,pcov_X= curve_fit(linearPM,dates,Motion_X)
			popt_Y,pcov_Y= curve_fit(linearPM,dates,Motion_Y)
			
			#skip if curvefit couldn't compute the error
			if type(pcov_X)!= np.ndarray or type(pcov_Y)!= np.ndarray:
				continue
		
			#skip if error bigger than 1 pixel. Curvefit just didnt worked.
			if np.sqrt(pcov_X[0][0])>=1 or np.sqrt(pcov_X[0][0])>=1:
				continue

			#Valid Proper Motion. Transform from pix->mas.
			ProperMotion[0].append(popt_X[0]*vc_pix*1000)
			ProperMotion[1].append(popt_Y[0]*vc_pix*1000)
			ProperMotion[2].append(np.sqrt(pcov_X[0][0])*vc_pix*1000)
			ProperMotion[3].append(np.sqrt(pcov_Y[0][0])*vc_pix*1000)
			
			star_mask.append(i)
		
		for i in range(len(ProperMotion)):
			ProperMotion[i]=np.array(ProperMotion[i])
		PM=np.sqrt(np.square(ProperMotion[0])+np.square(ProperMotion[1]))
		
		#Discard stars with non-computed proper_motion.
		star_mask=np.array(star_mask).astype(int)
		StarID=self.starsID[star_mask].astype(int)
		Color=self.color[star_mask]
		Magnitude=self.magnitude[star_mask]
		Pos_X=self.pos_x[star_mask]
		Pos_Y=self.pos_y[star_mask]

		#Save Star Information.
		ascii.write([StarID,Pos_X,Pos_Y,Color,Magnitude,PM,ProperMotion[0],ProperMotion[1],ProperMotion[2],ProperMotion[3]],output_PM_Stars,names=["#ID","X(pix)","Y(pix)","COLOR","MAGNITUDE","ProperMotion(mas/year)","PM_X(mas/year)","PM_Y(mas/year)","err_PM_X(mas/year)","err_PM_Y(mas/year)"])
		

	


#Linear equation. Used for proper motions.
def linearPM(dates,PM,zero_point):
	return dates*PM+zero_point


#Transformations of the original Match code. Includes Rotations and Scalings.
#Linear used to transform each epoch to the masterframe.
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



def Index_from_List(starsID,ID_filename):
	#Get the Stars ID's from ID_filename that are present in StarsID.
	listID,buff_array=np.loadtxt(ID_filename,skiprows=1,unpack=True,usecols=(0,0))
	return np.in1d(starsID,listID)



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
	index_cmd=[]	
	for i in range(len(col_range)):
		index_col=np.where((color>=color_range[i][0]) & (color<=color_range[i][1]))[0]
		index_mag=np.where((magnitude>=magnitude_range[i][0]) & (magnitude<=magnitude_range[i][1]))[0]
		index_cmd.append(np.intersect1d(index_col,index_mag))

	index=index_cmd[0]	
	for i in range(len(index_cmd)-1):
		index=np.union1d(index,index_cmd[i+1])
	return index

def Index_out_CMD(color,magnitude,color_range,magnitude_range):
	index_cmd=[]
	for i in range(len(col_range)):
		index_col=np.where((color<color_range[i][0]) | (color>color_range[i][1]))[0]
		index_mag=np.where((magnitude<magnitude_range[i][0]) | (magnitude>magnitude_range[i][1]))[0]
		index_cmd.append(np.union1d(index_col,index_mag))
	
	index=index_cmd[0]	
	for i in range(len(index_cmd)-1):
		index=np.intersect1d(index,index_cmd[i+1])
	return index



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


catalogue_extension=".dao"				#Original Catalogues extension.
epoch_path="epoch/"					#Original Files Folders
match_path="matched_epochs/"				#Epochs Match Folder 
starmotion_path="Stars_Motion/"				#Motion of all the stars at each epoch
ProperMotion_outfile="Results/PM_BPR100.dat"	#Stars Proper Motion file (FINAL OUTPUT)


master_file="Master.dat"				#Masterframe
date_file="Dates.dat"					#File with dates. Sorted by epoch in MJD, JD, MBJD, etc.

#ID_filename="IndexList/PM_2sigma.dat"			#Stars to compute the transformations. Should be the ones with expected ProperMotion zero.
ID_filename = "Results/BPR.dat"

local_transformation=True				#Use Local Transformation or Global Transformation
neighbor_search=100				#Neighbor search for local transformation.
vc_pix=0.34						#Vircam Pix-arcsecond scale.




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


make_directory(starmotion_path)
masterframe=MasterFrame(master_file)

#Read the dates. Convert to years.
dates=np.loadtxt(date_file)
dates=(dates-dates[0])/365.25


#REGISTER ALL THE STARS AVAILABLE IN MASTERFRAME TO TRACE THEIR INDIVIDIAL PROPER MOTIONS
starsPM=StarsPM(starmotion_path,masterframe.ID,masterframe.pos[0],masterframe.pos[1],masterframe.color,masterframe.magnitude)
for ep in range(len(epoch_names)):
	name=epoch_names[ep]
	print "PROPER MOTION: "+name
	in_match=match_path+name+".match"
	
	pos_ref=[[],[]]		#Reference position. (Masterframe)
	pos_epoch=[[],[]]	#Epoch Position.
	pos_trans=[[],[]]	#Epoch Position with Linear Transformation from Epoch position to the Reference position.


	#Foreach epoch read it's match_file.	
	#Note that since they were matched the index corresponds to a matched pair.
	#Remember we keep using the ID's of the original reference epoch.
	matchedID,pos_epoch[0],pos_epoch[1],separation=np.loadtxt(in_match,skiprows=1, usecols=(0,11,12,14),unpack=True)

	#Which stars from the epoch are present in the masterframe
	#Wich stars from the masterframe are present in the epoch
	mask_master,mask_epoch=masterframe.get_Masks(matchedID)

	stars_ID=masterframe.ID[mask_master]
	pos_ref=[masterframe.pos[0][mask_master],masterframe.pos[1][mask_master]]
	pos_epoch=[pos_epoch[0][mask_epoch],pos_epoch[1][mask_epoch]]
	mag=masterframe.magnitude[mask_master]
	col=masterframe.color[mask_master]

	
	#################################################################
	#	
	#	FIT A LINEAR TRANSFORMATION (Local/CMD/Region/IndexList)	
	#	
	#################################################################
	
	#LOCAL TRANSFORMATION USING THE CLOSEST NEIGHBORS AVAILABLE IN ID_filename INDEX LIST.
	if local_transformation:
		index=Index_from_List(stars_ID,ID_filename)		#Indexes of the ID_file that exist on stars ID. (apply on stars_ID, pos_ref, pos_epoch)
		tree=KDTree(zip(pos_ref[0][index],pos_ref[1][index]))	#KDTree with the stars at the file.
		
		neighbor_distance,neighbor_index=tree.query(zip(pos_ref[0],pos_ref[1]), k=neighbor_search)	
		#Foreach star in pos_ref find the k closest neighbors at the tree (the ID_file) (apply neighbor_index in tree.data,pos_ref[index],pos_epoch[index])
		#neighbor_index is to be applied on data of the tree dimension,  neighbor_index.size=query_search*k		

		#Positions at the reference and epoch systems that are used to the local transformation. (Apply neighbor_index on these ones).
		pos_epoch_selection=[pos_epoch[0][index],pos_epoch[1][index]]
		pos_ref_selection=[pos_ref[0][index],pos_ref[1][index]]

		pos_trans[0]=np.zeros(stars_ID.size)
		pos_trans[1]=np.zeros(stars_ID.size)
	
		
		#For every Star find it's local transformation using the neighbors from the IndexList.
		for i in range(stars_ID.size):
		#for i in range(74,77):
			n_idx=neighbor_index[i]
			#Transformation to be applied to the star i

			popt_X,pcov_X= curve_fit(linear,[pos_epoch_selection[0][n_idx],pos_epoch_selection[1][n_idx]],pos_ref_selection[0][n_idx])
			popt_Y,pcov_Y= curve_fit(linear,[pos_epoch_selection[0][n_idx],pos_epoch_selection[1][n_idx]],pos_ref_selection[1][n_idx])

			pos_trans[0][i]=linear([pos_epoch[0][i],pos_epoch[1][i]],popt_X[0],popt_X[1],popt_X[2])
			pos_trans[1][i]=linear([pos_epoch[0][i],pos_epoch[1][i]],popt_Y[0],popt_Y[1],popt_Y[2])




	#LINEAR FIT USING STARS FROM ID_filename INDEX LIST AS REFERENCE FOR ALL THE IMAGE. GLOBAL TRANSFORMATION.
	else:	
		index=Index_from_List(stars_ID,ID_filename)

		popt_X,pcov_X= curve_fit(linear,[pos_epoch[0][index],pos_epoch[1][index]],pos_ref[0][index])
		popt_Y,pcov_Y= curve_fit(linear,[pos_epoch[0][index],pos_epoch[1][index]],pos_ref[1][index])
	
		pos_trans[0]=linear([pos_epoch[0],pos_epoch[1]],popt_X[0],popt_X[1],popt_X[2])
		pos_trans[1]=linear([pos_epoch[0],pos_epoch[1]],popt_Y[0],popt_Y[1],popt_Y[2])

		
	#################################################################
	#	
	#	PROPER MOTIONS AT THIS EPOCH	
	#	
	#################################################################
	#OBTAINED THE POSITION IN THE REFERENCE SYSTEM AFTER THE TRANSFORMATION		
	#Add epoch statistics here using constrain functions.	
	
	#Compute the Motion at this epoch
	Motion_x=pos_trans[0]-pos_ref[0]
	Motion_y=pos_trans[1]-pos_ref[1]
	Motion=np.sqrt(np.square(Motion_x) + np.square(Motion_y))
	
	#SAVE THE MOTION OF EACH STAR AT THIS EPOCH
	starsPM.Add_Motion(stars_ID,dates[ep],Motion_x,Motion_y)
	

	#################################################################
	#	
	#	PLOT RESULTS OF EACH EPOCH	
	#	
	#################################################################
	#Add Plots of your statistics here using constrain functions.
	#Print Last Transformation.
	param_X = ['a','b','c']
	param_Y= ['d','e','f']		
	print name
	for i in range(len(popt_X)):
	    print param_X[i]+': ',popt_X[i],'+-',np.sqrt(pcov_X[i,i])
	for i in range(len(popt_Y)):
	    print param_Y[i]+': ',popt_Y[i],'+-',np.sqrt(pcov_Y[i,i])
		

#COMPUTE STARS INDIVIDUAL PROPER MOTIONS
starsPM.Calculate_ProperMotion(ProperMotion_outfile)



