folder    = 'hola/hola'
ref_image = 'hola_soy_m28.dao' #Img name

#########
import os
import glob


os.chdir(folder)
imgs     = sorted(glob.glob('*.fits'))
catalog  = sorted(glob.glob('*.dao'))

for cat in catalog:

    if cat == ref_image:
        continue

    os.system('java -jar -Xmx4096M stilts.jar tmatch2 in1='+ref_image+' values1="RA DEC" ifmt1=ascii in2='+cat+' values2="RA DEC" ifmt2=ascii matcher=sky params="0.3" find=best join=all1 out='+cat.replace('.dao','_'+cat[-7:-4]+'.match')+' ofmt=ascii')
