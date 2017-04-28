import numpy as np
import cv2 as cv
import data
import image
 # “Pickling” is the process whereby a Python object hierarchy is converted into a byte stream,
 # and “unpickling” is the inverse operation,
 # whereby a byte stream (from a binary file or bytes-like object) is converted back into an object hierarchy. 
 # Pickling (and unpickling) is alternatively known as “serialization”, “marshalling,” [1] or “flattening”;
 # however, to avoid confusion, the terms used here are “pickling” and “unpickling”. 
import pickle

#call to data.py
#x is a dictionary key value pair where key is folder name and value is a list of files inside it,folder is key
#folder:list of files

x = data.get_dir()
done = 0
for folder in x:
    try:
        #rb is read binary file
        #if try fails then except will run 
        # this is to check if that particular file is already done or not , if not then go to except 
        #we directly jump to done += 1 when try is successful
        f = open('dat/' + folder, 'rb')
    except:
        try:
            print('Folder: ', folder, '\tDone: ', done)
            #control goes to data.py after below line
            #now data is imlist (containing all frames belonging to one folder)
            #dat same as imlist they are 2d list whose 2 element is a 3d array of rgb
            #imlist contains image name as 1 element and 2 element as 3d array of rgb
            dat = data.read_dir({folder: x[folder]}, False)[folder]
            #changes is a list
            changes = []
            for i in range(len(dat) - 1):
                print('\t', i, '/', len(dat) - 1)
                #call to image.py function
                #we are starting with one bcoz at zero we have stored img name
                #dat[i+1][1] have 3d array of rgb values of that image
                im_change = image.image_change(dat[i + 1][1], dat[i][1])
                #changes will have no of connected components
                changes.append(image.get_cc_im(im_change))
            #if this fails then its except will run
            f = open('dat/' + folder, 'wb')
            pickle.dump(changes, f)
            f.close()
        except:
    		#this is printed when everything fails
            print('Folder ', folder, ' FAILED!')
    done += 1
