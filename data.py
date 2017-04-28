import cv2
import os

def get_dir(dir_name = None):
    "Get directory file names"
    if dir_name is None:
        #folders is a list which contains all folders in data as an element
        folders = [x for x in os.listdir('Data') if os.path.isdir('/'.join(('Data', x)))]
    else:
        folders = [x for x in os.listdir('Data') if os.path.isdir('/'.join(('Data', x))) and x == dir_name]
    
    #frames is a dictionary key value pair where key is folder name and value is a list of files inside it
    frames = {}
                   
    for folder in folders:
        frames[folder] = os.listdir('/'.join(('Data', folder)))
        frames[folder].sort()
    #after this control goes back to extract_data.py
    return frames

def read_dir(frames, prompt = False):
    "Read the specified images. Frames must be a output of get_dir()"
    #images is also a dictionary key is folder and value is imlist a list
    images = {}
    for folder in frames:
        if prompt:
            print(folder+'---')
        imlist = []
        for frame in frames[folder]:
            if prompt:
                print(folder+'/'+frame+'...')
            #it is to read an image.  cv2.IMREAD_COLOR:Loads a color image. 
            # Any transparency of image will be neglected.It is the default flag.
            #imlist contains image name also, imlist becomes 2D
            imlist.append([frame, cv2.imread('/'.join(('Data', folder, frame)))])
        if prompt:
            print('---')
        images[folder] = imlist
    return images
                                    

def get_class(dir_name):
    "Returns class of a filename. True for forward, False for backward"
    if dir_name.startswith('F'):
        return True
    else:
        return False

def rev_name(dir_name):
    "Interchange F_* and N_*"
    assert dir_name.startswith('F_') or dir_name.startswith('B_')
    prefix = dir_name[:2]
    suffix = dir_name[2:]
    prefix = 'B_' if prefix == 'F_' else 'F_'
    return prefix + suffix

def rev_dir(frames):
    "Produce reverse sample by reversing frame order"
    r_dict = {}
    for folder in frames:
        r_folder = rev_name(folder)
        assert r_folder not in frames
        r_dict[r_folder] = frames[folder][::-1]

    frames.update(r_dict)

