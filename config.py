# import the necessary packages
import os

# define the base path to the input dataset and then use it to derive
# the path to the input images and annotation CSV files
BASE_PATH = r"C:\Users\RRay Cha\Desktop\WORK\Coding_Challenge\MaskTrack_Solution"
IMAGES_PATH = os.path.sep.join([BASE_PATH, "images"])
ANNOTS_PATH = os.path.sep.join([BASE_PATH, "annotations"])


image_dir = os.path.sep.join([BASE_PATH, "DAVIS2017\Train"])
mask_dir =  os.path.sep.join([BASE_PATH, "DAVIS2017\Train_Annotated"])

batch_size = 32    
batch_count = 200   
  
image_height = 480    
image_width = 960  
nepochs = 50


