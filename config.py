
import os

# define the base path to the project and then use it to derive
# the path to the input images and annotation files
BASE_PATH = r'/mnt/batch/tasks/shared/LS_root/mounts/clusters/pocdeepeastusriddhi3/code/Users/riddhi.chaudhuri/MaskTrack_Solution'

image_dir = os.path.sep.join([BASE_PATH, "DAVIS2017/Train"])
mask_dir =  os.path.sep.join([BASE_PATH, "DAVIS2017/Train_Annotated"])

batch_size = 8    
batch_count = 10   
  
image_height = 480    
image_width = 960  
epochs_no = 2


