
import os
from tensorflow.keras.utils import Sequence
import numpy as np
#from PIL import Image
import cv2
import albumentations as A

# ================================================================== #
# CUSTOM DATA GENERATOR CLASS



class DataGen(Sequence):
    def __init__(self,  batch_count,image_dir, mask_dir, img_height, img_width, batch_size, mode = 'Train'):
       
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size
        self.bones = bones
        self.nclasses = len(bones)
        self.batch_x =  None 
        self.batch_y = None  
        self.augTransform = augTransform
        self.mode = mode       
        self.batch_count = batch_count
        self.class_count = np.zeros(len(os.listdir(self.image_dir))).tolist()
        self.classes = os.listdir(self.image_dir)

    def __len__(self):
              
        return  self.batch_count if self.batch_count >= len(os.listdir(self.image_dir)) else len(os.listdir(self.image_dir))

    def __getitem__(self, idx):
        
        if self.mode == "Train" :
            
            self.batch_x =  np.zeros((self.batch_size, self.img_height, self.img_width, 4))
            self.batch_y = np.zeros((self.batch_size, self.img_height, self.img_width, 1))
            idx = np.random.randint(0,len(self.classes))
            self.class_count[idx] += 1
            
            if self.class_count[idx] == np.floor(self.batch_count/len(os.listdir(self.image_dir))):
                (self.classes).remove(str(self.classes[idx]))
                self.class_count.pop(idx)
            
            selected_class_video = os.path.join(self.image_dir,str(self.classes[idx]))
            selected_class_mask =  os.path.join(self.mask_dir,str(self.classes[idx]))
            
           
            frame_ids = os.listdir(selected_class_video)
            ids = np.random.randint(1, len(frame_ids) - 3, self.batch_size)
            ct = 0
            for i in ids:
                     
                    im_org = cv2.imread(os.path.join(selected_class_video,frame_ids[i]))
                    im = np.asarray(im_org.resize((self.img_height, self.img_width)))
                    im = np.asarray(im / 255.0, dtype=np.float)
                    
                    mask_path = os.path.join(selected_class_mask,frame_ids[i-1])
                    mask_path = os.path.splitext(mask_path)[0]
                    mask_path = mask_path + ".png"
                    mask_org = cv2.imread(mask_path)
                    mask = np.asarray(mask_org.resize((self.img_height, self.img_width)))                   
                    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                    k = (np.unique(mask.flatten()))[1]
                    _,thresh = cv2.threshold(mask,(k-10),255,cv2.THRESH_BINARY)
                    x = np.random.randint(0,25,1)
                    y = np.random.randint(0,25,1)
                    M = np.float32([[1,0,x],[0,1,y]])
                    trans_thresh = cv2.warpAffine(thresh,M,( self.img_width,self.img_height))
                    trans_thresh = np.asarray(trans_thresh / 255.0, dtype=np.float)
                    aug = A.ElasticTransform(p=1, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03)
                    np.random.seed(7)
                    augmented = aug(image=im, mask= trans_thresh)
                    mask_final = augmented['mask']
                    new = np.dstack((im, mask_final))
                     
                    label_path = os.path.join(selected_class_mask,frame_ids[i])
                    label_path = os.path.splitext(mask_path)[0]
                    label_path = label_path + ".png"
                    label_org = cv2.imread(label_path)
                    label = np.asarray(label_org.resize((self.img_height, self.img_width)))
                    label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
                    k = (np.unique(label.flatten()))[1]
                    _,im_label = cv2.threshold(label,(k-10),255,cv2.THRESH_BINARY) 
                    thresh = np.asarray(label / 255.0, dtype=np.float)
                    
                    
                   
                    self.batch_y[ct, :, :, 0] = im_label[:, :]
                    self.batch_x[ct, :, :, :] = new[:, :, :]
                    ct += 1
           
            
           # return self.batch_x, self.batch_y
        
        elif   self.mode == "Valid" :
        
            self.batch_x =  np.zeros((2, self.img_height, self.img_width, 4))
            self.batch_y = np.zeros((2, self.img_height, self.img_width, 1))
        
            idx = np.random.randint(0,len(self.classes))
           
            
            selected_class_video = os.path.join(self.image_dir,str(self.classes[idx]))
            selected_class_mask =  os.path.join(self.mask_dir,str(self.classes[idx]))
            
           
            frame_ids = os.listdir(selected_class_video)
            ids = np.random.randint(1, len(frame_ids) - 3, 2)
            ct = 0
            for i in range(0,2):
                     
                    im_org = cv2.imread(os.path.join(selected_class_video,frame_ids[len(frame_ids)-3+i]))
                    im = np.asarray(im_org.resize((self.img_height, self.img_width)))
                    im = np.asarray(im / 255.0, dtype=np.float)
                    
                    mask_path = os.path.join(selected_class_mask,frame_ids[len(frame_ids)-4+i])
                    mask_path = os.path.splitext(mask_path)[0]
                    mask_path = mask_path + ".png"
                    mask_org = cv2.imread(mask_path)
                    mask = np.asarray(mask_org.resize((self.img_height, self.img_width)))      
                    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                    k = (np.unique(mask.flatten()))[1]
                    _,thresh = cv2.threshold(mask,(k-10),255,cv2.THRESH_BINARY)
                    thresh = np.asarray(thresh / 255.0, dtype=np.float)
                    new = np.dstack((im, thresh))
                    
                    label_path = os.path.join(selected_class_mask,frame_ids[len(frame_ids)-3+i])
                    label_path = os.path.splitext(mask_path)[0]
                    label_path = label_path + ".png"
                    label = cv2.imread(label_path)
                    label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
                    k = (np.unique(label.flatten()))[1]
                    _,im_label = cv2.threshold(label,(k-10),255,cv2.THRESH_BINARY)
                    
                    self.batch_y[ct, :, :, 0] = im_label[:, :]
                    self.batch_x[ct, :, :, :] = new[:, :, :]
                    ct +=1
           
           
            return self.batch_x, self.batch_y
                

    def get_label(self):
        return self.batch_x, self.batch_y