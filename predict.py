
import os
import numpy as np
import config
import utils
from utils.DataGenerator import DataGen, DataGenerator_Online

from tensorflow.python.keras.callbacks import  ModelCheckpoint,CSVLogger
from tensorflow.keras.models import load_model

import matplotlib.pyplot as plt
import cv2



if __name__ == '__main__':
    
    
    batch_size = config.batch_size    
    batch_count = config.batch_count    
    image_dir = config.image_dir    
    mask_dir = config.mask_dir    
    image_height = config.image_height    
    image_width = config.image_width    
    nepochs = config.epochs_no
    
    test_class = config.test_class
    test_vid = config.test_dir
    test_mask = config.test_mask_dir
    mode = config.mode
    Results = config.result_path
    
    filename = r'models/model-ep990-loss0.000-val_loss0.000.h5'
       #filename = r".\models_1000_1\model- ep001-loss0.051-val_loss0.051.h5"
    model = load_model(filename)
    
    frame_ids = os.listdir(test_vid)
    
    online_train_gen = DataGenerator_Online(batch_count, image_dir, mask_dir, test_class, image_height, image_width,batch_size)
    
    filepath = r'models/online/model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'        
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    csv_logger = CSVLogger('training_online.csv', separator = ',')
    
    
    online_model = model
    online_model.fit(online_train_gen ,batch_size=batch_size,callbacks=[checkpoint,csv_logger],
                  epochs=nepochs, verbose=1)
    
    for i in range(1,len(frame_ids)):
        
            im_org = cv2.imread(os.path.join(test_vid,frame_ids[i]))  
            im_org.resize((image_height, image_width,im_org.shape[2]))
            im = np.array(im_org, dtype="float32") / 255.0 
            
            label_path = os.path.join(test_mask,frame_ids[i-1])
            label_path = os.path.splitext(label_path)[0]
            label_path = label_path + ".png"
            label_org = cv2.imread(label_path)
            label_org.resize((image_height,image_width,label_org.shape[2]))
            label = cv2.cvtColor(label_org, cv2.COLOR_BGR2GRAY)   
            k = (np.unique(label.flatten()))[0]
            _,im_label = cv2.threshold(label,(k+10),255,cv2.THRESH_BINARY) 
            im_label = np.array(im_label,dtype = "float32")/255.0   
            
            new = np.dstack((im, im_label))
            
            gt_label_path = os.path.join(test_mask,frame_ids[i])
            gt_label_path = os.path.splitext(label_path)[0]
            gt_label_path = gt_label_path + ".png"
            gt_label_org = cv2.imread(gt_label_path)
            gt_label_org.resize((image_height,image_width,gt_label_org.shape[2]))
            gt_label = cv2.cvtColor(label_org, cv2.COLOR_BGR2GRAY)   
            k = (np.unique(gt_label.flatten()))[0]
            _,gt_im_label = cv2.threshold(gt_label,(k+10),255,cv2.THRESH_BINARY) 
            gt_im_label = np.array(gt_im_label,dtype = "float32")/255.0 
            
            if mode == 'offline':
                
                pred = model.predict(new)
                finalPred = (pred>0.5).astype(np.unit8)
                path = os.path.join(Results, test_class) + "\off" + str(i) + ".png"
                cv2.imwrite(path,np.squeeze(finalPred))
                
                evalResult = model.evaluate(finalPred,gt_im_label,batch_size = 1)
                print("acc :", evalResult[1])
                
            
                
            elif mode == 'online':
                
                pred = online_model.predict(new)
        
        
        
