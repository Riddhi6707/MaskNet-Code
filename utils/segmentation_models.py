import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam


import segmentation_models as sm
from segmentation_models.losses import bce_jaccard_loss
sm.set_framework('tf.keras')
K.set_image_data_format('channels_last')

def build_model(image_size,  class_count=1, channels=16,BACKBONE = None):
    

        input_data = Input(image_size+(1,), dtype='float32')
        
        
        base_model = sm.Unet(backbone_name=BACKBONE,encoder_weights=None,input_shape=image_size + (1,))
        out_data = base_model(input_data)
        #out_data = Conv2D(1, (1, 1))(out_data_temp)
        
        #    base_model = sm.Linknet(backbone_name=BACKBONE,encoder_weights=None, input_shape=image_size + (1,))
        #    out_data = base_model(input_data)
        
        #base_model = sm.Unet(backbone_name=BACKBONE, encoder_weights='imagenet')
        #l1 = Conv2D(3, (1, 1))(input_data) # map N channels data to 3 channels
        #out_data = base_model(l1)
        
        tf.identity(out_data, name="bcm_output_raw") 
        output = tf.cast(tf.round(out_data), dtype=tf.int32, name="bcm_output")
        
        model = Model(inputs=input_data, outputs=out_data, name=base_model.name)
        
        loss = sm.losses.JaccardLoss() +  sm.losses.BinaryFocalLoss()  #

        dice =  sm.metrics.IOUScore() 
        
        opt = Adam(lr = .0002)
        
        model.compile(optimizer=opt, loss=loss, metrics=[dice])

        return model


