
# coding: utf-8

# In[1]:
import numpy as np

import tensorflow.keras as keras
from tensorflow.keras import layers


# In[4]:


# Y_label=keras.utils.to_categorical(Y_label, 3)
from tensorflow.keras import backend as K


# In[5]:


def iou_loss_core(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    union = K.sum(y_true,-1) + K.sum(y_pred,-1) - intersection
    iou = (intersection + smooth) / ( union + smooth)
    return iou


# In[6]:


inp = layers.Input(shape=(480,640,3),name='input_image')
#[TO:DO] check why (28,28,1) instead of (28,28)
hl1 = layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu',name='conv3x3_hl1')(inp)
hl2 = layers.BatchNormalization(name='BatchNormalization_hl2') (hl1)
hl3 = layers.MaxPooling2D(pool_size=(4, 4), strides=2,name = 'MaxPooling2D_hl3') (hl2)
hl4 = layers.Conv2D(filters=40,kernel_size=(3,3),activation='relu',name='conv3x3_hl4')(hl3)
hl5 = layers.BatchNormalization(name='BatchNormalization_hl5') (hl4)
hl6 = layers.MaxPooling2D(pool_size=(4, 4), strides=2,name = 'MaxPooling2D_hl6') (hl5)
hl7 = layers.Conv2D(filters=64,kernel_size=(3,3),activation='relu',name='conv3x3_hl7')(hl6)
hl8 = layers.BatchNormalization(name='BatchNormalization_hl8') (hl7)
hl9 = layers.MaxPooling2D(pool_size=(4, 4), strides=2,name = 'MaxPooling2D_hl9') (hl8)
hl10 = layers.Conv2D(filters=64,kernel_size=(3,3),activation='relu',name='conv3x3_hl10')(hl9)
hl11 = layers.BatchNormalization(name='BatchNormalization_hl11') (hl10)
out = layers.MaxPooling2D(pool_size=(4, 4), strides=2,name = 'MaxPooling2D_hl12') (hl11)
out=layers.Flatten() (out)


# In[7]:


cls1=layers.Dense(256,activation='relu',name='Dense_cls1')(out)
out_cls=layers.Dense(3,activation='softmax',name='output_cls')(cls1)


# In[8]:


bbox1=layers.Dense(256,activation='relu',name='Dense_bbox1')(out)
out_bbox=layers.Dense(4,activation='relu',name='output_bbox')(bbox1)


# In[9]:


model = keras.models.Model([inp],[out_cls,out_bbox])
model.summary()


# In[13]:


model.compile(optimizer='adam',loss={'output_cls':'binary_crossentropy',
                                    'output_bbox':iou_loss_core},metrics=['accuracy'])


# In[ ]:
import random
def image_generator(x_train,class_train,bbox_train, batch_size = 64):
    length=x_train.shape[0]
    print(length)
    # y_train=np.expand_dims(y_train, axis=3)
    while True:
        # Select files (paths/indices) for the batch
        
        batch_input = []
        batch_class = [] 
        batch_bbox = []

        # Read in each input, perform preprocessing and get labels
        for i in range(batch_size):
            i= random.randint(0,length)
            input_1 = x_train[i]
            class_1 = class_train[i]
            bbox_1 = bbox_train[i]
            # input = preprocess_input(image=input)
            batch_input += [ input_1 ]
            batch_class += [ class_1 ]
            batch_bbox += [ bbox_1 ]
        # Return a tuple of (input,output) to feed the network
        batch_x = np.array( batch_input )
        batch_class = np.array( batch_class )
        batch_bbox = np.array( batch_bbox)

        yield( batch_x, [batch_class,batch_bbox] )



tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph2', histogram_freq=0, write_graph=True, write_images=True)

# # In[32]:

x_train=np.load('images.npy')
class_train=np.load('labels.npy')
bbox_train=np.load('bbox.npy')
print("shuffling xtrain")
length=x_train.shape[0]
print("length is : "+str(length))
trainGen=image_generator(x_train,class_train,bbox_train, batch_size = 4)

H = model.fit_generator(trainGen,steps_per_epoch=int(length/4)    ,epochs=4)

# model.fit(x=x_train,
#          y=[class_train,bbox_train],
#          batch_size=300,
#          epochs=2,
#          validation_split=0.1,
#          verbose =1,
#          callbacks=[tbCallBack])

model.save('task1.h5')