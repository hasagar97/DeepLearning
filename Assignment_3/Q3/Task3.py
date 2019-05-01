
# coding: utf-8

# In[1]:


#import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"    


# In[2]:


import cv2
import numpy as np
#import matplotlib.pyplot as plt


# In[3]:
print("importing DL libraries")

import tensorflow.keras as keras
from tensorflow.keras import layers


# In[4]:

print("reading Daata")
x_train=np.load("imgs.npy")
y_train = np.load("labels.npy")
print("done reading")





# In[5]:


# x_train=X[:9000]
# x_test=X[9000:]
# y_train=Y[:9000]
# y_test=Y[9000:]
# x_train = x_train.astype('float32') / 255
# x_test = x_test.astype('float32') / 255

# y_train = y_train.astype('float32') / 255
# y_test = y_test.astype('float32') / 255

print(x_train.shape,y_train.shape)
print(x_test.shape,y_test.shape)


def image_generator(x_train,y_train, batch_size = 64):
	count= -64
	y_train=np.expand_dims(y_train, axis=3)
	while True:
		# Select files (paths/indices) for the batch
		count+=64
		if(count>=9000-64):
			count=0
		batch_input = []
		batch_output = [] 

		# Read in each input, perform preprocessing and get labels
		for i in range(batch_size):
			input = x_train[i+count]
			output = y_train[i+count]

			# input = preprocess_input(image=input)
			batch_input += [ input ]
			batch_output += [ output ]
		# Return a tuple of (input,output) to feed the network
		batch_x = np.array( batch_input )
		batch_y = np.array( batch_output )

		yield( batch_x, batch_y )

# In[7]:


#plt.imshow(x_train[2000])


# In[8]:


#X


# In[9]:


#plt.imshow(y_train[2000])


# In[10]:


#Defining Model


inp = layers.Input(shape=(640,320,3),name='input_image')
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
hl12 = layers.MaxPooling2D(pool_size=(4, 4), strides=2,name = 'MaxPooling2D_hl12') (hl11)





convNet = layers.Flatten() (hl12)
dl1=layers.Dense(256,activation='relu',name='Dense_1')(convNet)
out=layers.Dense(2, activation='relu', name='Corepoint')(dl1)
# In[7]:

# In[11]:


model=keras.models.Model(inp,out)


# In[12]:


model.summary()


# In[13]:


model.compile(loss='binary_crossentropy',
      optimizer= 'Adam' ,
      metrics=['accuracy'])


# In[14]:
print("compiled model")

tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph2', histogram_freq=0, write_graph=True, write_images=True)


# In[15]:


yy=np.expand_dims(y_train, axis=3)#(y_train,(9000,300,400,1))


# In[16]:


yy.shape


# In[17]:

x_train.shape
y_train.shape


# In[18]:
trainGen=image_generator(x_train,y_train, batch_size = 16)
H = model.fit_generator(trainGen,steps_per_epoch=int(9000/16)	,epochs=2)
# model.fit(x=x_train,
         # y=yy,
         # batch_size=300,
         # epochs=1,
         # validation_split=0.1,
         # verbose =1,
         # callbacks=[tbCallBack])


# In[19]:


# abc=np.zeros((28,28))
# abc.shape
# abc=np.reshape(abc,(28,28,1))


# # In[20]:


# abc.shape

# yyy=np.expand_dims(y_test, axis=3)#(y_train,(9000,300,400,1))


# # In[16]:


# print(yyy.shape)
# score = model.evaluate(x_test,yyy )


# # In[34]:


# print(score)


# # In[35]:


# # model.save('CorrectActivations.h5')


# # In[36]:


# # from sklearn.metrics import f1_score,confusion_matrix,precision_score
# # import matplotlib.mlab as mlab


# # In[45]:


# #y_pred = model.predict(x_test)

# model.save('task2_big_2epochs.h5')


# y_pred = model.predict(x_test)


# for i in range(100):
# 	cv2.imwrite('output/'+str(i)+'_pred.png',y_pred[i]*255)
# 	cv2.imwrite('output/'+str(i)+'_inp.png',x_test[i]*255)
# 	cv2.imwrite('output/'+str(i)+'_gt.png',y_test[i]*255)


