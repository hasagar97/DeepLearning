
# coding: utf-8

# In[2]:


import cv2
import numpy as np


# In[3]:


#using a_b_c_d_*
# w=2
# x=2
# y=12
# z=2
iterations=3


# In[4]:


# X=[]
# Y=[]
Z=[]
for i in range(1,1001):
    cnt=0
#     print(i)
    for a in range(2):
        for b in range(2):
            for c in range(12):
                for d in range(2):
#                     img =cv2.imread('Images/'+str(a)+'_'+str(b)+'_'+str(c)+'_'+str(d)+'_'+str(i)+'.jpg')
#                     X.append(img)
#                     Y.append(cnt)
                    Z.append('Images/'+str(a)+'_'+str(b)+'_'+str(c)+'_'+str(d)+'_'+str(i)+'.jpg')
                    cnt+=1
                    
                    


# In[5]:


# X=np.asarray(X)
# Y=np.asarray(Y)


# In[6]:


# print(X.shape,Y.shape)


# In[7]:

#Images have been preloaded and saved into .npy files for faster loading
X=np.load('data/images.npy')
Y=np.load('data/labels.npy')


# In[8]:


import keras
from keras import layers


# In[9]:


Y = keras.utils.to_categorical(Y, 96)


# In[10]:


print(X.shape,Y.shape)


# In[11]:


x_train= X[:72000]
x_test=X[72000:]
y_train=Y[:72000]
y_test=Y[72000:]


# In[12]:


print(x_train.shape,x_test.shape)
print(y_train.shape,y_test.shape)


# In[13]:


import matplotlib.pyplot as plt


# In[14]:



plt.imshow(x_train[58354])


# In[15]:


print(Y[58354])
print(Z[58354])


# In[16]:


#Creating Model
inp = layers.Input(shape=(28,28,3),name='input_image')
#[TO:DO] check why (28,28,1) instead of (28,28)
hl1 = layers.Conv2D(filters=50,kernel_size=(7,7),activation='relu',name='conv7x7_hl1')(inp)
hl2 = layers.BatchNormalization(name='BatchNormalization_hl2') (hl1)
hl3 = layers.MaxPooling2D(pool_size=(2, 2), strides=2,name = 'MaxPooling2D_hl3') (hl2)
flat = layers.Flatten() (hl3)
hl4 = layers.Dense(1024,activation='relu', name='Dense_hl4') (flat)
out = layers.Dense(96,activation='softmax',name='output_layer') (hl4)
model = keras.models.Model(inp,out) 
model.summary()


# In[17]:


model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])


# In[18]:


# checkpointer = keras.callbacks.ModelCheckpoint(filepath='task1b_hrishi_model.weights.best.hdf5', verbose = 1, save_best_only=True)
tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph1b', histogram_freq=0, write_graph=True, write_images=True)


# In[ ]:


model.fit(x=x_train,
         y=y_train,
         batch_size=300,
         epochs=iterations,
         validation_split=0.1,
         verbose =1,
         callbacks=[tbCallBack] )


# In[ ]:


score = model.evaluate(x_test, y_test)


# In[ ]:


print(score)

model.save('task1b_hrishi_trained_model'+str(iterations)+'.h5')