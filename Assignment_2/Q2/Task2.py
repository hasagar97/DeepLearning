
# coding: utf-8

# In[1]:


import cv2
import numpy as np


# In[2]:


#using a_b_c_d_*
# w=2
# x=2
# y=12
# z=2
iterations=10


# In[3]:


# # X=[]
# Y=[]
# Z=[]
# color=[]
# length=[]
# angle=[]
# thick=[]
# for i in range(1,1001):
#     cnt=0
# #     print(i)
#     for a in range(2):
#         for b in range(2):
#             for c in range(12):
#                 for d in range(2):
# #                     img =cv2.imread('Images/'+str(a)+'_'+str(b)+'_'+str(c)+'_'+str(d)+'_'+str(i)+'.jpg')
# #                     X.append(img)
#                     Y.append(cnt)
#                     color.append(a)
#                     length.append(b)
#                     angle.append(c)
#                     thick.append(d)
#                     Z.append('Images/'+str(a)+'_'+str(b)+'_'+str(c)+'_'+str(d)+'_'+str(i)+'.jpg')
#                     cnt+=1
                    
                    


# In[4]:


# X=np.asarray(X)
# Y=np.asarray(Y)


# In[5]:


# color=np.asarray(color)
# length=np.asarray(length)
# angle=np.asarray(angle)
# thick=np.asarray(thick)
# print(color.shape,length.shape,angle.shape,thick.shape)


# In[6]:


# np.save('data/color',color)
# np.save('data/length',length)
# np.save('data/angle',angle)
# np.save('data/thick',thick)


# In[7]:


# print(X.shape,Y.shape)


# In[8]:


X=np.load('data/images.npy')
Y=np.load('data/labels.npy')
color=np.load('data/color.npy')
length=np.load('data/length.npy')
angle=np.load('data/angle.npy')
thick=np.load('data/thick.npy')


# In[9]:


import keras
from keras import layers


# In[10]:


Y = keras.utils.to_categorical(Y, 96)
# color =  keras.utils.to_categorical(color,2)
# length =  keras.utils.to_categorical(length,2)
angle =  keras.utils.to_categorical(angle,12)
# thick = keras.utils.to_categorical(thick,2)


# In[11]:


print(X.shape,Y.shape,color.shape,length.shape,angle.shape,thick.shape)


# In[12]:


x_train= X[:72000]
x_test=X[72000:]
y_train=Y[:72000]
y_test=Y[72000:]


# In[13]:


color_train=color[:72000]
color_test=color[72000:]

length_train=length[:72000]
length_test=length[72000:]

angle_train=angle[:72000]
angle_test=angle[72000:]

thick_train=thick[:72000]
thick_test=thick[72000:]


# In[14]:


print(x_train.shape,x_test.shape)
print(y_train.shape,y_test.shape)


# In[15]:


color[1].shape


# In[16]:


import matplotlib.pyplot as plt


# In[17]:



plt.imshow(x_train[58354])


# In[18]:


print(Y[58354])
# print(Z[58354])

print(color[58354],length[58354],angle[58354],thick[58354])


# In[19]:


#Creating Model
inp = layers.Input(shape=(28,28,3),name='input_image')
#[TO:DO] check why (28,28,1) instead of (28,28)
hl1 = layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu',name='conv3x3_hl1')(inp)
hl2 = layers.BatchNormalization(name='BatchNormalization_hl2') (hl1)
hl3 = layers.MaxPooling2D(pool_size=(2, 2), strides=2,name = 'MaxPooling2D_hl3') (hl2)
hl4 = layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu',name='conv3x3_hl4')(hl3)
hl5 = layers.BatchNormalization(name='BatchNormalization_hl5') (hl4)
out = layers.Flatten() (hl5)
# model = keras.models.Model(inp,out) 
# model.summary()


# In[20]:


#for color
cl1=layers.Dense(1024,activation='relu',name='Dense_cl1')(out)
outcl=layers.Dense(1,activation='sigmoid',name='output_cl')(cl1)


# In[21]:


ln1=layers.Dense(1024,activation='relu',name='Dense_ln1')(out)
outln=layers.Dense(1,activation='sigmoid',name='output_ln')(ln1)


# In[22]:


ang1=layers.Dense(1024,activation='relu',name='Dense_ang1')(out)
outang=layers.Dense(12,activation='softmax',name='output_ang')(ang1)


# In[23]:


th1=layers.Dense(1024,activation='relu',name='Dense_th1')(out)
outth=layers.Dense(1,activation='sigmoid',name='output_th')(th1)


# In[29]:


model = keras.models.Model([inp],[outcl,outln,outang,outth])
model.summary()


# In[30]:


model.compile(optimizer='adam',loss={'output_cl':'binary_crossentropy',
                                    'output_ln':'binary_crossentropy',
                                     'output_ang':'categorical_crossentropy',
                                     'output_th':'binary_crossentropy'
                                    },metrics=['accuracy'])


# In[31]:


# checkpointer = keras.callbacks.ModelCheckpoint(filepath='model.weights.best.hdf5', verbose = 1, save_best_only=True)
tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph2', histogram_freq=0, write_graph=True, write_images=True)


# In[32]:


model.fit(x=x_train,
         y=[color_train,length_train,angle_train,thick_train],
         batch_size=300,
         epochs=2,
         validation_split=0.1,
         verbose =1,
         callbacks=[tbCallBack])


# In[33]:


score = model.evaluate(x_test, [color_test,length_test,angle_test,thick_test ])


# In[34]:


print(score)


# In[35]:


# model.save('CorrectActivations.h5')


# In[36]:


# from sklearn.metrics import f1_score,confusion_matrix,precision_score
# import matplotlib.mlab as mlab


# In[45]:


y_pred = model.predict(x_test)

model.save('task2.h5')
# In[59]:


# y_angle = y_pred[2].argmax(axis=-1)
# Y=np.load('data/labels.npy')
# y_test=Y[72000:]


# # In[65]:


# angle=np.load('data/angle.npy')
# angle_train=angle[:72000]
# angle_test=angle[72000:]



# confusion_mat=confusion_matrix(angle_test.tolist(),y_angle.tolist()) 



# # f1_score=f1_score(angle_test,angle_test)


# # In[44]:


# print(f1_score)

