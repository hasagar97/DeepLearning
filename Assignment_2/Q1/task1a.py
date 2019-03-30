import numpy as np

import keras
from keras import layers

iterations= 5

(x_train,y_train) , (x_test,y_test) = keras.datasets.mnist.load_data()

print(x_train.shape,x_test.shape)
print(y_train.shape,y_test.shape)

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

inp = layers.Input(shape=(28,28,1),name='input_image')
#[TO:DO] check wehy (28,28,1) instead of (28,28)
hl1 = layers.Conv2D(filters=32,kernel_size=2,activation='relu',name='conv7x7_hl1')(inp)
hl2 = layers.BatchNormalization(name='BatchNormalization_hl2') (hl1)
hl3 = layers.MaxPooling2D(pool_size=(2, 2), strides=2,name = 'MaxPooling2D_hl3') (hl2)
flat = layers.Flatten() (hl3)
hl4 = layers.Dense(1024,activation='relu', name='Dense_hl4') (flat)
out = layers.Dense(10,activation='softmax',name='output_layer') (hl4)
model = keras.models.Model(inp,out) 
model.summary()


model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])




x_train = np.reshape(x_train,(x_train.shape[0],28,28,1))
x_test = np.reshape(x_test,(10000,28,28,1))


# One-hot encode the labels
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# from keras.callbacks import Mod.0elCheckpoint

# checkpointer = keras.callbacks.ModelCheckpoint(filepath='task1a_model.weights.best.hdf5', verbose = 1, save_best_only=True)
tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph1a', histogram_freq=0, write_graph=True, write_images=True)

print(x_train.shape,x_test.shape)
print(y_train.shape,y_test.shape)
model.fit(x=x_train,
         y=y_train,
         batch_size=128,
         epochs=iterations,
         validation_split=0.05,
         verbose=1,
         callbacks=[tbCallBack] )


score = model.evaluate(x_test, y_test)



print(score)


model.save('task1a_trained_model'+str(iterations)+'.h5')
