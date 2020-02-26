from deep_models.utils import load_cifar10_data, inception_module,kernel_init,bias_init
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  Dense,Conv2D,Flatten,Dropout,MaxPooling2D
import tensorflow as tf

def alexnet_model():
	model = Sequential()

	model.add(Conv2D(filters=96,input_shape=(224,224,3),kernel_size = (11,11),strides=(4,4),padding='valid',activation='relu'))

	model.add(MaxPooling2D(pool_size= (2,2),strides=(2,2),padding='valid'))


	model.add(Conv2D(filters=256,kernel_size=(11,11),strides=(1,1),padding='valid',activation='relu'))

	model.add(MaxPooling2D(pool_size= (2,2),strides=(2,2),padding='valid'))

	model.add(Conv2D(filters=384,kernel_size=(3,3),strides=(1,1),padding='valid',activation='relu'))

	model.add(Conv2D(filters=256,kernel_size=(3,3),strides=(1,1),padding='valid',activation='relu'))

	model.add(MaxPooling2D(pool_size= (2,2),strides=(2,2),padding='valid'))

	model.add(Flatten())

	model.add(Dense(4096,activation='relu'))

	model.add(Dropout(0.4))
	
	model.add(Dense(1000,activation='relu'))

	model.add(Dropout(0.4))

	model.add(Dense(10,activation='relu'))

	model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

	model.summary()

	with tf.Session() as sess:
		(X_train,y_train),(X_test,y_test) = load_cifar10_data(224,224)
		
		# sess.run(model.fit(X_train,y_train,
		# 			validation_data=(X_test,y_test),epochs=5,batch_size=256))

