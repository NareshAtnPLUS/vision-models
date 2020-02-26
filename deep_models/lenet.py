from deep_models.utils import load_cifar10_data,kernel_init,bias_init
from tensorflow.keras.layers import Input,MaxPool2D,Conv2D,Dense,Flatten
from tensorflow.keras.models import Model,Sequential
import tensorflow as tf
def lenet_model():

	model = Sequential()

	model.add(Input(shape=(28,28,3)))

	model.add(Conv2D(filters=6,kernel_size=(5,5),padding='same',activation='relu',
		name='conv1_5x5',kernel_initializer=kernel_init,bias_initializer=bias_init))

	model.add(MaxPool2D((2,2), padding = 'same',
		strides=(2,2),name = 'maxpool1_2x2_2s'))


	model.add(Conv2D(filters=16,kernel_size=(5,5),padding='same',activation='relu',
		name='conv2_5x5',kernel_initializer=kernel_init,bias_initializer=bias_init))

	model.add(MaxPool2D((2,2), padding = 'same',
		strides=(2,2),name = 'maxpool2_2x2_2s'))


	model.add(Flatten())

	model.add(Dense(400,activation='relu',name='fc1_400'))

	model.add(Dense(120,activation='relu',name='fc1_120'))

	model.add(Dense(84,activation='relu',name = 'fc1_84'))

	model.add(Dense(10,activation='softmax',name='output'))


	

	model.summary()

	model.compile(loss=['categorical_crossentropy'],optimizer='adam',metrics=['accuracy'])

	with tf.Session() as sess:
		(X_train,y_train),(X_test,y_test) = load_cifar10_data(28,28)
		sess.run(model.fit(X_train,y_train,
					validation_data=(X_test,y_test),epochs=5,batch_size=10000))

