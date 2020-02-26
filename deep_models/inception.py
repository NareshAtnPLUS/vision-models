from deep_models.utils import load_cifar10_data, inception_module,kernel_init,bias_init
from tensorflow.keras.layers import (Input,MaxPool2D,Conv2D,AveragePooling2D,
								GlobalAveragePooling2D,Flatten,Dense,Dropout)

from tensorflow.keras.models import Model

def inception_model():

	input_layer = Input(shape=(224,224,3))

	x = Conv2D(filters = 64, kernel_size = (7,7),
		padding= 'same',strides=(2,2),activation = 'relu',
		name = 'conv1_7x7_2s',kernel_initializer = kernel_init,
		bias_initializer = bias_init)(input_layer)

	x = MaxPool2D((3,3), padding = 'same',
		strides=(2,2),name = 'maxpool_3x3_2s')(x)

	x = Conv2D(filters = 64,kernel_size = (1,1),
		padding = 'same',strides=(1,1),activation = 'relu',
		name = 'conv2a_1x1_1s',kernel_initializer=kernel_init,
		bias_initializer = bias_init)(x)

	x = Conv2D(filters = 192,kernel_size = (3,3),
		padding = 'same',strides=(1,1),activation = 'relu',
		name = 'conv2b_3x3_1s',kernel_initializer=kernel_init,
		bias_initializer = bias_init)(x)

	x = MaxPool2D((3,3),padding = 'same',strides = (2,2),
		name = 'maxpool2_3x3_2s')(x)


	x = inception_module(x,
		filters_1x1=64,
		filters_3x3_a=96,
		filters_3x3_b=128,
		filters_5x5_a=16,
		filters_5x5_b=32,
		filters_pool=32,
		name='inception_3a')

	x = inception_module(x,
		filters_1x1=128,
		filters_3x3_a=128,
		filters_3x3_b=192,
		filters_5x5_a=32,
		filters_5x5_b=96,
		filters_pool=64,
		name='inception_3b')

	x = MaxPool2D((3,3),padding='same',strides=(2,2),name = 'maxpool_3x3_2s_1')(x)

	x = inception_module(x,
		filters_1x1=192,
		filters_3x3_a=96,
		filters_3x3_b=208,
		filters_5x5_a=16,
		filters_5x5_b=48,
		filters_pool=64,
		name='inception_4a')

	x1 = AveragePooling2D((5,5),strides=3)(x)

	x1 = Conv2D(filters=128,kernel_size=(1,1),padding='same',activation='relu')(x1)

	x1 = Flatten()(x1)

	

	x1 = Dropout(0.7)(x1)




	x = inception_module(x,
		filters_1x1=160,
		filters_3x3_a=112,
		filters_3x3_b=224,
		filters_5x5_a=24,
		filters_5x5_b=64,
		filters_pool=64,
		name='inception_4b')

	x = inception_module(x,
		filters_1x1=128,
		filters_3x3_a=128,
		filters_3x3_b=256,
		filters_5x5_a=24,
		filters_5x5_b=64,
		filters_pool=64,
		name='inception_4c')

	x = inception_module(x,
		filters_1x1=160,
		filters_3x3_a=142,
		filters_3x3_b=288,
		filters_5x5_a=32,
		filters_5x5_b=64,
		filters_pool=64,
		name='inception_4d')

	x2 = AveragePooling2D((5,5),strides=3)(x)

	x2 = Conv2D(filters=128,kernel_size=(1,1),padding='same',activation='relu')(x2)

	x2 = Flatten()(x2)

	x2 = Dense(1024,activation='relu')(x2)

	x2 = Dropout(0.7)(x2)

	x2 = Dense(10,activation='softmax',name='auxilary_output')(x2)

	x = inception_module(x,
		filters_1x1=256,
		filters_3x3_a=160,
		filters_3x3_b=320,
		filters_5x5_a=32,
		filters_5x5_b=128,
		filters_pool=128,
		name='inception_4e')

	x = MaxPool2D((3,3),padding='same',strides=(2,2),name='maxpool_4_3x3_2s')(x)

	x = inception_module(x,
		filters_1x1=256,
		filters_3x3_a=160,
		filters_3x3_b=320,
		filters_5x5_a=32,
		filters_5x5_b=128,
		filters_pool=128,
		name='inception_5a')

	x = inception_module(x,
		filters_1x1=384,
		filters_3x3_a=192,
		filters_3x3_b=384,
		filters_5x5_a=48,
		filters_5x5_b=128,
		filters_pool=128,
		name='inception_5b')

	x = GlobalAveragePooling2D(name='avg_pool_5_3x3_')(x)

	x = Dropout(0.4)(x)

	x = Dense(10,activation='softmax',name='output')(x)

	model = Model(input_layer,[x],name='inception_v1')


	model.summary()

	model.compile(loss=['categorical_crossentropy'],optimizer='adam',metrics=['accuracy'])
	
	# (X_train,y_train),(X_test,y_test) = load_cifar10_data(224,224)

	# with tf.Session() as sess:
	# 	history = sess.run(model.fit(X_train,[y_train,y_train,y_train],
	# 				validation_data=(X_test,[y_test,y_test,y_test]),epochs=5,batch_size=2))



