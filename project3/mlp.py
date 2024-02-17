"""
Vellios Georgios Serafeim AEM:9471
"""
import tensorflow.keras as keras
from keras.layers import Flatten, Dense, Dropout
from keras import initializers, regularizers
import time
import matplotlib.pyplot as plt


def create_model():

	# model = keras.models.Sequential([
	# 	Flatten(input_shape=(28, 28)),
	# 	Dense(128, activation='relu'),
	# 	Dense(256, activation='relu'),
	# 	Dense(10, activation='softmax')
	# ])

	# model = keras.models.Sequential([
	# 	Flatten(input_shape=(28,28)),
	# 	Dense(128, kernel_initializer=initializers.RandomNormal(mean=10) , activation='relu'),
	# 	Dense(256, kernel_initializer=initializers.RandomNormal(mean=10) , activation='relu'),
	# 	Dense(10, activation='softmax')
	# ])
	#
	# l2val=0.1
	# l2val=0.01
	# l2val=0.001
	# model = keras.models.Sequential([
	# 	Flatten(input_shape=(28,28)),
	# 	Dense(128, kernel_initializer=initializers.RandomNormal(mean=10) , kernel_regularizer=regularizers.l2(l2=l2val),activation='relu'),
	# 	Dense(256, kernel_initializer=initializers.RandomNormal(mean=10) ,kernel_regularizer=regularizers.l2(l2=l2val), activation='relu'),
	# 	Dense(10, activation='softmax')
	# ])

	model = keras.models.Sequential([
		Flatten(input_shape=(28,28)),
		Dense(128, kernel_regularizer=regularizers.l1(l1=0.01), activation='relu'),
		Dropout(0.3),
		Dense(256, kernel_regularizer=regularizers.l1(l1=0.01), activation='relu'),
		Dropout(0.3),
		Dense(10, activation='softmax')
	])

	# opt=keras.optimizers.RMSprop(
	#     learning_rate=0.001,
	#     rho=0.99,
	# )
	opt = keras.optimizers.SGD(learning_rate=0.01)
	model.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer=opt)

	return model


def train_model_1(X_train, y_train, batch_size=256, epochs=100):

	model = create_model()

	start_time = time.time()
	history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)

	print("---Training time is %s seconds ---" % (time.time() - start_time))

	plt.figure(1)
	plt.title("Καμπύλες Ακριβείας")
	plt.xlabel('Epochs')
	plt.ylabel("Accuracy")
	a, = plt.plot(history.history['accuracy'])
	b, = plt.plot(history.history['val_accuracy'])
	plt.legend([a,b], ['Training', 'Validation'])

	plt.figure(2)
	plt.title("Καμπύλες Κόστους")
	plt.xlabel('Epochs')
	plt.ylabel("Loss")
	a, = plt.plot(history.history['loss'])
	b, = plt.plot(history.history['val_loss'])
	plt.legend([a,b], ['Training', 'Validation'])

	plt.show()
	return