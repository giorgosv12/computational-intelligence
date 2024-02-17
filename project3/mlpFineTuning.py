"""
Vellios Georgios Serafeim AEM:9471
"""
import tensorflow.keras as keras
import tensorflow as tf
from keras.layers import Flatten, Dense, Dropout
from keras import initializers, regularizers
import keras.backend as K
from keras_tuner import RandomSearch
import keras_tuner
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

def f1(y_true, y_pred):
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = tp / (predicted_positives + K.epsilon())
    recall = tp / (positives + K.epsilon())
    f1 = 2 * (precision*recall)/(precision+recall+K.epsilon())
    return f1


def fine_tune_model2(X_train, y_train):

	tuner = RandomSearch(
		test_model,
		objective= keras_tuner.Objective('val_f1', direction='max'),
		max_trials=200,
		executions_per_trial=2,
		overwrite=True
	)
	tuner.search_space_summary()
	callback = tf.keras.callbacks.EarlyStopping(monitor='val_f1', patience=200, restore_best_weights=False)
	tuner.search(X_train, y_train, epochs=1000, batch_size=256, callbacks=[callback],
				 validation_split=0.2, verbose=True)
	tuner.results_summary()


def test_model(hp):

	units1 = hp.Choice("Units_1", values=[64, 128])
	units2 = hp.Choice("Units_2", values=[256, 512])
	l2_val = hp.Choice("l2", values=[0.1, 0.001, 0.000001])
	lr = hp.Choice("Learning Rate", values=[0.1, 0.01, 0.001])

	model = keras.models.Sequential([
		Flatten(input_shape=(28, 28)),
		Dense(units=units1, kernel_initializer=initializers.HeNormal(), kernel_regularizer=regularizers.l2(l2=l2_val), activation='relu'),
		Dense(units=units2, kernel_initializer=initializers.HeNormal(), kernel_regularizer=regularizers.l2(l2=l2_val), activation='relu'),
		Dense(10, activation='softmax')
	])

	opt = keras.optimizers.RMSprop(learning_rate=lr)
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=[f1])
	return model

def create_model():
	units1 = 128
	units2 = 256
	l2_val = 1e-06
	lr = 0.001

	model = keras.models.Sequential([
		Flatten(input_shape=(28, 28)),
		Dense(units=units1, kernel_initializer=initializers.HeNormal(), kernel_regularizer=regularizers.l2(l2=l2_val),
			  activation='relu'),
		Dense(units=units2, kernel_initializer=initializers.HeNormal(), kernel_regularizer=regularizers.l2(l2=l2_val),
			  activation='relu'),
		Dense(10, activation='softmax')
	])

	opt = keras.optimizers.RMSprop(learning_rate=lr)
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=[f1])
	return model


def train_model2(X_train, y_train, X_test, y_test):
	model = create_model()

	callback = tf.keras.callbacks.EarlyStopping(monitor='val_f1', patience=200, restore_best_weights=False)
	history = model.fit(X_train, y_train, batch_size=256, epochs=1000,callbacks=[callback], validation_split=0.2)

	rounded_y_test = np.argmax(y_test, axis=1)
	y_pred = np.argmax(model.predict(X_test) ,axis=1)

	cm = confusion_matrix(rounded_y_test, y_pred)
	print('Confusion Matrix:\n', cm)

	recall = np.diag(cm) / np.sum(cm, axis=1)
	precision = np.diag(cm) / np.sum(cm, axis=0)
	recall = np.mean(recall)
	precision=np.mean(precision)
	F1 = 2 * (precision * recall) / (precision + recall)

	print("\n Accuracy: {}\n Precision: {}\n Recall: {}\n f1: {}".format(np.sum(np.diag(cm))/10000, precision, recall, F1))

	plt.figure(1)
	plt.title("Καμπύλες f1")
	plt.xlabel('Epochs')
	plt.ylabel("Accuracy")
	a, = plt.plot(history.history['f1'])
	b, = plt.plot(history.history['val_f1'])
	plt.legend([a, b], ['Training', 'Validation'])

	plt.figure(2)
	plt.title("Καμπύλες Κόστους")
	plt.xlabel('Epochs')
	plt.ylabel("Loss")
	a, = plt.plot(history.history['loss'])
	b, = plt.plot(history.history['val_loss'])
	plt.legend([a, b], ['Training', 'Validation'])

	plt.show()
	return