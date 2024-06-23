import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import numpy as np
'''
print("Hello World")
#loading the dataset 
#load the mnist(image dataset from google)
#load the training and the testing parts of the dataset
#all files in the dataset are shuffled(improves model learning order-based patterns)
#as_supervised returns dataset as a tuple, tuple of (input, label) 
#with_info returns a information object about the dataset which is ds_info

(ds_train, ds_test), ds_info = tfds.load('mnist', split=['train','test'], shuffle_files=True, as_supervised = True, with_info=True)

#data setup
#scaling the pixel values from 0 to 255 to floats (0-1)
def normalize_img(image, label):
    return tf.cast(image, tf.float32) / 255., label

#setting up all of the training data
#map(apply transformation) to each of the datasets the transformations: normalizing images and doing it in parallel
ds_train = ds_train.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
#store the training data in the ram 
ds_train = ds_train.cache()
#shuffle the trianing dataset with a buffer size of all the number of training examples
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
#split training data into 128 chunksthat help train
ds_train = ds_train.batch(128)
#optimize the amount of time it takes to load data by letting tensorflow automatically choose to load data
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

#testing data
#same as for training but not shuffled and caching is done after batching 
ds_test = ds_test.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)
#Model
#intialize
'''
#old model
'''
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28,28)), tf.keras.layers.Dense(128, activation='relu'), tf.keras.layers.Dense(10)])
#for tyre 
model = tf.keras.models.Sequential([tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001), input_shape=(input_features.shape[1],)), tf.keras.layers.Dense(3, activation='softmax')])

#compile
model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True), metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],)
#training
model.fit(ds_train,epochs=6,validation_data=ds_test,)
'''
#for tyre

#mock inputs 
race_progress = np.array([0.5])  # Numerical
remaining_pit_stops = np.array([1])  # Categorical
relative_compound = np.array([0, 1, 0])  # One-hot encoded categorical {soft, medium, hard}
race_track = np.array([1, 0, 0, 0])  # One-hot encoded categorical {Austin, Baku, ..., Yas Marina}
fulfilled_second_compound = np.array([1])  # Categorical
number_of_avail_compounds = np.array([1, 0])  # One-hot encoded categorical {2, 3}

#concanate the inputs
input_features = np.concatenate((race_progress, remaining_pit_stops, relative_compound, race_track, fulfilled_second_compound, number_of_avail_compounds), axis=None)
#transform the inputs into a 2d array
input_features= input_features.reshape(1, -1)

#Model 
#initialization
model = tf.keras.models.Sequential([tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001), input_shape=(input_features.shape[1],)), tf.keras.layers.Dense(3, activation='softmax')])

#compile
model.compile(optimizer=tf.keras.optimizers.Nadam(), loss=SparseCategoricalCrossentropy(from_logits = True), metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],)

#training
labels = np.ones_like(input_features)
model.fit(input_features, labels, epochs=6, validation_data=(input_features, labels))