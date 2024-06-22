import tensorflow as tf
import tensorflow_datasets as tfds
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
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28,28)), tf.keras.layers.Dense(128, activation='relu'), tf.keras.layers.Dense(10)])
#compile
model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True), metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],)
#training
model.fit(ds_train,epochs=6,validation_data=ds_test,)



