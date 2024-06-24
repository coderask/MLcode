import tensorflow as tf 

#model intialization 
pit_stop_model = tf.keras.models.Sequential()
#hidden layers
pit_stop_model.add(tf.keras.layers.Dense(64, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.0005) ))
pit_stop_model.add(tf.keras.layers.Dense(64, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.0005) ))
pit_stop_model.add(tf.keras.layers.Dense(64, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.0005) ))
#output layer
pit_stop_model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

#model compilation
pit_stop_model.compile(optimizer=tf.keras.optimizers.Nadam(), loss='binary_crossentropy', metrics=['accuracy'])
#model training
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
pit_stop_model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test), callbacks=[stop_early])

