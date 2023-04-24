import tensorflow as tf
import matplotlib.pyplot as plt


def create_model(input_length, learning_rate, optimizer, model):
    input_shape = (input_length, 3)
    

    inputs = tf.keras.Input(input_shape)

    if model == "model1":
        x = tf.keras.layers.LSTM(512)(inputs)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)

    if model == "model3":
        x = tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu')(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LSTM(512, return_sequences=True)(inputs)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LSTM(512)(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.BatchNormalization()(x)


    outputs = tf.keras.layers.Dense(3, activation="softmax", name="Event")(x)
    outputs = tf.keras.layers.Reshape((1, 3))(outputs)



    model = tf.keras.Model(inputs, outputs)

    loss = tf.keras.losses.CategoricalCrossentropy()

    if optimizer == "RMS":
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    elif optimizer == "Adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        print("Found no optimizer called", optimizer)
        exit()
    
    model.compile(
        loss=loss,
        #make loss the weighted sum since pitch loss is much higher

        optimizer=optimizer,
        )

    return model, loss, optimizer

def create_model_sequence(input_length, learning_rate, optimizer, model):
    input_shape = (input_length, 3)

    inputs = tf.keras.Input(input_shape)

    if model == "model2":
        x = tf.keras.layers.LSTM(512)(inputs)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)

    if model == "model4":
        x = tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu')(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LSTM(512, return_sequences=True)(inputs)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LSTM(512)(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.BatchNormalization()(x)

    x = tf.expand_dims(x, axis=1) 
    outputs = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(3, activation="softmax"))(x)

    model = tf.keras.Model(inputs, outputs)

    loss = tf.keras.losses.MeanSquaredError()

    if optimizer == "RMS":
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    elif optimizer == "Adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        print("Found no optimizer called", optimizer)
        exit()

    model.compile(
    loss=loss,
    optimizer=optimizer,
    )

    return model, loss, optimizer

def mse_with_positive_pressure(y_true: tf.Tensor, y_pred: tf.Tensor):
  mse = (y_true - y_pred) ** 2
  positive_pressure = 10 * tf.maximum(-y_pred, 0.0)
  return tf.reduce_mean(mse + positive_pressure)

def train_model(model, train_ds, val_ds, save_model_path, epochs):

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath='./training_checkpoints/ckpt_{epoch}',
            save_weights_only=False,
            monitor='val_loss',
            mode='max',
            save_best_only=True),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            min_delta=0,
            patience=125,
            verbose=1,
            restore_best_weights=True),
    ]

    history = model.fit(
        train_ds,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=val_ds,
    )

    model.save_weights(save_model_path)

    plt.plot(history.epoch, history.history['loss'], label='total training loss')
    plt.savefig(save_model_path+'training_loss.png')
    plt.figure()
    plt.plot(history.epoch, history.history['val_loss'], label='total val loss')
    plt.savefig(save_model_path+'validation_loss.png') 