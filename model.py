import matplotlib.pyplot as plt
import tensorflow as tf


def create_model(input_length, learning_rate, model):
    input_shape = (input_length, 129)

    inputs = tf.keras.Input(input_shape)

    if model == "model1":
        x = tf.keras.layers.LSTM(512)(inputs)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(3, activation='linear')(x)

    if model == "model3":

        x = tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu')(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LSTM(units=512, return_sequences=True)(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LSTM(units=512)(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(units=256, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(3, activation='linear')(x)

        # x = tf.keras.layers.LSTM(512, return_sequences=True)(inputs)
        # x = tf.keras.layers.Dropout(0.3)(x)
        # x = tf.keras.layers.BatchNormalization()(x)
        # x = tf.keras.layers.LSTM(512, return_sequences=True)(x)
        # x = tf.keras.layers.Dropout(0.3)(x)
        # x = tf.keras.layers.BatchNormalization()(x)
        # x = tf.keras.layers.LSTM(512)(x)
        # x = tf.keras.layers.Dropout(0.3)(x)
        # x = tf.keras.layers.BatchNormalization()(x)
        # x = tf.keras.layers.Dense(256, activation='relu')(x)
        # x = tf.keras.layers.Dropout(0.3)(x)
        # x = tf.keras.layers.BatchNormalization()(x)
        # x = tf.keras.layers.Dense(3, activation='linear')(x)


    outputs = tf.keras.layers.Dense(129, activation="softmax", name='piano_roll')(x)


    model = tf.keras.Model(inputs, outputs)

    loss = tf.keras.losses.BinaryCrossentropy()

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)


    model.compile(
    loss=loss,
    optimizer=optimizer,
    )

    return model, loss, optimizer

def create_model_sequence(input_length, learning_rate):
    input_shape = (input_length, 129)

    inputs = tf.keras.Input(input_shape)

    if model == "model2":
        x = tf.keras.layers.LSTM(512)(inputs)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)

    if model == "model4":
        x = tf.keras.layers.LSTM(512, return_sequences=True)(inputs)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LSTM(512, return_sequences=True)(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LSTM(512, return_sequences=True)(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(3, activation='linear')(x)

    outputs = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(129, activation="softmax", name='piano_roll'))(x)

    model = tf.keras.Model(inputs, outputs)

    loss = tf.keras.losses.BinaryCrossentropy()

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)


    model.compile(
    loss=loss,
    optimizer=optimizer,
    )

    return model, loss, optimizer

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
        patience=5,
        verbose=1,
        restore_best_weights=True),
    ]
    print(len(list(val_ds)))

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

