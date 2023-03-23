import sys
import pretty_midi
import collections
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


from data_process import *

np.set_printoptions(threshold=sys.maxsize)


BATCH_SIZE = 32
NUM_PREDICTIONS = 120
VALIDATION_SIZE = 0.15
LEARNING_RATE = 0.005
NOISE_SCALE = 1
VOCAB_SIZE = 128
EPOCHS = 2
TEMPERATURE = 1
PROB = 0.3
INPUT_LENGTH = 32
LABEL_LENGTH = 1



def piano_roll_to_pretty_midi(piano_roll, fs=100, program=0):
    '''Convert a Piano Roll array into a PrettyMidi object
     with a single instrument.
    Parameters
    ----------
    piano_roll : np.ndarray, shape=(128,frames), dtype=int
        Piano roll of one instrument
    fs : int
        Sampling frequency of the columns, i.e. each column is spaced apart
        by ``1./fs`` seconds.
    program : int
        The program number of the instrument.
    Returns
    -------
    midi_object : pretty_midi.PrettyMIDI
        A pretty_midi.PrettyMIDI class instance describing
        the piano roll.
    '''
    notes, frames = piano_roll.shape
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program)

    # pad 1 column of zeros so we can acknowledge inital and ending events
    piano_roll = np.pad(piano_roll, [(0, 0), (1, 1)], 'constant')

    # use changes in velocities to find note on / note off events
    velocity_changes = np.nonzero(np.diff(piano_roll).T)

    # keep track on velocities and note on times
    prev_velocities = np.zeros(notes, dtype=int)
    note_on_time = np.zeros(notes)

    for time, note in zip(*velocity_changes):
        # use time + 1 because of padding above
        velocity = piano_roll[note, time + 1]
        time = time / fs
        if velocity > 0:
            if prev_velocities[note] == 0:
                note_on_time[note] = time
                prev_velocities[note] = velocity
        else:
            pm_note = pretty_midi.Note(
                velocity=prev_velocities[note],
                pitch=note,
                start=note_on_time[note],
                end=time)
            instrument.notes.append(pm_note)
            prev_velocities[note] = 0
    pm.instruments.append(instrument)
    return pm


def midi_to_notes(pm) -> pd.DataFrame:
  instrument = pm.instruments[0]
  notes = collections.defaultdict(list)

  
  # Sort the notes by start time
  sorted_notes = sorted(instrument.notes, key=lambda note: note.start)
  
  prev_note = None
  for note in sorted_notes:
    notes["transition"].append(note.pitch - prev_note.pitch) if prev_note else notes["transition"].append(0)
    notes["duration"].append(note.end - note.start)
    notes["pitch"].append(note.pitch)
    prev_note = note

  return pd.DataFrame({name: np.array(value) for name, value in notes.items()})

def split_data(dataset):
  '''
  This function creates and trains a model with all midi files found in the given path.
  The model is saved in the training_checkpoint folder.
  '''
  # dataset = dataset.shuffle(buffer_size=len(list(dataset)))
  
  # Split dataset into training and validation sets
  train_size = int((1-VALIDATION_SIZE) * len(list(dataset)))

  train_dataset = dataset.take(train_size)
  val_dataset = dataset.skip(train_size)

  print(len(list(val_dataset)))
  # # batch the datasets
  val_dataset = val_dataset.batch(BATCH_SIZE, drop_remainder=True).cache().prefetch(tf.data.experimental.AUTOTUNE)
  train_dataset = train_dataset.batch(BATCH_SIZE, drop_remainder=True).cache().prefetch(tf.data.experimental.AUTOTUNE)

  def squeeze_label(x, y):
    return x, tf.squeeze(y)

  train_dataset = train_dataset.map(squeeze_label)
  val_dataset = val_dataset.map(squeeze_label)

  return train_dataset, val_dataset

def mse_with_positive_pressure(y_true: tf.Tensor, y_pred: tf.Tensor):
  mse = (y_true - y_pred) ** 2
  positive_pressure = 10 * tf.maximum(-y_pred, 0.0)
  return tf.reduce_mean(mse + positive_pressure)

def create_model():
  input_shape = (INPUT_LENGTH, 128)
  
  inputs = tf.keras.Input(input_shape)

  x = tf.keras.layers.LSTM(512,return_sequences=True)(inputs)
  x = tf.keras.layers.Dropout(0.3)(x)
  x = tf.keras.layers.LSTM(512,return_sequences=True)(x)
  x = tf.keras.layers.Dropout(0.3)(x)
  x = tf.keras.layers.LSTM(512)(x)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.Dropout(0.3)(x)
  x = tf.keras.layers.Dense(256, activation='relu')(x)

  outputs = tf.keras.layers.Dense(128, activation="softmax", name='piano_roll')(x)


  model = tf.keras.Model(inputs, outputs)

  loss = tf.keras.losses.CategoricalCrossentropy()

  optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

  
  model.compile(
    loss=loss,
    optimizer=optimizer,
  )

  return model, loss, optimizer

def train_model(model, train_ds, val_ds, save_model_path):
  callbacks = [
    # tf.keras.callbacks.ModelCheckpoint(
    #   filepath='./training_checkpoints/ckpt_{epoch}',
    #   save_weights_only=True),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=125,
        verbose=1,
        restore_best_weights=True),
  ]

  history = model.fit(
      train_ds,
      epochs=EPOCHS,
      callbacks=callbacks,
      validation_data=val_ds,
  )

  model.save_weights(save_model_path)

  plt.plot(history.epoch, history.history['loss'], label='total training loss')
  plt.savefig(save_model_path+'training_loss.png')
  plt.figure()
  plt.plot(history.epoch, history.history['val_loss'], label='total val loss')
  plt.savefig(save_model_path+'validation_loss.png') 


if __name__  == "__main__":
    
  load_model = False
  only_train = True

  load_model_path = 'models/beatles/melody/model1/250_epochs/250_epochs'

  seq_ds = prepare_data("data/melody/test", INPUT_LENGTH, LABEL_LENGTH)
    
  train_ds, val_ds = split_data(seq_ds)

  model, loss, optimizer = create_model()

  if load_model:
    model.load_weights(load_model_path)
  else:
      train_model(model, train_ds, val_ds, "models/model1/e_50")
  
  # if not only_train:
  #   generated_notes, first_note = eval_model(model, raw_notes, "results/beatles/melody/model2/50_epochs", "Acoustic Grand Piano")
  #   plot_piano_roll(generated_notes, first_note)

  # plt.show()




'''
for input_seq, label_seq in dataset.take(5):
    print("Input sequence:\n", input_seq.numpy())
    print("Label sequence:\n", label_seq.numpy())
    print()
'''