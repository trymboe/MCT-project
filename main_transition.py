import os
import pretty_midi
import numpy as np
import collections
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import json
from tensorflow import keras

SEQ_LENGTH = 25
BATCH_SIZE = 32
NUM_PREDICTIONS = 120
VALIDATION_SIZE = 0.15
LEARNING_RATE = 0.005
NOISE_SCALE = 1
KEY_ORDER = ['transition', 'duration']
EPOCHS = 50

def prepare_data(training_data_path):
  all_notes = []
  for i in os.listdir(training_data_path):
    full_path = training_data_path+'/'+i
    if ".mid" in i:
      pm = pretty_midi.PrettyMIDI(full_path)
      raw_notes = midi_to_notes(pm)
      all_notes.append(raw_notes)
  all_notes = pd.concat(all_notes)


  #If you want to save the max value. Needs to be done if it is the first run of a new dataset.
  #It is used for scaling.
  save_values(all_notes.max(axis=0).iloc[0], all_notes.max(axis=0).iloc[1], "scaling.json")

  train_notes = np.stack([all_notes[key] for key in KEY_ORDER], axis=1)
  notes_ds = tf.data.Dataset.from_tensor_slices(train_notes)

  seq_ds = create_sequences(notes_ds)

  return raw_notes, all_notes, seq_ds

def save_values(value1, value2, filename):
  # Save the values to a file
  with open(filename, 'w') as f:
      json.dump({'transition_max': value1, 'duration_max': value2}, f)

def load_values(filename): 
  # Load the values from a file
  
  with open(filename, 'r') as f:
      data = json.load(f)
  return data['transition_max'], data['duration_max']

def create_sequences(dataset: tf.data.Dataset) -> tf.data.Dataset:
  """Returns TF Dataset of sequence and label examples."""


  # Take 1 extra for the labels
  windows = dataset.window(SEQ_LENGTH+1, shift=1, stride=1,
                              drop_remainder=True)
  

  # `flat_map` flattens the" dataset of datasets" into a dataset of tensors
  flatten = lambda x: x.batch(SEQ_LENGTH+1, drop_remainder=True)
  sequences = windows.flat_map(flatten)


  # Normalize transition
  def scale_transition(x):
    transition_max, duration_max = load_values('scaling.json')

    x = x/[transition_max,duration_max]
    return x

  # Split the labels
  def split_labels(sequences):
    scale_transition(sequences)
    inputs = sequences[:-1]
    labels_dense = sequences[-1]
    labels = {key:labels_dense[i] for i,key in enumerate(KEY_ORDER)}
    return inputs, labels

  return sequences.map(split_labels, num_parallel_calls=tf.data.experimental.AUTOTUNE)

def midi_to_notes(pm) -> pd.DataFrame:
  instrument = pm.instruments[0]
  notes = collections.defaultdict(list)

  
  # Sort the notes by start time
  sorted_notes = sorted(instrument.notes, key=lambda note: note.start)
  
  prev_note = None
  for note in sorted_notes:
    notes["transition"].append(note.pitch - prev_note.pitch) if prev_note else notes["transition"].append(0)
    notes["duration"].append(note.end - note.start)
    prev_note = note

  return pd.DataFrame({name: np.array(value) for name, value in notes.items()})

def split_data(buffer_size, seq_ds):
  '''
  This function creates and trains a model with all midi files found in the given path.
  The model is saved in the training_checkpoint folder.
  '''
  dataset = (seq_ds.shuffle(buffer_size))

  val_size = int(len(list(seq_ds))*VALIDATION_SIZE)
  
  train_ds = dataset.skip(val_size)
  val_ds = dataset.take(val_size)

  val_ds = val_ds.batch(BATCH_SIZE, drop_remainder=True).cache().prefetch(tf.data.experimental.AUTOTUNE)
  train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=True).cache().prefetch(tf.data.experimental.AUTOTUNE)

  return val_ds, train_ds

def mse_with_positive_pressure(y_true: tf.Tensor, y_pred: tf.Tensor):
  mse = (y_true - y_pred) ** 2
  positive_pressure = 10 * tf.maximum(-y_pred, 0.0)
  return tf.reduce_mean(mse + positive_pressure)

def create_model():
  input_shape = (SEQ_LENGTH, len(KEY_ORDER))
  
  inputs = tf.keras.Input(input_shape)

  # #Common part
  x = tf.keras.layers.LSTM(512)(inputs)



  #separate part for transition
  out_trans = tf.keras.layers.Dense(128, activation='relu')(x)
  out_trans = tf.keras.layers.BatchNormalization()(out_trans)
  out_trans = tf.keras.layers.Dropout(0.8)(out_trans)

  #separate part for duration
  out_dur = tf.keras.layers.Dense(128, activation='relu')(x) 
  out_dur = tf.keras.layers.BatchNormalization()(out_dur)
  out_dur = tf.keras.layers.Dropout(0.8)(out_dur)

  outputs = {
    'transition': tf.keras.layers.Dense(1, activation="relu", name='transition')(x),
    'duration': tf.keras.layers.Dense(1, activation="relu",  name='duration')(x),
  }
  #pholophonic
  #variational encoders

  model = tf.keras.Model(inputs, outputs)

  loss = {
        'transition': mse_with_positive_pressure,
        'duration': mse_with_positive_pressure,
  }

  optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

  
  model.compile(
    loss=loss,
    loss_weights={
      'transition': 0.1,
      'duration':1.0,
    },
    optimizer=optimizer,
  )

  return model, loss, optimizer

def train_model(model, train_ds, val_ds, save_model_path):
  callbacks = [
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
  # plt.savefig(save_model_path+'training_loss.png')
  plt.figure()
  plt.plot(history.epoch, history.history['val_loss'], label='total val loss')
  # plt.savefig(save_model_path+'validation_loss.png') 

def eval_model(model, raw_notes, out_file, instrument, temperature=100):

  sample_notes = np.stack([raw_notes[key] for key in KEY_ORDER], axis=1)

  # The initial sequence of notes; pitch is normalized similar to training
  # sequences
  transition_max, duration_max = load_values('scaling.json')

  input_notes = (sample_notes[:SEQ_LENGTH] / np.array([transition_max, duration_max]))
  
  first_note = 60
  generated_notes = []
  prev_end = 0
  for _ in range(NUM_PREDICTIONS):
    transition, duration = predict_next_note(input_notes, model, temperature)
    start = prev_end
    end = start + duration
    input_note = (transition, duration)
    generated_notes.append((*input_note, start, end))
    input_notes = np.delete(input_notes, 0, axis=0)
    input_notes = np.append(input_notes, np.expand_dims(input_note, 0), axis=0)
    prev_end = end

  generated_notes = pd.DataFrame(
      generated_notes, columns=(*KEY_ORDER, 'start', 'end'))
  
    
  out_pm = notes_to_midi(
      generated_notes, first_note, out_file=out_file, instrument_name=instrument)

  return generated_notes, first_note

def add_noise(prediction):
    # Add Gaussian noise to the output of the 'transition' neuron
    noise = NOISE_SCALE * np.random.randn()
    print("noise ",noise)
    prediction['transition'] += noise

    # Return the output
    return prediction

def predict_next_note(notes: np.ndarray, model: tf.keras.Model, temperature: float = 1.0) -> int:
  """Generates a note IDs using a trained sequence model."""
  transition_max, duration_max = load_values('scaling.json')
  # Add batch dimension
  inputs = tf.expand_dims(notes, 0)

  predictions = model.predict(inputs)
  print("First prediction ", predictions)

  predictions["transition"] *= transition_max
  predictions["duration"] *= duration_max
  print("prediction after scaling ",predictions)
  
  # predictions = add_noise(predictions)
  print("prediction after noise ",predictions)
  print("\n")

  transition = predictions['transition']
  duration = predictions['duration']

  transition = tf.squeeze(transition, axis=-1)
  duration = tf.squeeze(duration, axis=-1)

  # `duration` values should be non-negative
  duration = tf.maximum(0, duration)

  return float(transition), float(duration)

def notes_to_midi(notes: pd.DataFrame, first_note, out_file: str, instrument_name: str, velocity: int = 100,) -> pretty_midi.PrettyMIDI:
  pm = pretty_midi.PrettyMIDI()
  instrument = pretty_midi.Instrument(
      program=pretty_midi.instrument_name_to_program(
          instrument_name))

  prev_end = 0
  cur_note = first_note
  for i, note in notes.iterrows():
    start = float(prev_end)
    end = float(start + note['duration'])
    new_note = pretty_midi.Note(
        velocity=velocity,
        pitch=int(cur_note),
        start=start,
        end=end,
    )
    instrument.notes.append(new_note)
    prev_end = end
    cur_note += note["transition"]
    if cur_note > 127 or cur_note < 0:
      cur_note = 60

  pm.instruments.append(instrument)
  pm.write(out_file)
  return pm

def plot_piano_roll(notes: pd.DataFrame, last_note, count=None):
  if count:
    title = f'First {count} notes'
  else:
    title = f'Whole track'
    count = len(notes['transition'])
  plt.figure(figsize=(20, 4))
  pitch = np.zeros(count)
  pitch[0] = last_note + notes['transition'][0]
  for i in range(1, count):
      pitch[i] = pitch[i-1] + notes['transition'][i]
  plot_pitch = np.stack([pitch, pitch], axis=0)
  plot_start_stop = np.stack([notes['start'], notes['end']], axis=0)
  plt.plot(
      plot_start_stop[:, :count], plot_pitch[:, :count], color="b", marker=".")
  plt.xlabel('Time [s]')
  plt.ylabel('Pitch')
  _ = plt.title(title)

if __name__  == "__main__":
    

  raw_notes, all_notes, seq_ds = prepare_data("data/beatles/melody")

  
  buffer_size = len(all_notes) - SEQ_LENGTH
  val_ds, train_ds = split_data(buffer_size, seq_ds)
  
  model, loss, optimizer = create_model()
  train_model(model, val_ds, train_ds, "melody/beatles1")
  generated_notes, first_note = eval_model(model, raw_notes, "results/beatles.mid", "Acoustic Grand Piano")
  plot_piano_roll(generated_notes, first_note)

  plt.show()

