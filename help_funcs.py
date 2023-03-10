import pretty_midi
import numpy as np
import pandas as pd
import collections
from matplotlib import pyplot as plt
import tensorflow as tf



def midi_to_notes(pm) -> pd.DataFrame:
  instrument = pm.instruments[0]
  notes = collections.defaultdict(list)

  # Sort the notes by start time
  sorted_notes = sorted(instrument.notes, key=lambda note: note.start)
  prev_start = sorted_notes[0].start

  for note in sorted_notes:
    start = note.start
    end = note.end
    notes['pitch'].append(note.pitch)
    notes['start'].append(start)
    notes['end'].append(end)
    notes['step'].append(start - prev_start)
    notes['duration'].append(end - start)
    prev_start = start
 
  return pd.DataFrame({name: np.array(value) for name, value in notes.items()})


def plot_piano_roll(notes: pd.DataFrame, count=None):
  if count:
    title = f'First {count} notes'
  else:
    title = f'Whole track'
    count = len(notes['pitch'])
  plt.figure(figsize=(20, 4))
  plot_pitch = np.stack([notes['pitch'], notes['pitch']], axis=0)
  plot_start_stop = np.stack([notes['start'], notes['end']], axis=0)
  plt.plot(
      plot_start_stop[:, :count], plot_pitch[:, :count], color="b", marker=".")
  plt.xlabel('Time [s]')
  plt.ylabel('Pitch')
  _ = plt.title(title)
  plt.show()

def notes_to_midi(notes: pd.DataFrame, out_file: str, instrument_name: str, velocity: int = 100,) -> pretty_midi.PrettyMIDI:
  pm = pretty_midi.PrettyMIDI()
  instrument = pretty_midi.Instrument(
      program=pretty_midi.instrument_name_to_program(
          instrument_name))

  prev_start = 0
  for i, note in notes.iterrows():
    start = float(prev_start + note['step'])
    end = float(start + note['duration'])
    note = pretty_midi.Note(
        velocity=velocity,
        pitch=int(note['pitch']),
        start=start,
        end=end,
    )
    instrument.notes.append(note)
    prev_start = start

  pm.instruments.append(instrument)
  pm.write(out_file)
  return pm


def create_sequences(dataset: tf.data.Dataset, seq_length: int, vocab_size = 128,) -> tf.data.Dataset:
  """Returns TF Dataset of sequence and label examples."""
  key_order = ['pitch', 'step', 'duration']
  seq_length = seq_length+1

  # Take 1 extra for the labels
  windows = dataset.window(seq_length, shift=1, stride=1,
                              drop_remainder=True)
  

  # `flat_map` flattens the" dataset of datasets" into a dataset of tensors
  flatten = lambda x: x.batch(seq_length, drop_remainder=True)
  sequences = windows.flat_map(flatten)


  # Normalize note pitch
  def scale_pitch(x):
    x = x/[vocab_size,1.0,1.0]
    return x

  # Split the labels
  def split_labels(sequences):
    inputs = sequences[:-1]
    labels_dense = sequences[-1]
    labels = {key:labels_dense[i] for i,key in enumerate(key_order)}

    return scale_pitch(inputs), labels

  return sequences.map(split_labels, num_parallel_calls=tf.data.AUTOTUNE)


def mse_with_positive_pressure(y_true: tf.Tensor, y_pred: tf.Tensor):
  mse = (y_true - y_pred) ** 2
  positive_pressure = 10 * tf.maximum(-y_pred, 0.0)
  return tf.reduce_mean(mse + positive_pressure)

def create_model(seq_length):
  input_shape = (seq_length, 3)
  learning_rate = 0.005

  inputs = tf.keras.Input(input_shape)
  print(inputs.shape)


  # #separate part for pitch
  # inputPitch = tf.keras.layers.Input(shape=(input_shape))
  # inputPitch = tf.keras.layers.LSTM(256, input_shape=(input_shape), return_sequences=True)(inputPitch)
  # inputPitch = tf.keras.layers.Dropout(0.2)(inputPitch)

  # #separate part for step
  # inputStep = tf.keras.layers.Input(shape=(input_shape))
  # inputStep = tf.keras.layers.LSTM(256, input_shape=(input_shape), return_sequences=True)(inputStep)
  # inputStep = tf.keras.layers.Dropout(0.2)(inputStep)
	
  # #separate part for duration
  # inputDurations = tf.keras.layers.Input(shape=(input_shape))
  # inputDurations = tf.keras.layers.LSTM(256, input_shape=(input_shape), return_sequences=True)(inputDurations)
  # inputDurations = tf.keras.layers.Dropout(0.2)(inputDurations)
	

  # inputs = tf.keras.layers.concatenate([inputPitch, inputStep, inputDurations])



  # #Common part
  x = tf.keras.layers.LSTM(512,return_sequences=True)(inputs)
  x = tf.keras.layers.Dropout(0.3)(x)
  x = tf.keras.layers.LSTM(512,return_sequences=True)(x)
  x = tf.keras.layers.Dropout(0.3)(x)
  x = tf.keras.layers.LSTM(512)(x)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.Dropout(0.3)(x)
  x = tf.keras.layers.Dense(256, activation='relu')(x)

  #separate part for pitch
  out_pitch = tf.keras.layers.Dense(128, activation='relu')(x)
  out_pitch = tf.keras.layers.BatchNormalization()(out_pitch)
  out_pitch = tf.keras.layers.Dropout(0.3)(out_pitch)

  #separate part for step
  out_step = tf.keras.layers.Dense(128, activation='relu')(x)
  out_step = tf.keras.layers.BatchNormalization()(out_step)
  out_step = tf.keras.layers.Dropout(0.3)(out_step)

  #separate part for duration
  out_dur = tf.keras.layers.Dense(128, activation='relu')(x) 
  out_dur = tf.keras.layers.BatchNormalization()(out_dur)
  out_dur = tf.keras.layers.Dropout(0.3)(out_dur)

  outputs = {
    'pitch': tf.keras.layers.Dense(128, activation="softmax", name='pitch')(out_pitch),
    'step': tf.keras.layers.Dense(1, activation="softmax", name='step')(out_step),
    'duration': tf.keras.layers.Dense(1, activation="softmax",  name='duration')(out_dur),
  }
  #pholophonic
  #variational encoders

  model = tf.keras.Model(inputs, outputs)

  loss = {
        'pitch': tf.keras.losses.SparseCategoricalCrossentropy(),
        'step': mse_with_positive_pressure,
        'duration': mse_with_positive_pressure,
  }

  optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

  
  model.compile(
    loss=loss,
    #make loss the weighted sum since pitch loss is much higher
    loss_weights={
        'pitch': 0.05,
        'step': 1.0,
        'duration':1.0,
    },
    optimizer=optimizer,
)

  return model, loss, optimizer

  

def train_model(model, train_ds, val_ds):
  callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath='./training_checkpoints/ckpt_{epoch}',
        save_weights_only=True),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=25,
        verbose=1,
        restore_best_weights=True),
  ]


  epochs = 150

  # pitch_dataset = train_ds[:,:,0]
  # step_dataset = train_ds[:,:,1]
  # duration_dataset = train_ds[:,:,2]

  # history = model.fit(
  #     [pitch_dataset, step_dataset, duration_dataset],
  #     epochs=epochs,
  #     callbacks=callbacks,
  #     validation_data=val_ds,
  # )

  history = model.fit(
      train_ds,
      epochs=epochs,
      callbacks=callbacks,
      validation_data=val_ds,
  )
  return history


def predict_next_note(notes: np.ndarray, model: tf.keras.Model, temperature: float = 1.0) -> int:
  """Generates a note IDs using a trained sequence model."""
  assert temperature > 0

  # Add batch dimension
  inputs = tf.expand_dims(notes, 0)

  predictions = model.predict(inputs)
  pitch_logits = predictions['pitch']
  step = predictions['step']
  duration = predictions['duration']

  pitch_logits /= temperature
  pitch = tf.random.categorical(pitch_logits, num_samples=1)
  pitch = tf.squeeze(pitch, axis=-1)
  duration = tf.squeeze(duration, axis=-1)
  step = tf.squeeze(step, axis=-1)

  # `step` and `duration` values should be non-negative
  step = tf.maximum(0, step)
  duration = tf.maximum(0, duration)

  return int(pitch), float(step), float(duration)