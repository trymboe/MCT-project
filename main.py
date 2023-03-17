import os
import sys
import json
import pretty_midi
import collections
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
import librosa.display

from pretty_midi import PrettyMIDI, Instrument, Note

np.set_printoptions(threshold=sys.maxsize)

SEQ_LENGTH = 25
BATCH_SIZE = 32
NUM_PREDICTIONS = 120
VALIDATION_SIZE = 0.15
LEARNING_RATE = 0.005
NOISE_SCALE = 1
VOCAB_SIZE = 128
EPOCHS = 150
TEMPERATURE = 1
PROB = 0.3

def prepare_data(training_data_path):
  all_rolls = []
  for i in os.listdir(training_data_path):
    full_path = training_data_path+'/'+i
    if ".mid" in i:
      pm = pretty_midi.PrettyMIDI(full_path)
      
      pr = pm.get_piano_roll(fs=20).transpose()
      all_rolls.append(pr)


  for pr in all_rolls:
    pr = remove_silence(pr, threshold=100)

  # pm = piano_roll_to_pretty_midi(pr.transpose(), fs=20)

  # pm.write('output.mid')

  return 1,2,3
  notes_ds = tf.data.Dataset.from_tensor_slices(all_rolls)

  seq_ds = create_sequences(notes_ds)

  return raw_notes, all_notes, seq_ds

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

def remove_silence(pr, threshold=100):
  """
  Removes silence from a piano roll.

  Args:
      pr (numpy.ndarray): A piano roll as a numpy array.
      threshold (int): The number of consecutive silent timesteps required to remove a row.

  Returns:
      numpy.ndarray: The modified piano roll with silence removed.
  """
  # Compute the sum of each row in the piano roll
  row_sums = np.sum(pr, axis=1)

  # Find the silent rows
  silent_rows = np.where(row_sums == 0)[0]
  remove_rows = []
  count = 1
  for i in range(0,len(silent_rows)):
    if silent_rows[i] == silent_rows[i-1] + 1:
      count += 1
    elif count >= threshold:
      start_remove = silent_rows[i-1]-count+1
      end_remove = start_remove + count
      remove_rows.append(list(range(start_remove, end_remove)))
      count = 1
    else:
      count = 1


  if count >= threshold:
    start_remove = silent_rows[i]-count+1
    end_remove = start_remove + count
    remove_rows.append(list(range(start_remove, end_remove)))


  remove_rows = [num for sublist in remove_rows for num in sublist]
  keep_rows = np.ones(pr.shape[0], dtype=bool)
  keep_rows[remove_rows] = False

  # use boolean indexing to remove the specified rows
  pr = pr[keep_rows]
  

  # Remove the silent rows from the piano roll

  return pr

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

    x = x/[transition_max,duration_max, VOCAB_SIZE]
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
    notes["pitch"].append(note.pitch)
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
  x = tf.keras.layers.LSTM(512,return_sequences=True)(inputs)
  x = tf.keras.layers.Dropout(0.3)(x)
  x = tf.keras.layers.LSTM(512,return_sequences=True)(x)
  x = tf.keras.layers.Dropout(0.3)(x)
  x = tf.keras.layers.LSTM(512)(x)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.Dropout(0.3)(x)


  #separate part for transition
  out_trans = tf.keras.layers.Dense(128, activation='relu')(x)
  out_trans = tf.keras.layers.BatchNormalization()(out_trans)
  out_trans = tf.keras.layers.Dropout(0.3)(out_trans)

  #separate part for duration
  out_dur = tf.keras.layers.Dense(128, activation='relu')(x) 
  out_dur = tf.keras.layers.BatchNormalization()(out_dur)
  out_dur = tf.keras.layers.Dropout(0.3)(out_dur)

  #separate part for pitch
  out_pitch = tf.keras.layers.Dense(128, activation='relu')(x)
  out_pitch = tf.keras.layers.BatchNormalization()(out_pitch)
  out_pitch = tf.keras.layers.Dropout(0.3)(out_pitch)

  outputs = {
    'transition': tf.keras.layers.Dense(1, activation="relu", name='transition')(out_trans),
    'duration': tf.keras.layers.Dense(1, activation="relu",  name='duration')(out_dur),
    'pitch': tf.keras.layers.Dense(128, activation="softmax", name='pitch')(out_pitch),
  }
  #pholophonic
  #variational encoders

  model = tf.keras.Model(inputs, outputs)

  loss = {
        'pitch': tf.keras.losses.SparseCategoricalCrossentropy(),
        'transition': mse_with_positive_pressure,
        'duration': mse_with_positive_pressure,
  }

  optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

  
  model.compile(
    loss=loss,
    loss_weights={
      'transition': 1.0,
      'duration':1.0,
      'pitch': 0.1,
    },
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

def decision():
    return np.random.random() < PROB

def eval_model(model, raw_notes, out_file, instrument):

  sample_notes = np.stack([raw_notes[key] for key in KEY_ORDER], axis=1)

  # The initial sequence of notes; pitch is normalized similar to training
  # sequences
  transition_max, duration_max = load_values('scaling.json')

  input_notes = (sample_notes[:SEQ_LENGTH] / np.array([transition_max, duration_max, VOCAB_SIZE]))
  first_note = int(input_notes[-1][-1]*VOCAB_SIZE)
  last_note = input_notes[-1]
  generated_notes = []
  prev_end = 0

  for _ in range(NUM_PREDICTIONS):
    print("input note", last_note)
    transition, duration, pitch = predict_next_note(input_notes, model)
    start = prev_end
    end = start + duration
    #pitch rules
    if decision():
      print("pitch rules")
      print(pitch)
      transition = pitch - last_note[2]
      input_note = (transition, duration, pitch/VOCAB_SIZE)
    #transition rules
    else:
      print("transition rules")
      pitch = last_note[2] + transition
      input_note = (transition, duration, pitch/VOCAB_SIZE)


    generated_notes.append((*input_note, start, end))
    input_notes = np.delete(input_notes, 0, axis=0)
    input_notes = np.append(input_notes, np.expand_dims(input_note, 0), axis=0)
    prev_end = end
    last_note = input_note
    print("\n")
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

def predict_next_note(notes: np.ndarray, model: tf.keras.Model) -> int:
  """Generates a note IDs using a trained sequence model."""
  # Add batch dimension
  inputs = tf.expand_dims(notes, 0)

  predictions = model.predict(inputs)

  predictions["transition"]
  # predictions["duration"] *= duration_max
  predictions['pitch']

  
  # predictions = add_noise(predictions)


  duration = predictions['duration']
  transition = predictions['transition']
  pitch = predictions['pitch']


  # pitch_logits /= TEMPERATURE
  # transition_logits /= TEMPERATURE

  pitch = tf.random.categorical(pitch, num_samples=1)
  pitch = tf.squeeze(pitch, axis=-1)


  # transition = tf.random.categorical(transition, num_samples=1)
  transition = tf.squeeze(transition, axis=-1)
  

  duration = tf.squeeze(duration, axis=-1)

  # `duration` values should be non-negative
  duration = tf.maximum(0.00001, duration)
  pitch = tf.maximum(0, pitch)

  return float(transition), float(duration), float(pitch)

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
  pm.write(out_file+'_'+str(TEMPERATURE)+'.mid')
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
    
  load_model = False
  only_train = True

  load_model_path = 'models/beatles/melody/model1/250_epochs/250_epochs'

  raw_notes, all_notes, seq_ds = prepare_data("data/melody")
  
  # buffer_size = len(all_notes) - SEQ_LENGTH
  # val_ds, train_ds = split_data(buffer_size, seq_ds)
  
  # model, loss, optimizer = create_model()
  # if load_model:
  #   model.load_weights(load_model_path)
  # else:
  #   train_model(model, val_ds, train_ds, "models/beatles/melody/model2/150_epochs")
  
  # if not only_train:
  #   generated_notes, first_note = eval_model(model, raw_notes, "results/beatles/melody/model2/50_epochs", "Acoustic Grand Piano")
  #   plot_piano_roll(generated_notes, first_note)

  # plt.show()