import os
import argparse
import pretty_midi
import numpy as np
import collections
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras

seed = 42
# tf.random.set_seed(seed)
# np.random.seed(seed)

# Sampling rate for audio playback
FS = 16000
EPOCHS = 100
MIDI_INSTRUMENT = "Acoustic Grand Piano"

def prepare_data(training_data_path,seq_length=25, vocab_size=128):
  all_notes = []
  for i in os.listdir(training_data_path):
      full_path = training_data_path+'/'+i
      if ".mid" in i:
          pm = pretty_midi.PrettyMIDI(full_path)
          raw_notes = midi_to_notes(pm)
          all_notes.append(raw_notes)
  all_notes = pd.concat(all_notes)

  # all_notes = normalize_pitch(all_notes)

  key_order = ['pitch', 'step', 'duration']
  train_notes = np.stack([all_notes[key] for key in key_order], axis=1)
  notes_ds = tf.data.Dataset.from_tensor_slices(train_notes)

  seq_ds = create_sequences(notes_ds, seq_length, vocab_size)

  return raw_notes, all_notes, seq_ds

def normalize_pitch(notes):
  count = 0
  for i in range(notes.shape[0]):
    if notes['pitch'].iloc[i] >= 72 or notes['pitch'].iloc[i] <=60 :
      notes['pitch'].iloc[i] = (notes['pitch'].iloc[i] % 12) + 60
      count += 1
  return notes

def create_and_train_model(buffer_size, seq_length, seq_ds, save_model_path, batch_size=64):
  '''
  This function creates and trains a model with all midi files found in the given path.
  The model is saved in the training_checkpoint folder.
  '''

  
  dataset = (seq_ds.shuffle(buffer_size))


  val_size = int(len(list(seq_ds))*0.10)
  
  train_ds = dataset.skip(val_size)
  val_ds = dataset.take(val_size)

  val_ds = val_ds.batch(batch_size, drop_remainder=True).cache().prefetch(tf.data.experimental.AUTOTUNE)
  train_ds = train_ds.batch(batch_size, drop_remainder=True).cache().prefetch(tf.data.experimental.AUTOTUNE)

  model, loss, optimizer = create_model(seq_length)

  #losses = model.evaluate(train_ds, return_dict=True)

  history = train_model(model, train_ds, val_ds)

  model.save_weights(save_model_path)

  plt.plot(history.epoch, history.history['loss'], label='total training loss')
  plt.savefig(save_model_path+'training_loss.png')
  plt.figure()
  plt.plot(history.epoch, history.history['val_loss'], label='total val loss')
  plt.savefig(save_model_path+'validation_loss.png')
  
  return model

def eval_model(model, key_order, raw_notes, seq_length, vocab_size, out_file, instrument, temperature=2, num_predictions=120):

  sample_notes = np.stack([raw_notes[key] for key in key_order], axis=1)

  # The initial sequence of notes; pitch is normalized similar to training
  # sequences
  input_notes = (
      sample_notes[:seq_length] / np.array([vocab_size, 1, 1]))

  generated_notes = []
  prev_start = 0
  for _ in range(num_predictions):
    pitch, step, duration = predict_next_note(input_notes, model, temperature)
    start = prev_start + step
    end = start + duration
    input_note = (pitch, step, duration)
    generated_notes.append((*input_note, start, end))
    input_notes = np.delete(input_notes, 0, axis=0)
    input_notes = np.append(input_notes, np.expand_dims(input_note, 0), axis=0)
    prev_start = start

  generated_notes = pd.DataFrame(
      generated_notes, columns=(*key_order, 'start', 'end'))
  
  #generated_notes = normalize_pitch(generated_notes)
  out_pm = notes_to_midi(
      generated_notes, out_file=out_file, instrument_name=instrument)

  return generated_notes

def main(train_model=False, dataset=None, save_model_name=None, load_model_name=None, instrument=None, temperature=2):
  seq_length = 25
  vocab_size = 128
  batch_size = 128

  midi_instrument="Acoustic Grand Piano"
  if instrument == 'piano':
    midi_instrument="Acoustic Grand Piano"
  elif instrument == 'drums':
    midi_instrument='Music box'
  elif instrument=='bass':  
    midi_instrument='Acoustic Bass'
  elif instrument=='melody':
    midi_instrument='Acoustic Grand Piano'

  
  key_order = ['pitch', 'step', 'duration']

  num_predictions = 120

  raw_notes, all_notes, seq_ds = prepare_data(dataset, seq_length, vocab_size)
  # print(all_notes)
  # return
  n_notes = len(all_notes)
  buffer_size = n_notes - seq_length  # the number of items in the dataset



  if train_model:
    save_model_path = "models/"+save_model_name

    #If training, we need to make validation set
    # _, _, seq_ds_val = prepare_data(val_set, seq_length, vocab_size)
    # n_notes_val = len(all_notes)

    # buffer_size_val = n_notes_val - seq_length

    out_file = 'results/'+save_model_name+'_temp'+str(temperature)+'.mid'
    model = create_and_train_model(buffer_size, seq_length, seq_ds, save_model_path, batch_size)
    generated_notes = eval_model(model, key_order, raw_notes, seq_length, vocab_size, out_file, midi_instrument)

  else:
    load_model_path = "models/"+load_model_name

    out_file = 'results/'+load_model_name+'_temp'+str(temperature)+'.mid'

    #to not overwrite a result file
    check_out_file = out_file
    n=1
    while os.path.isfile(check_out_file):
      check_out_file = out_file
      check_out_file+=str(n)
    out_file=check_out_file

    model,_,_ = create_model(seq_length)
    model.load_weights(load_model_path)
    generated_notes = eval_model(model, key_order, raw_notes, seq_length, vocab_size, out_file, midi_instrument, temperature, num_predictions)

  plot_piano_roll(generated_notes)

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

  return sequences.map(split_labels, num_parallel_calls=tf.data.experimental.AUTOTUNE)

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

  # #separate part for pitch
  # out_pitch = tf.keras.layers.Dense(128, activation='relu')(x)
  # out_pitch = tf.keras.layers.BatchNormalization()(out_pitch)
  # out_pitch = tf.keras.layers.Dropout(0.3)(out_pitch)

  # #separate part for step
  # out_step = tf.keras.layers.Dense(128, activation='relu')(x)
  # out_step = tf.keras.layers.BatchNormalization()(out_step)
  # out_step = tf.keras.layers.Dropout(0.3)(out_step)

  # #separate part for duration
  # out_dur = tf.keras.layers.Dense(128, activation='relu')(x) 
  # out_dur = tf.keras.layers.BatchNormalization()(out_dur)
  # out_dur = tf.keras.layers.Dropout(0.3)(out_dur)

  outputs = {
    'pitch': tf.keras.layers.Dense(128, activation="softmax", name='pitch')(x),
    'step': tf.keras.layers.Dense(1, activation="softmax", name='step')(x),
    'duration': tf.keras.layers.Dense(1, activation="softmax",  name='duration')(x),
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
  # callbacks = [
    # tf.keras.callbacks.ModelCheckpoint(
    #     filepath='./training_checkpoints/ckpt_{EPOCH}',
    #     save_weights_only=True),
  #   tf.keras.callbacks.EarlyStopping(
  #       monitor='val_loss',
  #       min_delta=0,
  #       patience=25,
  #       verbose=1,
  #       restore_best_weights=True),
  # ]


  # pitch_dataset = train_ds[:,:,0]
  # step_dataset = train_ds[:,:,1]
  # duration_dataset = train_ds[:,:,2]

  # history = model.fit(
  #     [pitch_dataset, step_dataset, duration_dataset],
  #     epochs=EPOCHS,
  #     callbacks=callbacks,
  #     validation_data=val_ds,
  # )

  history = model.fit(
      train_ds,
      epochs=EPOCHS,
      # callbacks=callbacks,
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


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Description of your program')
  parser.add_argument('-t','--train',default=False)
  
  parser.add_argument('-ds','--dataset', required=True)

  parser.add_argument('-smn','--save_model_name')
  parser.add_argument('-lmn','--load_model_name')
  
  parser.add_argument('--instrument')
  parser.add_argument('--temp')

  parser.add_argument('--test')
  args = vars(parser.parse_args())

  temp = None
  inst = "piano" if not args['instrument'] else args['instrument']


  if args["train"] and (args["dataset"] is None or args["save_model_name"] is None):
    parser.error("--train requires --dataset and --save_model_name.")
  if not args["train"] and (args["load_model_name"] is None):
    parser.error("--load_model_name is required")
  if args["temp"]:
    temp = float(args["temp"])

  main(train_model = args['train'], dataset=args['dataset'],
      save_model_name=args['save_model_name'], load_model_name=args['load_model_name'],
      instrument=inst, temperature=temp)
  
  plt.show()
