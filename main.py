import os
import pretty_midi
import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import argparse

from help_funcs import midi_to_notes, plot_piano_roll, notes_to_midi, create_sequences, create_model, train_model, predict_next_note


seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

# Sampling rate for audio playback
FS = 16000

def prepare_data(training_data_path,seq_length=25, vocab_size=128):
  all_notes = []
  for i in os.listdir(training_data_path):
      full_path = training_data_path+'/'+i
      if ".mid" in i:
          pm = pretty_midi.PrettyMIDI(full_path)
          raw_notes = midi_to_notes(pm)
          all_notes.append(raw_notes)
  all_notes = pd.concat(all_notes)

  key_order = ['pitch', 'step', 'duration']
  train_notes = np.stack([all_notes[key] for key in key_order], axis=1)
  notes_ds = tf.data.Dataset.from_tensor_slices(train_notes)

  seq_ds = create_sequences(notes_ds, seq_length, vocab_size)

  return raw_notes, all_notes, seq_ds

def create_and_train_model(buffer_size, seq_length, seq_ds, save_model_path, val_seq,buffer_size_val, batch_size=64 ):
  '''
  This function creates and trains a model with all midi files found in the given path.
  The model is saved in the training_checkpoint folder.
  '''
  
  train_ds = (seq_ds
              .shuffle(buffer_size)
              .batch(batch_size, drop_remainder=True)
              .cache()
              .prefetch(tf.data.experimental.AUTOTUNE))
  
  val_ds = (val_seq
              .shuffle(buffer_size_val)
              .batch(batch_size, drop_remainder=True)
              .cache()
              .prefetch(tf.data.experimental.AUTOTUNE))

  model, loss, optimizer = create_model(seq_length)

  losses = model.evaluate(train_ds, return_dict=True)

  history = train_model(model, train_ds, val_ds)

  model.save_weights(save_model_path)

  plt.plot(history.epoch, history.history['loss'], label='total loss')
  plt.show()
  return model

def eval_model(model, key_order, raw_notes, seq_length, vocab_size, out_file, temperature=2, num_predictions=120):

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
  
    
  out_pm = notes_to_midi(
      generated_notes, out_file=out_file, instrument_name="steel drums")

  return generated_notes



def main(train_model):
  seq_length = 25
  vocab_size = 128
  batch_size = 64
  load_model_path = "models/piano_model1"
  save_model_path = "models/mozart_model1"
  out_file = 'results/mozart1.mid'
  key_order = ['pitch', 'step', 'duration']

  num_predictions = 120
  temperature = 2

  raw_notes, all_notes, seq_ds = prepare_data("piano/train", seq_length, vocab_size)

  raw_notes_val, all_notes_val, seq_ds_val = prepare_data("piano/val", seq_length, vocab_size)

  n_notes = len(all_notes)
  buffer_size = n_notes - seq_length  # the number of items in the dataset

  n_notes_val = len(all_notes)
  buffer_size_val = n_notes_val - seq_length

  if train_model:
    model = create_and_train_model(buffer_size, seq_length, seq_ds, save_model_path, seq_ds_val, buffer_size_val, batch_size)
    generated_notes = eval_model(model, key_order, raw_notes, seq_length, vocab_size, out_file)

  else:
    model,_,_ = create_model(seq_length)
    model.load_weights(load_model_path)
    generated_notes = eval_model(model, key_order, raw_notes, seq_length, vocab_size, out_file, temperature, num_predictions)

  plot_piano_roll(generated_notes)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Description of your program')
  parser.add_argument('-t','--train',default=False)
  args = vars(parser.parse_args())

  main(args['train'])
