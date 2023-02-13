import os
import pretty_midi
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras

import argparse

from help_funcs import midi_to_notes, plot_piano_roll, notes_to_midi, create_sequences, create_model, train_model, predict_next_note
from transformer import prepare_data_transformers

seed = 42
# tf.random.set_seed(seed)
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

def create_and_train_model(buffer_size, seq_length, seq_ds, save_model_path, batch_size=64):
  '''
  This function creates and trains a model with all midi files found in the given path.
  The model is saved in the training_checkpoint folder.
  '''

  
  dataset = (seq_ds.shuffle(buffer_size))


  val_size = int(len(list(seq_ds))*0.15)
  
  train_ds = dataset.skip(val_size)
  val_ds = dataset.take(val_size)

  val_ds = val_ds.batch(batch_size, drop_remainder=True).cache().prefetch(tf.data.experimental.AUTOTUNE)
  train_ds = train_ds.batch(batch_size, drop_remainder=True).cache().prefetch(tf.data.experimental.AUTOTUNE)

  model, loss, optimizer = create_model(seq_length)

  losses = model.evaluate(train_ds, return_dict=True)

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
  
    
  out_pm = notes_to_midi(
      generated_notes, out_file=out_file, instrument_name=instrument)

  return generated_notes



def main(train_model=False, dataset=None, save_model_name=None, load_model_name=None, instrument=None, temperature=2):
  seq_length = 25
  vocab_size = 128
  batch_size = 128

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
  print((list(seq_ds)[0][0]))
  print((list(seq_ds)[1][0]))
  return
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


def main_transformer(train_model=False, dataset="data/melody/test", save_model_name=None, load_model_name=None, instrument=None, temperature=2):
  seq_length = 25
  vocab_size = 128
  batch_size = 128
  
  if instrument == 'piano':
    midi_instrument="Acoustic Grand Piano"
  elif instrument == 'drums':
    midi_instrument='Music box'
  elif instrument=='bass':  
    midi_instrument='Acoustic Bass'
  elif instrument=='melody':
    midi_instrument='Acoustic Grand Piano'

  key_order = ['pitch', 'step', 'duration']
  print(prepare_data_transformers(dataset, seq_length, vocab_size))
  inputs, outputs, labels = prepare_data_transformers(dataset, seq_length, vocab_size)


  print(inputs)
  print(outputs)
  print(labels)




if __name__ == "__main__":
  # parser = argparse.ArgumentParser(description='Description of your program')
  # parser.add_argument('-t','--train',default=False)
  
  # parser.add_argument('-ds','--dataset', required=True)

  # parser.add_argument('-smn','--save_model_name')
  # parser.add_argument('-lmn','--load_model_name')
  
  # parser.add_argument('--instrument', required=True)
  # parser.add_argument('--temp')

  # parser.add_argument('--test')
  # args = vars(parser.parse_args())

  # temp = None


  main_transformer()

  if False:
    if args["train"] and (args["dataset"] is None or args["save_model_name"] is None):
      parser.error("--train requires --dataset and --save_model_name.")
    if not args["train"] and (args["load_model_name"] is None):
      parser.error("--load_model_name is required")
    if args["temp"]:
      temp = float(args["temp"])

    main(train_model = args['train'], dataset=args['dataset'],
        save_model_name=args['save_model_name'], load_model_name=args['load_model_name'],
        instrument=args['instrument'], temperature=temp)
    
    plt.show()
