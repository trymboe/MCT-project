import os
import pretty_midi
import numpy as np
import matplotlib.pyplot as plt
import collections
import pandas as pd
import random
import tensorflow as tf
from tensorflow import keras

def prepare_data(training_data_path,seq_length=25, vocab_size=128):
  dataset = []

  for idx, folder in enumerate(os.listdir(training_data_path)):
    if "DS" not in folder:
      for i in os.listdir(training_data_path+'/'+folder):
        if "DS" not in folder:
          full_path = training_data_path+'/'+folder+'/'+i
          if "melody" in i:
            melody_ds = extract_notes(full_path)
          if "piano" in i:
            piano_ds = extract_notes(full_path)
      dataset.append([piano_ds,melody_ds])

  dataset = batch_dataset(dataset,32)

  return dataset

def batch_dataset(dataset, batch_size):
   notes_to_midi(dataset[0][1], "test.mid", "Acoustic Grand Piano")

   for inst in dataset:
      i = 0
      while(True):
        if len(inst) > i*batch_size:
          piano = inst[1][batch_size*i:batch_size*(i+1)]
          print(piano)


        else:
           break
        i+=1

def extract_notes(path):
  pm_melody = pretty_midi.PrettyMIDI(path)
  all_notes = midi_to_notes(pm_melody)
  key_order = ['pitch', 'start', 'duration']
  train_notes = np.stack([all_notes[key] for key in key_order], axis=1)
  #notes_ds = tf.data.Dataset.from_tensor_slices(train_notes)
  return all_notes

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

def notes_to_midi(notes: pd.DataFrame, out_file: str, instrument_name: str, velocity: int = 100,) -> pretty_midi.PrettyMIDI:
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(
        program=pretty_midi.instrument_name_to_program(
            instrument_name))

    prev_start = 0
    first_note_start = notes['start'].iloc[0]
    # If the first note does not start at time 0, insert a silent note
    if first_note_start > 0:
        
        silent_note = pretty_midi.Note(
            velocity=0,
            pitch=0,
            start=0,
            end=first_note_start
        )
        instrument.notes.append(silent_note)
        prev_start = first_note_start

    # Add the notes from the pandas DataFrame
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
    print("saved to "+out_file)
    pm.write(out_file)
    return pm

# def create_and_train_model(dataset, seq_length, seq_ds, save_model_path, batch_size=64):
#   '''
#   This function creates and trains a model with all midi files found in the given path.
#   The model is saved in the training_checkpoint folder.
#   '''

#   random.shuffle(dataset)

#   val_size = int(len(list(seq_ds))*0.15)
  
#   train_ds = dataset[:val_size]
#   val_ds = dataset[val_size:]

#   val_ds = val_ds.batch(batch_size, drop_remainder=True).cache().prefetch(tf.data.experimental.AUTOTUNE)
#   train_ds = train_ds.batch(batch_size, drop_remainder=True).cache().prefetch(tf.data.experimental.AUTOTUNE)

#   model, loss, optimizer = create_model(seq_length)

#   #losses = model.evaluate(train_ds, return_dict=True)

#   history = train_model(model, train_ds, val_ds)

#   model.save_weights(save_model_path)

#   plt.plot(history.epoch, history.history['loss'], label='total training loss')
#   plt.savefig(save_model_path+'training_loss.png')
#   plt.figure()
#   plt.plot(history.epoch, history.history['val_loss'], label='total val loss')
#   plt.savefig(save_model_path+'validation_loss.png')
  
#   return model

if __name__ == "__main__":
    dataset = "data/duet/small"
    seq_length = 25
    vocab_size = 128

    #raw_notes, all_notes, seq_ds = 
    dataset = prepare_data(dataset, seq_length, vocab_size)

    