import os
import pretty_midi
import numpy as np
import matplotlib.pyplot as plt
import collections
import pandas as pd
import tensorflow as tf
from tensorflow import keras

def prepare_data(training_data_path,seq_length=25, vocab_size=128):
  all_notes = []
  for folder in os.listdir(training_data_path):
    for i in os.listdir(training_data_path+'/'+folder):
      print(folder, i)
      full_path = training_data_path+'/'+folder+'/'+i
      if "melody" in i:
        melody_ds = extract_notes(full_path)
      if "piano" in i:
        piano_ds = extract_notes(full_path)

    # seq_ds = create_sequences(melody_ds, piano_ds, seq_length, vocab_size)

  # return raw_notes, all_notes, seq_ds

def extract_notes(path):
  pm_melody = pretty_midi.PrettyMIDI(path)
  all_notes = midi_to_notes(pm_melody)
  notes_to_midi(all_notes, path, "Acoustic Grand Piano")
  key_order = ['pitch', 'step', 'duration']
  train_notes = np.stack([all_notes[key] for key in key_order], axis=1)
  notes_ds = tf.data.Dataset.from_tensor_slices(train_notes)
  return notes_ds

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
    print(notes['pitch'].iloc[0])
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


if __name__ == "__main__":
    dataset = "data/duet/test"
    seq_length = 25
    vocab_size = 128

    raw_notes, all_notes, seq_ds = prepare_data(dataset, seq_length, vocab_size)