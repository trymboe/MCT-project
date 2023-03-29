import os
import pretty_midi
import collections
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

def prepare_data(training_data_path, input_length, vocab_size, validation_size, batch_size):
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

    seq_ds = create_sequences(notes_ds, input_length, vocab_size)
    print(seq_ds)
    exit()
    train_ds, val_ds = split_data(seq_ds, validation_size, batch_size)
    for input_seq, label in seq_ds.take(1):
        print(input_seq)
        print(label)

    return train_ds, val_ds

def normalize_pitch(notes):
  count = 0
  for i in range(notes.shape[0]):
    if notes['pitch'].iloc[i] >= 72 or notes['pitch'].iloc[i] <=60 :
      notes['pitch'].iloc[i] = (notes['pitch'].iloc[i] % 12) + 60
      count += 1
  return notes

def create_sequences(dataset: tf.data.Dataset, input_length: int, vocab_size = 128,) -> tf.data.Dataset:
  """Returns TF Dataset of sequence and label examples."""
  key_order = ['pitch', 'step', 'duration']
  input_length = input_length+1

  # Take 1 extra for the labels
  windows = dataset.window(input_length, shift=1, stride=1,
                              drop_remainder=True)
  

  # `flat_map` flattens the" dataset of datasets" into a dataset of tensors
  flatten = lambda x: x.batch(input_length, drop_remainder=True)
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

def split_data(dataset, validation_size, batch_size):
    dataset = (dataset.shuffle(buffer_size=len(list(dataset))))


    val_size = int(len(list(dataset))*validation_size)

    train_ds = dataset.skip(val_size)
    val_ds = dataset.take(val_size)

    val_ds = val_ds.batch(batch_size, drop_remainder=True).cache().prefetch(tf.data.experimental.AUTOTUNE)
    train_ds = train_ds.batch(batch_size, drop_remainder=True).cache().prefetch(tf.data.experimental.AUTOTUNE)

    return train_ds, val_ds