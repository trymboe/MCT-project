import os
import pretty_midi
import collections
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

def prepare_data(training_data_path, input_length, label_length, vocab_size, validation_size, batch_size):
    """
    Prepare the data for training and validation by loading MIDI files, creating sequences and splitting the data.

    Args:
    training_data_path (str): The path to the directory containing the training MIDI files.
    input_length (int): The length of the input sequence.
    label_length (int): The length of the label sequence.
    vocab_size (int): The number of unique tokens in the vocabulary.
    validation_size (float): The proportion of data to be used for validation.
    batch_size (int): The size of each batch.

    Returns:
    tuple: A tuple of two tf.data.Dataset objects containing the training and validation data.

    """
    all_notes = []
    for i in os.listdir(training_data_path):
        full_path = training_data_path+'/'+i
        if ".mid" in i:
            pm = pretty_midi.PrettyMIDI(full_path)
            raw_notes = midi_to_notes(pm)
            all_notes.append(raw_notes)
    all_notes = pd.concat(all_notes)
    print("Number of training points:", len(all_notes))
    # all_notes = normalize_pitch(all_notes)

    key_order = ['pitch', 'step', 'duration']
    train_notes = np.stack([all_notes[key] for key in key_order], axis=1)

    seq_ds = create_sequences(train_notes, input_length, label_length, vocab_size)

    train_ds, val_ds = split_data(seq_ds, validation_size, batch_size)


    return train_ds, val_ds


def create_sequences(notes, input_length, label_length, vocab_size):
    """
    Create sequences from the input list of notes to be used in a sequence model.

    Args:
    notes (np.array): An array of notes in numerical form.
    input_length (int): The length of the input sequence.
    label_length (int): The length of the output/label sequence.
    vocab_size (int): The number of unique values in the notes list.

    Returns:
    tf.data.Dataset: A TensorFlow Dataset object containing the input and output sequences
    for the sequence model.
    """
    def scale_pitch(x):
        x = x/[vocab_size,1.0,1.0]
        return x
    dataset = tf.data.Dataset.from_tensor_slices(notes)
    dataset = dataset.window(input_length + label_length, shift=1, stride=1, drop_remainder=True)

    dataset = dataset.flat_map(lambda window: window.batch(input_length + label_length, drop_remainder=True))

    dataset = dataset.map(lambda window: (scale_pitch(window[:-label_length]), scale_pitch(window[-label_length:])))
    return dataset

def midi_to_notes(pm) -> pd.DataFrame:
  '''
   converts a MIDI file into a pandas DataFrame containing note information.

  Args:

  pm: A pretty_midi.PrettyMIDI object representing the MIDI file to be converted.
  
  Returns:
  A pandas DataFrame where each row represents a note and contains the following columns:
  pitch: The MIDI note number representing the pitch of the note.
  start: The start time of the note, in seconds.
  end: The end time of the note, in seconds.
  step: The time duration between the start time of the current note and the start time of the previous note, in seconds.
  duration: The duration of the note, in seconds.
  '''
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
  """
  converts a pandas DataFrame of notes into a MIDI file using the pretty_midi package.

  Args:
  notes (pd.DataFrame): A pandas DataFrame containing the notes to be converted. The DataFrame should have three columns: step, pitch, and duration.
  out_file (str): The file path of the output MIDI file.
  instrument_name (str): The name of the instrument to be used in the MIDI file.
  velocity (int): The velocity of the notes in the MIDI file. Default is 100.
  
  Returns:
  pm (pretty_midi.PrettyMIDI): A PrettyMIDI object representing the converted MIDI file.
  """
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

  return pm

def split_data(dataset, validation_size, batch_size):
    """
    Split the given dataset into training and validation sets.

    Args:
    dataset: A tf.data.Dataset object containing the dataset to be split.
    validation_size: A float value indicating the fraction of data to be used for validation.
    batch_size: An integer value indicating the batch size for the datasets.

    Returns:
    train_ds: A tf.data.Dataset object containing the training dataset.
    val_ds: A tf.data.Dataset object containing the validation dataset.

    Raises:
    ValueError: If the validation size is not between 0 and 1 or if the batch size is not positive.
    """
    dataset = (dataset.shuffle(buffer_size=len(list(dataset))))


    val_size = int(len(list(dataset))*validation_size)

    train_ds = dataset.skip(val_size)
    val_ds = dataset.take(val_size)

    val_ds = val_ds.batch(batch_size, drop_remainder=True)#.cache().prefetch(tf.data.experimental.AUTOTUNE)
    train_ds = train_ds.batch(batch_size, drop_remainder=True)#.cache().prefetch(tf.data.experimental.AUTOTUNE)

    return train_ds, val_ds