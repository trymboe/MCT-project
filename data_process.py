import os
import pretty_midi
import numpy as np
import tensorflow as tf


def prepare_data(training_data_path, input_length, label_length, fs, validation_size, batch_size):
  all_rolls = []
  for i in os.listdir(training_data_path):
    full_path = training_data_path+'/'+i
    if ".mid" in i:
      pm = pretty_midi.PrettyMIDI(full_path)
      
      pr = pm.get_piano_roll(fs=fs).transpose()
      pr = remove_silence(pr, threshold=fs*1)
      pr[pr != 0] = 1
      # create a silence row
      silence_row = np.zeros((pr.shape[0], 1))
      pr = np.hstack((pr, silence_row))
      pr = add_silence(pr)
      all_rolls.append(pr)


  seq_ds = create_sequences(all_rolls, input_length, label_length)
  num_training_points = seq_ds.reduce(0, lambda x, _: x + 1).numpy()
  print("Number of training points:", num_training_points)

  train_ds, val_ds = split_data(seq_ds, validation_size, batch_size)

  return train_ds, val_ds

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

def create_sequences(piano_rolls, input_length, label_length):
    
    """
    Creates a TensorFlow dataset of input-label pairs from a list of piano roll arrays.

    Args:
        piano_rolls (List[np.ndarray]) : A list of numpy arrays representing the piano rolls to use for creating the sequences.
        Each array should have shape (time_steps, pitch_classes) where time_steps is the number of time steps in the sequence
        and pitch_classes is the number of pitch classes in the music. The arrays should have the same number of pitch classes.
        input_length (int) : The length of the input sequence to use for training.
        label_length (int) : The length of the label sequence to use for training.

    Returns:
        dataset (tf.data.Dataset): A TensorFlow dataset of input-label pairs.
    """

    piano_roll_array = np.concatenate(piano_rolls, axis=0)

    dataset = tf.data.Dataset.from_tensor_slices(piano_roll_array)
    dataset = dataset.window(input_length + label_length, shift=1, stride=1, drop_remainder=True)

    dataset = dataset.flat_map(lambda window: window.batch(input_length + label_length, drop_remainder=True))

    dataset = dataset.map(lambda window: (window[:-label_length], window[-label_length:]))
    return dataset

def split_data(dataset, validation_size, batch_size):
  '''
  This function creates and trains a model with all midi files found in the given path.
  The model is saved in the training_checkpoint folder.
  '''
  dataset = dataset.shuffle(buffer_size=len(list(dataset)))
  
  # Split dataset into training and validation sets
  train_size = int((1-validation_size) * len(list(dataset)))

  train_dataset = dataset.take(train_size)
  val_dataset = dataset.skip(train_size)

  # # batch the datasets
  val_dataset = val_dataset.batch(batch_size, drop_remainder=True)#.cache().prefetch(tf.data.experimental.AUTOTUNE)
  train_dataset = train_dataset.batch(batch_size, drop_remainder=True)#.cache().prefetch(tf.data.experimental.AUTOTUNE)

  def squeeze_label(x, y):
    return x, tf.squeeze(y)

  train_dataset = train_dataset.map(squeeze_label)
  val_dataset = val_dataset.map(squeeze_label)

  return train_dataset, val_dataset

def piano_roll_to_pretty_midi(piano_roll, fs, program=0):
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

def add_silence(pr):
  for timestep in pr:
    if np.nonzero(timestep)[0].size:
      timestep[128] = 1
  return pr
