import os
import pretty_midi
import numpy as np
import tensorflow as tf


def prepare_data(training_data_path, input_length, label_length):
  all_rolls = []
  for i in os.listdir(training_data_path):
    full_path = training_data_path+'/'+i
    if ".mid" in i:
      pm = pretty_midi.PrettyMIDI(full_path)
      
      pr = pm.get_piano_roll(fs=20).transpose()
      all_rolls.append(pr)

  for idx, pr in enumerate(all_rolls):
    pr = remove_silence(pr, threshold=10)
    pr[pr != 0] = 127
    all_rolls[idx] = pr

  seq_ds = create_sequences(all_rolls, input_length, label_length)
  num_training_points = seq_ds.reduce(0, lambda x, _: x + 1).numpy()
  print("Number of training points:", num_training_points)

  return seq_ds

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