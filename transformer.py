import os
import pretty_midi
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from help_funcs import midi_to_notes


def create_sequences_transformers(dataset: tf.data.Dataset, seq_length: int, vocab_size = 128,) -> tf.data.Dataset:
  """Returns TF Dataset of sequence and label examples."""
  key_order = ['pitch', 'step', 'duration']
  # Want even numbers
  


  # Take 1 extra for the labels
  windows = dataset.window(seq_length*2, shift=1, stride=1,
                              drop_remainder=True)


  # `flat_map` flattens the" dataset of datasets" into a dataset of tensors
  flatten = lambda x: x.batch(seq_length*2, drop_remainder=True)
  sequences = windows.flat_map(flatten)


  # Normalize note pitch
  def scale_pitch(x):
    x = x/[vocab_size,1.0,1.0]
    return x


  # Split the labels
  def split_labels(sequences):
    inputs = sequences[:seq_length]
    outputs = sequences[seq_length:(2*seq_length-1)]
    labels = sequences[seq_length+1:]

    return tf.transpose(inputs), tf.transpose(outputs), tf.transpose(labels)

  return sequences.map(split_labels)

def prepare_data_transformers(training_data_path,seq_length=25, vocab_size=128):
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
    
  seq = create_sequences_transformers(notes_ds, seq_length, vocab_size)
  return seq

def positional_encoding(length, depth):
  depth = depth/2

  positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
  depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)

  angle_rates = 1 / (10000**depths)         # (1, depth)
  angle_rads = positions * angle_rates      # (pos, depth)

  pos_encoding = np.concatenate(
      [np.sin(angle_rads), np.cos(angle_rads)],
      axis=-1) 

  return tf.cast(pos_encoding, dtype=tf.float32)


class PositionalEmbedding(tf.keras.layers.Layer):
  def __init__(self, vocab_size, d_model):
    super().__init__()
    self.d_model = d_model
    self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True) 
    self.pos_encoding = positional_encoding(length=2048, depth=d_model)

  def compute_mask(self, *args, **kwargs):
    return self.embedding.compute_mask(*args, **kwargs)

  def call(self, x):
    length = tf.shape(x)[1]
    x = self.embedding(x)
    # This factor sets the relative scale of the embedding and positonal_encoding.
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x = x + self.pos_encoding[tf.newaxis, :length, :]
    return x


class BaseAttention(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super().__init__()
    self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
    self.layernorm = tf.keras.layers.LayerNormalization()
    self.add = tf.keras.layers.Add()


class CrossAttention(BaseAttention):
  #Called by passing the target sequence x as query
  #And context sequence as key/value
  def call(self, x, context):
    attn_output, attn_scores = self.mha(
        query=x,
        key=context,
        value=context,
        return_attention_scores=True)

    # Cache the attention scores for plotting later.
    self.last_attn_scores = attn_scores

    x = self.add([x, attn_output])
    x = self.layernorm(x)

    return x