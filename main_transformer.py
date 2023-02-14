import os
import pretty_midi
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from transformer import *


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
  seq = prepare_data_transformers(dataset, seq_length, vocab_size)


  for inputs, outputs, labels in seq.take(1):
    break

  sample_ca = CrossAttention(num_heads=2, key_dim=512)
  
  print((inputs).shape)
  print(outputs.shape)
  print(sample_ca(inputs, outputs).shape)



if __name__ == "__main__":

  main_transformer()