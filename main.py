import os
import argparse
import pretty_midi
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras
from model import create_model, create_model_sequence, train_model
from data_process import prepare_data, notes_to_midi, plot_piano_roll
from generate import eval_model

seed = 42
# tf.random.set_seed(seed)
# np.random.seed(seed)

# Sampling rate for audio playback
FS = 16000
EPOCHS = 100
MIDI_INSTRUMENT = "Acoustic Grand Piano"
LEARNING_RATE = 0.005
INPUT_LENGTH = 25
VOCAB_SIZE = 128
BATCH_SIZE = 32
KEY_ORDER = ['pitch', 'step', 'duration']
NUM_PREDICTIONS = 120
VALIDATION_SIZE = 0.15
OPTIMIZER = "Adam"
TEMPERATURE = 1


if __name__ == "__main__":

  train = True
  sequence = False
  dataset = "test"
  model_name = "model2"

  load_model_path = f'models/{model_name}/{dataset}/e_{EPOCHS}_{INPUT_LENGTH}_{OPTIMIZER}'
  out_file = f"results/melody/{model_name}/{dataset}/e_{EPOCHS}_{INPUT_LENGTH}_{OPTIMIZER}"

  if sequence:
    if train:
      train_ds, val_ds = prepare_data(f"data/melody/test", INPUT_LENGTH, VOCAB_SIZE, VALIDATION_SIZE, BATCH_SIZE)
    else:
      train_ds, val_ds = prepare_data(f"data/melody/{dataset}", INPUT_LENGTH, VOCAB_SIZE, VALIDATION_SIZE, BATCH_SIZE)
    model, loss, optimizer = create_model_sequence(INPUT_LENGTH, LEARNING_RATE, OPTIMIZER)
  else:
    train_ds, val_ds = prepare_data(f"data/melody/{dataset}", INPUT_LENGTH, VOCAB_SIZE, VALIDATION_SIZE, BATCH_SIZE)
    model, loss, optimizer = create_model(INPUT_LENGTH, LEARNING_RATE, OPTIMIZER)

  if not train:
    model.load_weights(load_model_path)
  else:
      train_model(model, train_ds, val_ds, f"models/{model_name}/{dataset}/e_{EPOCHS}_{INPUT_LENGTH}_{OPTIMIZER}", EPOCHS)

  if not train:
    
    generated_notes = eval_model(model, KEY_ORDER, train_ds, INPUT_LENGTH, VOCAB_SIZE, TEMPERATURE, NUM_PREDICTIONS, sequence)
    pm = notes_to_midi(generated_notes, out_file, MIDI_INSTRUMENT)

    pm.write(out_file+".mid")
  plt.show()
