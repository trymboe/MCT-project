import os
import sys
import pretty_midi
import collections
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


from data_process import prepare_data, piano_roll_to_pretty_midi
from generate import eval_model
from model import create_model, create_model_sequence, train_model

np.set_printoptions(threshold=sys.maxsize)


BATCH_SIZE = 32
NUM_PREDICTIONS = 1
VALIDATION_SIZE = 0.15
LEARNING_RATE = 0.005
NOISE_SCALE = 1
VOCAB_SIZE = 128
EPOCHS = 2
TEMPERATURE = 1
PROB = 0.3
INPUT_LENGTH = 120
LABEL_LENGTH = 120
# 120 bpm, 2 bps, 3*2 (represent triplets), 6*2 (nyqvist rate)
FS = 12

if __name__  == "__main__":
    
  train = True
  sequence = True
  dataset = "small"
  model_name = "model2"

  os.environ['TF_DEVICE']='/gpu:0'
  physical_devices = tf.config.list_physical_devices('GPU')
  tf.config.experimental.set_virtual_device_configuration(
      physical_devices[0],
      [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])

  load_model_path = f'models/{model_name}/{dataset}/e_{EPOCHS}'
  out_file = f"results/melody/{model_name}/{dataset}/e_{EPOCHS}"

  train_ds, val_ds = prepare_data(f"data/melody/{dataset}", INPUT_LENGTH, LABEL_LENGTH, FS, VALIDATION_SIZE, BATCH_SIZE)
  if sequence:
    model, loss, optimizer = create_model_sequence(INPUT_LENGTH, LEARNING_RATE)
  else:
    model, loss, optimizer = create_model(INPUT_LENGTH, LEARNING_RATE)

  if not train:
    model.load_weights(load_model_path)
  else:
      train_model(model, train_ds, val_ds, f"models/{model_name}/{dataset}/e_{EPOCHS}", EPOCHS)

  if not train:
    generated_notes = eval_model(model, train_ds, INPUT_LENGTH, num_predictions=NUM_PREDICTIONS, sequence=sequence)
    pm = piano_roll_to_pretty_midi(generated_notes.transpose(), FS)

    pm.write(out_file+".mid")


  # plt.show()




'''
for input_seq, label_seq in dataset.take(5):
    print("Input sequence:\n", input_seq.numpy())
    print("Label sequence:\n", label_seq.numpy())
    print()
'''