import os
import sys
import numpy as np
import tensorflow as tf

from data_process import prepare_data, relative_pitch_to_pretty_midi
from generate import eval_model
from model import create_model, create_model_sequence, train_model

np.set_printoptions(threshold=sys.maxsize)


BATCH_SIZE = 64
NUM_PREDICTIONS = 150
VALIDATION_SIZE = 0.15
LEARNING_RATE = 0.005
VOCAB_SIZE = 26
EPOCHS = 100
TEMPERATURE = 0.5
INPUT_LENGTH = 40
LABEL_LENGTH = 1
OPTIMIZER = "Adam"
# 120 bpm, 2 bps, 3*2 (represent triplets), 6*2 (nyqvist rate)
FS = 12

if __name__  == "__main__":
  train = True
  sequence = False
  dataset = "small"
  model_name = "model3"

  gb = 8

  os.environ['TF_DEVICE']='/gpu:0'
  gpus = tf.config.list_physical_devices('GPU')
  if gpus:
      gpu = gpus[0]
      # Limit GPU memory to n*1GB
      tf.config.experimental.set_virtual_device_configuration(gpu, [
          tf.config.experimental.VirtualDeviceConfiguration(memory_limit=gb*1024)
      ])

  
  load_model_path = f'models/{model_name}/{dataset}/e_{EPOCHS}_{INPUT_LENGTH}_{OPTIMIZER}'
  out_file = f"results/melody/{model_name}/{dataset}/e_{EPOCHS}_{INPUT_LENGTH}_{OPTIMIZER}"

  if sequence:
    if train:
      train_ds, val_ds = prepare_data(f"data/melody/{dataset}", INPUT_LENGTH, INPUT_LENGTH, FS, VALIDATION_SIZE, BATCH_SIZE)
    else:
       train_ds, val_ds = prepare_data(f"data/melody/test", INPUT_LENGTH, INPUT_LENGTH, FS, VALIDATION_SIZE, BATCH_SIZE)
    model, loss, optimizer = create_model_sequence(INPUT_LENGTH, LEARNING_RATE, OPTIMIZER)
  else:
    if train:
      train_ds, val_ds = prepare_data(f"data/melody/{dataset}", INPUT_LENGTH, LABEL_LENGTH, FS, VALIDATION_SIZE, BATCH_SIZE)
    else:
      train_ds, val_ds = prepare_data(f"data/melody/test", INPUT_LENGTH, LABEL_LENGTH, FS, VALIDATION_SIZE, BATCH_SIZE)


    model, loss, optimizer = create_model(INPUT_LENGTH, LEARNING_RATE, OPTIMIZER)

  if not train:
    model.load_weights(load_model_path)
  else:
      train_model(model, train_ds, val_ds, f"models/{model_name}/{dataset}/e_{EPOCHS}_{INPUT_LENGTH}_{OPTIMIZER}", EPOCHS)

  if not train:
    generated_notes = eval_model(model, train_ds, INPUT_LENGTH, num_predictions=NUM_PREDICTIONS, sequence=sequence, temp=TEMPERATURE)
    pm = relative_pitch_to_pretty_midi(generated_notes, FS)

    pm.write(out_file+".mid")


  # plt.show()

'''
for input_seq, label_seq in dataset.take(5):
    print("Input sequence:\n", input_seq.numpy())
    print("Label sequence:\n", label_seq.numpy())
    print()
'''
