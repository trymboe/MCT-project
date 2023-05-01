import absl.logging
import matplotlib.pyplot as plt

from model import create_model, create_model_sequence, train_model
from data_process import prepare_data, notes_to_midi
from generate import eval_model

absl.logging.set_verbosity(absl.logging.ERROR)

seed = 42
# tf.random.set_seed(seed)
# np.random.seed(seed)

EPOCHS = 100
MIDI_INSTRUMENT = "Acoustic Grand Piano"
LEARNING_RATE = 0.005
VOCAB_SIZE = 128
BATCH_SIZE = 64
KEY_ORDER = ['pitch', 'step', 'duration']
VALIDATION_SIZE = 0.15

NUM_PREDICTIONS = 50
INPUT_LENGTH = 25

if __name__ == "__main__":

  train = False
  sequence = False
  big_model = True
  dataset = "small"
  optimizer= "Adam"
  if not sequence and not big_model:
    model_name = "model1"
  elif sequence and not big_model:
    model_name = "model2"
  elif not sequence and big_model:
    model_name = "model3"
  elif sequence and big_model:
    model_name = "model4"

  load_model_path = f'models/{model_name}/{dataset}/e_{EPOCHS}_{INPUT_LENGTH}_{optimizer}'
  out_file = f"results/melody/{model_name}/{dataset}/e_{EPOCHS}_{INPUT_LENGTH}_{optimizer}"

  if sequence:
    if train:
      train_ds, val_ds = prepare_data(f"data/melody/{dataset}", INPUT_LENGTH, INPUT_LENGTH, VOCAB_SIZE, VALIDATION_SIZE, BATCH_SIZE)
    else:
      train_ds, val_ds = prepare_data(f"data/melody/xx_small", INPUT_LENGTH, INPUT_LENGTH, VOCAB_SIZE, VALIDATION_SIZE, BATCH_SIZE)
    model, loss, _ = create_model_sequence(INPUT_LENGTH, LEARNING_RATE, optimizer, model_name)
  else:

    train_ds, val_ds = prepare_data(f"data/melody/{dataset}", INPUT_LENGTH, 1, VOCAB_SIZE, VALIDATION_SIZE, BATCH_SIZE)

    model, loss, _ = create_model(INPUT_LENGTH, LEARNING_RATE, optimizer, model_name)

  if not train:
    model.load_weights(load_model_path)
  else:
      train_model(model, train_ds, val_ds, f"models/{model_name}/{dataset}/e_{EPOCHS}_{INPUT_LENGTH}_{optimizer}", EPOCHS)

  if not train:

    generated_notes = eval_model(model, train_ds, INPUT_LENGTH, VOCAB_SIZE, NUM_PREDICTIONS, sequence, KEY_ORDER)
    pm = notes_to_midi(generated_notes, out_file, MIDI_INSTRUMENT)

    pm.write(out_file+".mid")
  plt.show()


"""
    for input_seq, label in seq_ds.take(1):
        print(input_seq.shape)
        print(label.shape)
"""