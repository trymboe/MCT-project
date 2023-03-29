import os
import argparse
import pretty_midi
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

def eval_model(model, key_order, dataset, input_length, vocab_size, temperature, num_predictions, sequence):

    for input_seq, _ in dataset.take(1):
       input_notes = input_seq.numpy()[0]

    generated_notes = np.empty((input_length,26))


    generated_notes[:input_length, :] = input_notes

    print(generated_notes.shape)
    print(input_notes.shape)

    # print(len(np.nonzero(input_notes)[1]))
    # print(len(np.nonzero(generated_notes)[1]))
    print("generated_notes", np.nonzero(generated_notes)[1])
    for i in range(num_predictions):
        if sequence:
            next_note = predict_next_note_sequence(input_notes, model)
            generated_notes = np.concatenate((generated_notes, next_note), axis=0)
            input_notes = next_note

            
        else:
            next_note = predict_next_note(input_notes, model)
            generated_notes[i+input_length] = next_note
            input_notes = np.delete(input_notes, 0, axis=0)
            input_notes = np.append(input_notes, next_note, axis=0)

    generated_notes[generated_notes != 0] = 1


    
    # sample_notes = np.stack([raw_notes[key] for key in key_order], axis=1)

    # input_notes = (
    #     sample_notes[:input_length] / np.array([vocab_size, 1, 1]))

    # generated_notes = []
    # prev_start = 0
    # for _ in range(num_predictions):
    # pitch, step, duration = predict_next_note(input_notes, model, temperature)
    # start = prev_start + step
    # end = start + duration
    # input_note = (pitch, step, duration)
    # generated_notes.append((*input_note, start, end))
    # input_notes = np.delete(input_notes, 0, axis=0)
    # input_notes = np.append(input_notes, np.expand_dims(input_note, 0), axis=0)
    # prev_start = start

    # generated_notes = pd.DataFrame(
    #     generated_notes, columns=(*key_order, 'start', 'end'))

    #generated_notes = normalize_pitch(generated_notes)


    return generated_notes

def predict_next_note(notes: np.ndarray, model: tf.keras.Model, temperature: float = 1.0) -> int:
  """Generates a note IDs using a trained sequence model."""
  assert temperature > 0

  # Add batch dimension
  inputs = tf.expand_dims(notes, 0)

  predictions = model.predict(inputs)
  pitch_logits = predictions['pitch']
  step = predictions['step']
  duration = predictions['duration']

  pitch_logits /= temperature
  pitch = tf.random.categorical(pitch_logits, num_samples=1)
  pitch = tf.squeeze(pitch, axis=-1)
  duration = tf.squeeze(duration, axis=-1)
  step = tf.squeeze(step, axis=-1)

  # `step` and `duration` values should be non-negative
  step = tf.maximum(0, step)
  duration = tf.maximum(0, duration)

  return int(pitch), float(step), float(duration)

def predict_next_note_sequence(notes: np.ndarray, model: tf.keras.Model, temperature: float = 1.0) -> int:
    pass