import os
import sys
import pretty_midi
import collections
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

def eval_model(model, dataset, input_length, num_predictions=120):


    for input_seq, _ in dataset.take(1):
        input_notes = input_seq.numpy()[0]


    generated_notes = np.empty((num_predictions+input_length,128))

    generated_notes[:input_length, :] = input_notes

    for i in range(num_predictions):
        next_note = predict_next_note(input_notes, model)
        next_note[next_note != 0] = 127
        generated_notes[i+input_length] = next_note

        input_notes = np.delete(input_notes, 0, axis=0)
        input_notes = np.append(input_notes, (next_note), axis=0)

    return generated_notes


def predict_next_note(notes: np.ndarray, model: tf.keras.Model, temperature: float = 2.0) -> int:
    """Generates a note IDs using a trained sequence model."""
    assert temperature > 0

    # Add batch dimension
    inputs = tf.expand_dims(notes, 0)

    predictions_logits = model.predict(inputs)

    return predictions_logits