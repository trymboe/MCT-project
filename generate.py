import os
import sys
import pretty_midi
import collections
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

def eval_model(model, dataset, input_length, num_predictions=120, sequence=False):


    for input_seq, _ in dataset.take(1):
        input_notes = input_seq.numpy()[0]


    generated_notes = np.empty((num_predictions + input_length,128))

    generated_notes[:input_length, :] = input_notes

    for i in range(num_predictions):
        if sequence:
            next_note = predict_next_note(input_notes, model)
            generated_notes = np.concatenate((input_notes, next_note), axis=0)
            input_notes = np.delete(input_notes, 0, axis=0)
            break

            
        else:
            next_note = predict_next_note(input_notes, model)
            generated_notes[i+input_length] = next_note
            input_notes = np.delete(input_notes, 0, axis=0)
            input_notes = np.append(input_notes, (next_note), axis=0)

    generated_notes[generated_notes != 0] = 127
    
    return generated_notes


def predict_next_note(notes: np.ndarray, model: tf.keras.Model, temperature: float = 0) -> int:
    """Generates a note IDs using a trained sequence model."""
    assert temperature >= 0

    # Add batch dimension
    inputs = tf.expand_dims(notes, 0)

    predictions_logits = np.squeeze(model.predict(inputs), axis=0)

    index = get_index(predictions_logits, temperature)

    max_idx = np.argmax(predictions_logits, axis=1)

    print(max_idx)
    
    predictions = np.zeros_like(predictions_logits)
    predictions[np.arange(predictions_logits.shape[0]), index] = 1
    print(predictions.shape)

    return predictions


def get_index(prediction_logits, epsilon):
    # Create an array of random values to compare with epsilon
    rand_vals = np.random.rand(prediction_logits.shape[0])

    # Create an array of indices from 0 to 127
    indices = np.arange(prediction_logits.shape[1])

    # Determine whether to choose the index greedily or randomly
    greedy = rand_vals > epsilon

    # Get the index with the highest probability for each of the 120 distributions
    max_indices = np.argmax(prediction_logits, axis=1)

    # Choose the index either greedily or randomly
    chosen_indices = np.where(greedy, max_indices, np.random.choice(indices, size=prediction_logits.shape[0]))

    print(chosen_indices)
    return chosen_indices
