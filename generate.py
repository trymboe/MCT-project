import numpy as np
import pandas as pd
import tensorflow as tf


def eval_model(model, dataset, input_length, vocab_size, num_predictions, sequence, key_order):
    """
    This function takes in a trained model, a dataset, input length, vocabulary size, number of predictions, a boolean indicating whether to use sequence generation or not, and a key order. It generates new musical notes based on the given parameters and returns a pandas dataframe containing the generated notes with their corresponding pitch, step, duration, start, and end times.

    Parameters:
        model: a trained TensorFlow model
        dataset: a tensorflow dataset object containing the input sequences
        input_length: an integer representing the length of the input sequence
        vocab_size: an integer representing the size of the vocabulary
        num_predictions: an integer representing the number of notes to generate
        sequence: a boolean representing whether to use sequence generation or not
        key_order: a tuple representing the order of the columns in the returned dataframe
    
    Returns:
        generated_notes_pd: a pandas dataframe containing the generated notes with their corresponding pitch, step, duration, start, and end times.
    """

    for input_seq, _ in dataset.take(1):
       input_notes = input_seq.numpy()[0]

    generated_notes = np.empty((input_length,3))

    generated_notes[:input_length, :] = input_notes


    for i in range(num_predictions):
        if sequence:
            next_note = predict_next_note_sequence(input_notes, model)[0]
            generated_notes = np.concatenate((generated_notes, next_note), axis=0)
            input_notes = next_note

            
        else:
            next_note = (predict_next_note(input_notes, model))
            generated_notes = np.vstack((generated_notes, next_note))

            input_notes = np.delete(input_notes, 0, axis=0)
            input_notes = np.append(input_notes, [next_note], axis=0)


    generated_notes_full = []
    prev_start = 0

    print("input_notes:")
    for idx, note in enumerate(generated_notes):
        pitch, step, duration = note
        pitch *= vocab_size
        pitch = int(pitch)
        start = prev_start + step
        end = start + duration
        full_note  = [pitch, step, duration, start, end]
        generated_notes_full.append(full_note)
        prev_start = start
        print(pitch, end=' ')
        if idx+1 == input_length:
            print("\ngenerated_notes:")


    generated_notes_pd = pd.DataFrame(generated_notes_full, columns=(*key_order, 'start', 'end'))

    return generated_notes_pd

def predict_next_note(notes: np.ndarray, model: tf.keras.Model, temperature: float = 1.0) -> float:
    """
    The predict_next_note function generates a note IDs using a trained sequence model.

    Args:
        notes (np.ndarray): The sequence of notes used as input for the model. A 3D numpy array with dimensions (sequence_length, 3).
        model (tf.keras.Model): The trained sequence model used for prediction.
        temperature (float): The temperature parameter used for sampling from the model distribution. Default is 1.0.
        
    Returns:
        A list of three floating point values representing the predicted pitch, step, and duration of the next note in the sequence.
    """
    assert temperature > 0

    # Add batch dimension
    inputs = tf.expand_dims(notes, 0)

    predictions = model.predict(inputs)[0][0]


    pitch = predictions[0]
    step = predictions[1]
    duration = predictions[2]
    

    #values should be non-negative
    step = np.maximum(0, step)
    duration = np.maximum(0, duration)
    pitch = np.maximum(0, pitch)

    return [float(pitch), float(step), float(duration)]

#Discontinued
def predict_next_note_sequence(notes: np.ndarray, model: tf.keras.Model, temperature: float = 1.0) -> int:
    inputs = tf.expand_dims(notes, 0)
    predictions_logits = model.predict(inputs)



    return predictions_logits