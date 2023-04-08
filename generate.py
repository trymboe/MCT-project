import numpy as np
import tensorflow as tf

def eval_model(model, dataset, input_length, num_predictions=120, sequence=False, temp=0):


    for input_seq, _ in dataset.take(1):
        input_notes = input_seq.numpy()[0]

    
    generated_notes = np.empty((input_length,26))
    if not sequence:
        generated_notes = np.empty((input_length+num_predictions,26))

    generated_notes[:input_length, :] = input_notes

    print(generated_notes.shape)
    print(input_notes.shape)

    # print(len(np.nonzero(input_notes)[1]))
    # print(len(np.nonzero(generated_notes)[1]))
    # print("generated_notes", np.nonzero(generated_notes)[1])
    for i in range(num_predictions):
        if sequence:
            next_note = predict_next_note_sequence(input_notes, model)
            generated_notes = np.concatenate((generated_notes, next_note), axis=0)
            input_notes = next_note

            
        else:
            next_note = predict_next_note(input_notes, model, temp)
            generated_notes[i+input_length] = next_note
            input_notes = np.delete(input_notes, 0, axis=0)
            input_notes = np.append(input_notes, next_note, axis=0)

    generated_notes[generated_notes != 0] = 1
    # print("generated_notes", np.nonzero(generated_notes)[1])
    return generated_notes

def predict_next_note(notes: np.ndarray, model: tf.keras.Model, temperature: float) -> int:
    """Generates a note IDs using a trained sequence model."""
    assert temperature >= 0

    # Add batch dimension
    inputs = tf.expand_dims(notes, 0)

    predictions_logits = model.predict(inputs)
    max_idx = get_index(predictions_logits, temperature)
    print("max_idx", max_idx)
    # Create a new array with 1 at the index of the maximum value
    next_note = np.zeros_like(predictions_logits)
    next_note[0][max_idx] = 1

    return next_note

def predict_next_note_sequence(notes: np.ndarray, model: tf.keras.Model, temperature: float = 0) -> int:
    """Generates a note IDs using a trained sequence model."""
    assert temperature >= 0

    # Add batch dimension
    inputs = tf.expand_dims(notes, 0)

    predictions_logits = np.squeeze(model.predict(inputs), axis=0)

    # index = get_index(predictions_logits, temperature)

    max_idx = np.argmax(predictions_logits, axis=1)

    print("max_idx",max_idx)
    predictions = np.zeros_like(predictions_logits)
    predictions[np.arange(predictions_logits.shape[0]), max_idx] = 1


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


    # return max_indices
    
    sorted_indices = np.argsort(prediction_logits)[0]
    # print(sorted_indices)
    # Choose the first index with probability p, and the second index with probability 1-p
    print(sorted_indices[-1], sorted_indices[-2], sorted_indices[-3])
    if np.random.random() > epsilon:
        return sorted_indices[-1]
    else:
        if np.random.random() > epsilon:
            return sorted_indices[-2]
        else:
            if np.random.random() > epsilon:
                return sorted_indices[-3]
            else:
                return sorted_indices[-4]