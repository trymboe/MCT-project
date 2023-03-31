# MCT-project
Individual project in MCT4052-"Music and machine learning" at UiO

Note: This project is still WIP

# Usage
For now, all parameters needs to be adjusted in the main.py file.

## Parameters
**train:** True if you want to train a model, False if you want to evaulate a model (generate MIDI)

**sequence:** True if you want the train or evaluate a model that predicts a sequence of notes/timesteps. The predicted sequence is if equal length to the input length of the model. A model trained on sequence can only predict sequences, and vice versa.

**INPUT_LENGHT:** The number of notes/timesteps a model will get before predicting the next note(s)/timestep(s). If sequence is True, this will also be the number of notes the model predicts.

**NUM_PREDICTIONS:** The number of notes/timesteps to be generated. If the model predicts a sequence, this is the number of sequences that will be generated. Only for evaluation of model. 

**EPOCHS:** The number of epochs the model shall be trained on.

**dataset:** What dataset the model should be trained on.
"full"    : 909 songs
"small"   : 90 songs (10% of full)
"x_small  : 30 songs 
"xx_small": 15 songs
"test"    : 1 song

**model_name:** What model is being trained/evaluated. This dictates where the model will be saved, or where the model will be loaded from, and is used to distigush between sequence models and non-sequence models
"model1" : non-sequence model
"model2" : sequence model

## Training and evaluation
After setting the parameters to your likeing, you can train a model, and the model will be saved. If you want to evaluate this exact model, you just switch "train" to False, and remain the exact same parameters, and the model will be evaluated.

Each model is saved so that it knows what parameters it was trained on, and when evaluating, the parameters needs to be the same for the right model to be loaded.

NOTE: The first notes are not generated, but the input to the model


# File structure
The file structure is set up to make running and evaluation as flawless as possible.

All the training data is in the data directory, and you can choosing what dataset size to use with the dataset parameter.

The models are saved in the models folder directory, under either model1 or model2

The results after evaluating a model are saved in the results directory with a automatic name generation to specify what model the result is evaluated from.
