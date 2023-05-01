# MCT-project
Individual project in MCT4052-"Music and machine learning" at UiO

# Usage
all parameters that needs to be adjusted are in the main.py file.

# Parameters

## Primary parameters
**train:** True if you want to train a model, False if you want to evaulate a model (generate MIDI)

**EPOCHS:** The number of epochs the model shall be trained on.

**dataset:** What dataset the model should be trained on. \
"full"    : 909 songs \
"small"   : 90 songs (10% of full) \
"x_small  : 30 songs \
"xx_small": 15 songs \
"test"    : 1 song

**big_model** If True, this wll train/evalute using the big model. If False it will train/evaluate the small model.

**INPUT_LENGHT:** The number of notes/timesteps a model will get before predicting the next note(s)/timestep(s).

**NUM_PREDICTIONS:** The number of notes/timesteps to be generated. Only for evaluation of model. 

**TEMPERATURE:** How much randomness should be used when generating melodies. Between 0 and 1. Default 0.5


## Secondary parameters
These parameters can be set as is, and are used when experimenting with different methods.

**sequence:** True if you want the train or evaluate a model that predicts a sequence of notes/timesteps. The predicted sequence is if equal length to the input length of the model. A model trained on sequence can only predict sequences, and vice versa. Default False.


**FS:** The sampling rate for piano roll sampling. Default 12.


## Training and evaluation
After setting the parameters to your likeing, you can train a model, and the model will be saved. If you want to evaluate this exact model, you just switch "train" to False, and remain the exact same parameters, and the model will be evaluated.

Each model is saved so that it knows what parameters it was trained on, and when evaluating, the parameters needs to be the same for the right model to be loaded.

NOTE: The first notes are not generated, but the input to the model

## Pretrained models
There are some pretrained models in the models directory.
To run one of them, set these parameter:
train        : False
sequence     : False
big_model    : True
dataset      : "small"
EPOCHS       : 100
INPUT_LENGTH : 40

The results can be found in results/melody/model3/small


# File structure
The file structure is set up to make running and evaluation as flawless as possible.

All the training data is in the data directory, and you can choosing what dataset size to use with the dataset parameter.

The models are saved in the models folder directory, under either \
**model1** (big_model = False, squence = False)\
**model2** (big_model = False, squence = True)\
**model3** (big_model = True, sequence = False)\
**model4** (big_model = True, sequence = True)

The results after evaluating a model are saved in the results directory with a automatic name generation to specify what model the result is evaluated from.
