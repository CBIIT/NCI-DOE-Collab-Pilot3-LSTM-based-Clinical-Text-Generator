## Model Description

The LSTM-based Clinical Text Generator trains a Long Short-Term Memory, or LSTM, recurrent neural network, using Keras, a deep learning library, on a given sample corpus of biomedical text (such as pathology reports). You can then use the trained model to synthesize text documents similar in context to the sample corpus.

Some of the parameters you can adjust include: 
 * Number of iterations for training the model
 * Number of layers in the LSTM model
 * Variability of text synthesis
 * Length of the synthesized text

## Setup

To set up the Python environment needed to train and run this model:
1. Install [conda](https://docs.conda.io/en/latest/) package manager. 
2. Clone this repository. 
3. Create the environment as shown below.

```bash
   conda env create -f environment.yml -n P3B2
   conda activate P3B2
```

To download the preprocessed data needed to train and test the model:
1. Create an account on the Model and Data Clearinghouse ([MoDaC](https://modac.cancer.gov)). 
2. Follow the instructions in the Training section below.
3. When prompted by the training script, enter your MoDaC credentials.

## Training

To train the model:

1. (Optional) Modify the parameters of the model or the sample text in [p3b2_default_model.txt](https://github.com/CBIIT/NCI-DOE-Collab-Pilot3-RNN-LSTM-based-Clinical-Text-Generator/blob/master/Pilot3/P3B2/p3b2_default_model.txt). 

   |	Parameter	|	Description	|
   |	-------------	|	-------------	|
   |	rnn_size	|	Size of LSTM internal state (int)|
   |	n_layers	|	Number of layers in the LSTM model (int)	|
   | epochs | Number of training iterations (int) |
   | learning_rate | Learning rate (float) |
   | dropout | Fraction of units to drop for the linear transformation of the inputs (float)|
   |recurrent_dropout | Fraction of units to drop for the linear transformation of the recurrent state (float)|
   | loss | Loss function to use|
   | activation | Activation function to use|
   | optimizer | Name of optimizer to use|
   |	do_sample	|	Generate sample of synthesized text	(boolean) |
   |	temperature	|	Variability of text synthesis (float)	|
   |	primetext	|	Source type of text synthesis (string)	|
   |	length	|	Length of synthesized text	(int) |


2. Run [p3b2_baseline_keras2.py](https://github.com/CBIIT/NCI-DOE-Collab-Pilot3-RNN-LSTM-based-Clinical-Text-Generator/blob/master/Pilot3/P3B2/p3b2_baseline_keras2.py). 

   ```
   python p3b2_baseline_keras2.py
   ```

This script performs the following tasks:
 * Downloads and uncompresses the preprocessed data file from MoDaC.
 * Prepares the training and testing data.
 * Builds the LSTM model.
 * Trains the model.
 * Generates example text output files.

For every epoch of the training process, the script outputs the model architecture and configuration in a JSON file and the model weights in an H5 file within the output folder. 

The following paragraph demonstrates an example model (in JSON format):

```
{"class_name": "Sequential", "keras_version": "1.1.0", "config": [{"class_name": "LSTM", "config": {"inner_activation": "hard_sigmoid", "trainable": true, "inner_init": "orthogonal", "output_dim": 256, "unroll": false, "consume_less": "cpu", "init": "glorot_uniform", "dropout_U": 0.0, "input_dtype": "float32", "batch_input_shape": [null, 20, 99], "input_length": null, "dropout_W": 0.0, "activation": "tanh", "stateful": false, "b_regularizer": null, "U_regularizer": null, "name": "lstm_1", "go_backwards": false, "input_dim": 99, "return_sequences": false, "W_regularizer": null, "forget_bias_init": "one"}}, {"class_name": "Dense", "config": {"W_constraint": null, "b_constraint": null, "name": "dense_1", "activity_regularizer": null, "trainable": true, "init": "glorot_uniform", "bias": true, "input_dim": null, "b_regularizer": null, "W_regularizer": null, "activation": "linear", "output_dim": 99}}, {"class_name": "Activation", "config": {"activation": "softmax", "trainable": true, "name": "activation_1"}}]}
```

In addition, the script generates a sample text file after each epoch and stores it as ```example_<epoch>_<model-loss-value>.txt``` in the same output folder with the JSON and H5 files. 

The following paragraph demonstrates an example output text file:

```
----- Generating with seed: "Diagnosis"
                    DiagnosisWZing Pathology Laboratory is certified under this report. **NAME[M. SSS dessDing Adientation of the tissue is submitted in the same container labeled with the patient's name and designated 'subcarinal lymph node is submitted in toto in cassette A1. B. Received in formalin labeled "right lower outer quadrant; A11-A10 - slice 16 with a cell block and submitted in cassette A1. B. Received fresh for
```

## Acknowledgments
   
This work has been supported in part by the Joint Design of Advanced Computing Solutions for Cancer (JDACS4C) program established by the U.S. Department of Energy (DOE) and the National Cancer Institute (NCI) of the National Institutes of Health.
