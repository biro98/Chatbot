# Chatbot
This is a fun project which attempt to use pytorch to create an easy example of a Chatbot.

The project is based on the Kaggle project at : https://www.kaggle.com/code/himanshukumar7079/pre-determined-chatbot-ai

The data used is taken from Kaggle at: https://www.kaggle.com/datasets/grafstor/simple-dialogs-for-chatbot

Structure:
1) Dataset_and_processing.ipynb:
Set up and Preprocessing: It sets a random seed for reproducibility in PyTorch and defines functions to preprocess text data. This includes converting Unicode to ASCII, cleaning and standardizing text (like expanding contractions), and removing punctuation.

Custom Dataset Class: Defines a DialogDataset class for handling dialog data. This class tokenizes questions and answers, converts them into numerical representations using vocabularies, and formats them for PyTorch models.

Data Loading and Processing: The script loads dialogue data from a file, preprocesses the text, and splits it into training and validation sets. It then builds vocabularies from the training data.

DataLoader Creation: It creates PyTorch DataLoader instances for both the training and validation datasets to facilitate batch processing during model training.

Batch Processing Check: There is a section for checking the shape of batches from the DataLoader.

Saving Resources: The script saves the source and target vocabularies, as well as the preprocessed training and validation data, using the pickle module for later use.


2) model.ipynb:
Environment Setup: It sets a random seed for reproducibility and specifies the device (CPU in this case) for running the model.

Vocabulary Loading: The script loads pre-built source (src_vocab) and target (tgt_vocab) vocabularies from saved files. These vocabularies are likely used for tokenizing sentences into words or subwords.

Model Parameters: It defines parameters for the neural network such as batch size, embedding dimension, and the size of the GRU (Gated Recurrent Unit) layers.

Encoder: The Encoder class is defined as a part of the seq2seq model. It uses an embedding layer to convert input tokens into vectors and a GRU layer to process these embeddings sequentially. The encoder processes the input sequence and condenses information into a context state.

Attention Mechanism: The BahdanauAttention class implements the attention mechanism, which helps the model to focus on specific parts of the input sequence when predicting each word of the output sequence. This is crucial for handling long input sequences and improving the quality of the model's predictions.

Decoder: The Decoder class uses the context state provided by the encoder and applies attention to specific parts of the input sequence. It then generates the output sequence one token at a time.

Attention in Decoder: The decoder's forward pass computes the attention weights using the current hidden state and the encoder output. It then forms a context vector as a weighted sum of the encoder outputs, which is used along with the input token to generate the next token in the sequence.

3) Training_Utilities.ipynb:
Importing Modules and Scripts: It imports necessary PyTorch modules and two other scripts (Dataset_and_preprocessing.ipynb and model.ipynb) which likely contain the dataset preparation and the model architecture.

Initializing Model Components: It initializes the encoder, attention layer, and decoder parts of the seq2seq model, and sets the model to use the CPU as the computing device.

Optimizer: An Adam optimizer is created, combining parameters from both the encoder and decoder for optimization.

Custom Loss Function: A loss function is defined specifically for this task. It calculates the cross-entropy loss, applying a mask to ignore padding tokens in the target sequences.

Training Step Function: This function, train_step, takes input sequences, target sequences, and an initial hidden state for the encoder. It performs a forward pass through the model and calculates the loss for each timestep in the target sequence.

Training Loop: The script then runs a training loop for a specified number of epochs. In each epoch, it iterates over the training data, computes the loss for each batch using train_step, and performs backpropagation. After every few epochs, it prints the average loss.

Saving the Model: Finally, the trained state dictionaries of both the encoder and decoder are saved to files. This allows for the model to be reloaded later for further training or inference.

4) Evaluate.ipynb:
   
Module Imports and Data Loading: The script imports necessary libraries and loads previously trained model components (encoder and decoder) and data (train_data and val_data) using PyTorch and Pandas.

Model Initialization: It initializes the encoder, attention layer, and decoder components of a sequence-to-sequence (seq2seq) model with an attention mechanism. These components are loaded with pre-trained weights.

Evaluation Function: The evaluate function is defined to process a given sentence (question) through the seq2seq model to generate a response. It involves preprocessing the input, passing it through the encoder and decoder, and constructing a response.

Utility Functions: Additional functions are defined for processing text, such as unicode_to_ascii and clean_text, which standardize the input text by converting it to ASCII, making it lowercase, removing punctuation, and handling various textual contractions.

Interactive Function: The interact_with_model function allows for interactive usage of the model. Users can input a question, and the model generates a response based on the trained seq2seq model.

User Interaction Loop: The script includes a loop that prompts the user to input questions and then displays the model-generated answers. The loop continues until the user types 'exit'.

Example Usage: There's an ask function provided as an example of how to use the evaluate function, demonstrating the script's application in generating answers to questions.

chatboot_full_code_trial.ipynb was a trial to get the function to work properly before finalizing the project.
