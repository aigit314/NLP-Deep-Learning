# NLP-Deep-Learning
The goal of the project is the Sequential Sentence Classification in Medical Abstracts using Pubmed 200k RCT dataset.
I've used Deep Learining Large Language Model like Bidirectional Encoder Representations From Transformers (BERT) to classify each sentence to one of the 5 classes: ['OBJECTIVE', 'METHODS', 'RESULTS', 'CONCLUSIONS', 'BACKGROUND']

**Why Bidirectional Encoder Representations From Transformers (BERT)?**

With LSTM/GRU, data can only be read in a sequential manner in one direction and they can not capture the local context of a word in a sentence. This is where BERT come into play. It can process the whole text document in parallel, relying on
attention mechanism. In the attention mechanism, instead of processing text sequentially, the text is
processed in parallel, which allows the attention system to assign weightage to important parts of the
text in a parallel manner.
BERT models are able to generate word representations that capture local context.
For example, with word embeddings, the representation of the word “Jaguar” will be the same in the
following two sentences.
“I bought a Jaguar from a car dealer.” “I saw a Jaguar in a zoo.”
However, with BERT, two different word representations of the word “Jaguar” will be generated that
capture the local context.
Since I’m dealing with Sequential Sentence Classification task and I think context plays an important
role in any sentence and a pre-trained BERT model is avaiable that has been already trained on
Sequential Sentence Classification task (with huge amount of data), I chose BERT as a Deep Learning
algorithm to solve the problem.

I’ve fine-tuned pre-trained BERT on both GPU and TPU at google colab.

**Steps of the solution:**
1. Preprocessing of Data: Cleaned the text, Removed Tags, and converted the target variable to
 five numerical labels y:{0, 1, 2, 3, 4, 5}
2. Calculated word count per sentence distribution (Fig.2) on training data (helped me to set max
input tokens/words=100)
3. Calculated target (y) distributions in the training and development set.
4. Created training, validation (development) and test datasets
5. Generate BERT tokens using BERT tokenizer (with max input token/words=100) and convert texts
into the input format that can be used by a BERT model.
7. Convert the input data into tensors.
8. Load a Pre-trained text classification model and defined the followings:
optimizer = Adam(learning_rate=3e-5, epsilon=1e-08)
loss = SparseCategoricalCrossentropy(from_logits=True)
metric= SparseCategoricalAccuracy('accuracy')
9. Training model on training dataset (64,000 training samples while training on GPU and 300,000 training samples while training on TPU) and validate on validation dataset with multiple epochs and
saving the weights of the Best model.
10. Evaluate Model Performance (classification_report, confusion_matrix, accuracy_score) using
the best saved model on Test Dataset.

