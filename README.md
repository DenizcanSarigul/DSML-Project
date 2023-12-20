# DSML UNIL_Alpina

Welcome to our GitHub repository,

Here you will find the files which we used for our participation in the text classification challenge "Detecting the difficulty level of French texts".

This ReadMe page is used as report of our journey through this challenge. You will be taken through the steps that we took to get to our results, as well as the difficulties we encountered. 

## Introduction

For this challenge, we were given a list of texts, in French, which were each labelled with their according level of difficulty (A1, A2, B1, B2, C1, C2).  The task of the challenge was to develop a machine learning model, which could predict the level of difficulty for new texts/sentences.  Figure 1, shows an extract of our data.



![image](https://github.com/DenizcanSarigul/DSML-Project/assets/119871445/a96442ec-661b-4fac-b730-89fc6edb1998)

*Figure 1: First 5 values of the list (training data)*




Once we created our model, we were given a test data set on which we could use to find how accurate our model was. This set of data was simply a list of texts/sentences without their labels (level of difficulty). Figure 2, shows an extract of this test data.
 


![image](https://github.com/DenizcanSarigul/DSML-Project/assets/119871445/db1cde80-f61f-4256-b700-3d1f22abdef2)

*Figure 2: First 5 values of the test list*





## 2. Methodology

For this challenge we applied two kinds of techniques of text representation. The first being a series of models using TI-DF representation, and for the second we used the CamamBERT Natural Language Processing model, which is based off the RoBERTa architecture (camembert-model.fr, n.d.). 


## 3. Models

### 3.1 Logistic Regression, K Nearest Neighbours, Decision Tree, Random Forests & Multinominal Naïve Bayes

For this first series of classification methods, we decided to transformed our text using TI-DF representation. As a tokenizer, we defined our own tokenizer function using methods from the Spacy, String, and nltk.stem.snowball libraires. 

Naturally, beforehand we first split our data into 80% train data (X_train) and 20% test data(y_test).
The tokenizer function we used was originally designed for the English language and designed by our Teaching Assistants: Stergios Konstantinidis and Ludovic Mareemootoo. We then modified it to suit the French language. Below, Figure 3 is a snapshot of our tokenizer. 
 


![image](https://github.com/DenizcanSarigul/DSML-Project/assets/119871445/46140ef9-4f06-4b96-bc6d-bfc01fc305de)


*Figure 3: Tokenizer*








To construct our model, we used Pipeline method from the sklearn.pipeline library, in which we inserted our Vectorizer, TI-DF transformer, and classification method.  Figure 4, shows and example of the Pipeline we used. 
 

![image](https://github.com/DenizcanSarigul/DSML-Project/assets/119871445/d07f144b-6232-468d-85b6-571405495a54)

*Figure 4: Pipeline example using Logistic Regression as a classification method*







After conducting a pipeline for each classification method, hence creating a model, we then used it on our unlabelled test data mentioned earlier in the introduction. Our results are presented in the table below. 


---- Table goes here 

As one may note, our results struggle to surpass a test accuracy of 0.5. Therfore, we decided to explore the Natural Learning Processing Model CamamBERT to improve our results. 

### 3.2 CamamBERT
We were briefly introduced to NLP models in class for sentiment analysis, but not multi-label classification. Thus, we researched different ways of implementing such a model and found a YouTube video tutorial on how to use BERT for text classification. The producer of the video also shared a GitHub repository where we could find all the required material. 

- YouTube video link: [Tutorial](https://www.youtube.com/watch?v=TmT-sKxovb0&t=608s)
- GitHub repository link: [GitHub](https://github.com/RajKKapadia/Transformers-Text-Classification-BERT-Blog)

The code provided by Mr. Kapadia was already very detailed and functional. The modifications which we brought were for the processing and testing phase. As the original code was written for a binary classification, we changed what was needed to fit a multi-label classification problem. In addition, we changed the NLP model to CamamBERT, which is the BERT model adapted to the French language. The model is found in the transformers library from HuggingFace.

For this model, the training is done in one file, and the testing in a separate one. 
#### 3.2.1 Training 

Essentially, the important aspects of the code are the processing of the data, where the cleaning is done, and the training arguments. 
We start by defining our tokenizer and then the processing begins. Figure 5, is a snapshot of how the data is cleaned and, encoded and labelled. 

 ![image](https://github.com/DenizcanSarigul/DSML-Project/assets/119871445/7d78cf10-1c00-493f-a7f1-066f2c90bdfb)

*Figure 6: Processing of the data*











We then store the encodings into an array and create a new Data Frame. The data is then split, and is then ready to be used for the training.
The model is defined using AutoModelForSequenceClassification method, and naturally we insert the ‘camambert-base’). Before training, et define our training arguments. For this model, use the epoch method for the evaluation method. Once that’s done, we then define the trainer and run the training.

![image](https://github.com/DenizcanSarigul/DSML-Project/assets/119871445/46d4c49c-9d39-458a-b47d-5ba3b2e1ed08)

*Figure 7: Defining the model, training arguments, and trainer*





