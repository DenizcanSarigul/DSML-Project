# DSML
UNIL_Alpina

Welcome to our GitHub repository for our text classification challenge.
This page is the report of our journey through this challenge. This  report will you take you through the steps that we took to get to our results, as well as the difficulties. 

1. Introduction
For this challenge, we were given a list of texts, in French, which were each labelled with their according level of difficulty (A1, A2, B1, B2, C1, C2).  The task of the challenge was to develop a machine learning model, which could predict the level of difficulty for new texts/sentences.  Figure 1, shows an extract of our data.


Figure 1, first 5 values of the list








Once we created our model, we were given a test data set on which we could use to find how accurate our model was. This set of data was simply a list of texts/sentences without their labels (level of difficulty). Figure 2, shows an extract of this test data.
 
Figure 2, first 5 values of the test list








2. Methodology
For this challenge we applied two kinds of techniques of text representation. The first being a series of models using TI-DF representation, and for the second we used the CamamBERT Natural Language Processing model, which is based off the RoBERTa architecture (camembert-model.fr, n.d.). 


3. Models
3.1 Logistic Regression, K Nearest Neighbours, Decision Tree, Random Forests & Multinominal Naïve Bayes
For this first series of classification methods, we decided to transformed our text using TI-DF representation. As a tokenizer, we defined our own tokenizer function using methods from the Spacy, String, and nltk.stem.snowball libraires. 
Naturally, beforehand we first split our data into 80% train data (X_train) and 20% test data(y_test).
The tokenizer function we used was originally designed for the English language and designed by our Teaching Assistants: Stergios Konstantinidis and Ludovic Mareemootoo. We then modified it to suit the French language. Below, Figure 3 is a snapshot of our tokenizer. 
 
Figure 3, tokenizer












To construct our model, we used Pipeline method from the sklearn.pipeline library, in which we inserted our Vectorizer, TI-DF transformer, and classification method.  Figure 4, shows and example of the Pipeline we used. 
 
Figure 4, Pipline example using Logistic Regression as classification method.








After conducting a pipeline for each classification method, hence creating a model, we then used it on our unlabelled test data mentioned earlier in the introduction. Our results are presented in the table below. 


---- Table goes here 

As one may note, our results struggle to surpass a test accuracy of 0.5. Therfore, we decided to explore the Natural Learning Processing Model CamamBERT to improve our results. 

 3.2 CamamBERT
We were briefly introduced to NLP models in class for sentiment analysis, but not multi-label classification. Thus, we researched different ways of implementing such a model and found a YouTube video tutorial on how to use BERT for text classification. The producer of the video also shared a GitHub repository where we could find all the required material. 
The following link is the YouTube video, https://www.youtube.com/watch?v=TmT-sKxovb0&t=608s and this one is the link to the GitHub repository, https://github.com/RajKKapadia/Transformers-Text-Classification-BERT-Blog.
The code provided by Mr. Kapadia was already very detailed and functional. The modifications which we brought were for the processing and testing phase. As the original code was written for a binary classification, we changed what was needed to fit a multi-label classification problem. In addition, we changed the NLP model to CamamBERT, which is the BERT model adapted to the French language. The model is found in the transformers library from HuggingFace.
For this model, the training is done in one file, and the testing in a separate one. 
Training 3.2.1
Essentially, the important aspects of the code are the processing of the data, where the cleaning is done, and the training arguments. 
We start by defining our tokenizer and then the processing begins. Figure 5, is a snapshot of how the data is cleaned and, encoded and labelled. 
 
Figure 6, processing of the data











We then store the encodings into an array and create a new Data Frame. The data is then split, and is then ready to be used for the training.
The model is defined using AutoModelForSequenceClassification method, and naturally we insert the ‘camambert-base’). Before training, et define our training arguments. For this model, use the epoch method for the evaluation method. Once that’s done, we then define the trainer and run the training. 
 
Figure 7, defining the model, training arguments and trainer.






