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

For this challenge we applied two kinds of techniques of text representation. The first being a series of models using TI-DF representation and classification with conventional models such as Logistic Regression, and for the second we used the CamamBERT Natural Language Processing model, which is based off the RoBERTa architecture. 


## 2. Results
The table below shows the best results for each of our models. You will notice that for the models K Nearest Neighbours and Random Forests, we did not obtain a test accuracy as we did not submit our results. Due to the low training accuracy results, we did not consider submitting our predictions as we knew that the test accuracy would be low.

We started the project by exploring all possibilites with the conventional classificataion models seen in class, but quickly realised that achieving a high level of accuracy would be challenging. Therefore, we decided to explore the avenue of NLP models and found that Camambert was well suited for our project, and eventually generated a better score of 0.581.

## Table of our best Results

| Model                  | Train Accuracy | Train Precision | Train Recall | Train F1-Score | Kaggle Test Accuracy |
|--------------------------|----------------|------------------|--------------|------------------|-----------------------|
| Logistic Regression      | 0.451          | 0.443            | 0.449        | 0.442            | 0.465                |
| kNN                      | 0.291          | 0.380            | 0.286        | 0.261            | -                      |
| Decision Tree            | 0.302          | 0.297            | 0.300        | 0.295            | 0.341                |
| Random Forests           | 0.321          | 0.364            | 0.317        | 0.285            |--                   |
| MultinominalNaiveBayes  | 0.454          | 0.469            | 0.452        | 0.451            | 0.470                |
| CamamBert                | 0.580          | 0.593            | 0.581        | 0.580            | 0.581                |


## 3. Models

### 3.1 Logistic Regression, K Nearest Neighbours, Decision Tree, Random Forests & Multinominal Naïve Bayes

For this first series of classification methods, we decided to transform our text using TI-DF representation. As a tokenizer, we defined our own tokenizer function using methods from the Spacy, String, and nltk.stem.snowball libraires. 

Naturally, beforehand we first split our data into 80% train data (X_train) and 20% test data (y_test).
The tokenizer function we used was originally designed for the English language and designed by our Teaching Assistants: Stergios Konstantinidis and Ludovic Mareemootoo. We then modified it to suit the French language. Below, Figure 3 is a snapshot of our tokenizer. 
 


![image](https://github.com/DenizcanSarigul/DSML-Project/assets/119871445/a6c1a775-137f-4f11-89a8-f3001721e6b1)



*Figure 3: Tokenizer*

Initially, in our tokenizer, we included code to remove the stopwords and punctions. However, we noticed overtime these actually limited the accuracy of our model. Therefore, we decided to remove them. After multipile tests, we came to the astoninishing observation we achieved better results without a tokenizer at all. Hence, the figures in our results do not include a custom tokenizer but instead the default tokenizer in the Vectorizer when iniated in the Pipeline. 

To construct our model, we used the Pipeline method from the sklearn.pipeline library, in which we inserted our vectorizer, TI-DF transformer, and classification method.  Figure 4, shows and example of the Pipeline we used. 
 

![image](https://github.com/DenizcanSarigul/DSML-Project/assets/119871445/5ed7df2e-9ed7-4ffa-932a-cacc926347d1)


*Figure 4: Pipeline example using Logistic Regression as a classification method*

We then used the model on our unllabeled test data mentioned above to create our predictions. The results of those are found above in section 2.

Figure 5 is the confusion matrix ot our best result with MultinominalNaiveBayes.

![image](https://github.com/DenizcanSarigul/DSML-Project/assets/119871445/60e96485-13d5-4a27-abeb-54eb118d32fc)

*Figure 5: confusion matrix with MultinominalNaiveBayes*

### 3.2 CamemBERT
We were briefly introduced to NLP models in class for sentiment analysis, but not multi-label classification. We researched different ways of implementing such a model and found a YouTube video tutorial on how to use BERT for text classification. The producer of the video also shared a GitHub repository where we could find all the required material. 

- YouTube video link: [Tutorial](https://www.youtube.com/watch?v=TmT-sKxovb0&t=608s)
- GitHub repository link: [GitHub](https://github.com/RajKKapadia/Transformers-Text-Classification-BERT-Blog)

The code provided by Mr. Kapadia was already very detailed and functional. As the original code was written for a binary classification, we changed what was needed to fit a multi-label classification problem. In addition, we changed the NLP model to CamemBERT. The model is found in the transformers library from HuggingFace.

For this model, the training is done in one file, and the testing in a separate one.

#### 3.2.1 Training 

Essentially, the important aspects of the code are the processing of the data, where the cleaning is done, and the training arguments. 
We start by defining our tokenizer and then the processing begins. Figure 5, is a snapshot of how the data is cleaned and, encoded and labelled. 

 ![image](https://github.com/DenizcanSarigul/DSML-Project/assets/119871445/83d2e08a-3c81-42df-8081-6e68deccd9cc)


*Figure 6: Processing of the data*



We then store the encodings into an array and create a new Data Frame. The data is then split, and is then ready to be used for the training.
The model is defined using AutoModelForSequenceClassification method, and use ‘camambert-base’. Before training, we define our training arguments. 

For this model, use the epoch method for the evaluation method. It turns out that the simplicity of our training arguments (Figure 7) resulted in the best accuracies. Previously, we tried optimising our arguments by including warmup_steps, weight_decay, Dropout, learning_rate but this strategy only led to poorer scores. It is worth noting that some combinations worked better than others such as using learning_rate 10e-5 with weight_decay and Dropout rather than learning_rate on it's own. However, globally, fine-tunning the arguments did not yield the results we expected. 


![image](https://github.com/DenizcanSarigul/DSML-Project/assets/119871445/16ebbd86-1ef5-4c52-b7a5-4ef2bfc45b6b)

*Figure 7: training arguments for the camamBert Model*


In addition, figure 8 is a graph shows the accuracy performances of different argurment combinations. The graph demonstrates that changing the arguments did not create a variance in accuracies, ultimately not being significantly helpful. 


![image](https://github.com/DenizcanSarigul/DSML-Project/assets/119871445/38f104d0-969c-4492-8be3-08edc8efb7fd)



*Figure 8: train accuracies graph*


### 4. Takeaway

During our testing rounds, we noticed that after the second or third epoch, model would tend to over-fit. Meaning that the training loss decreases while the validation loss increases. Despite this, once applied for testing witht the unlabelled data, we still received strong accuracies for our predictions. Which theoretically shouldn't be the case. Instead, this suggests that both the train data and the test data are very similar, perhaps orginiating from the same source.  

Despite our efforts to achieve better test accuracy, we struggled to improve our best score of 0.581. We tried pre-processing the data in different ways and fine-tuning the arguments, but none of those attempts seemed to work. Overall, the simpler or less detailed our fine-tuning led to better results. This leaves us asking ourselves why our model behaved this way and what we could have done to improve it. Two strategies that we didn't have enough time to try, and that may be helpful, are augmenting the data with new examples to overcome the overfitting problem and trying to create our own neural network model. A simple neural network might help overcome the overfitting problem specific to this data, as Camembert may be too complex for our dataset


#### Link to our YouTube video 

[YouTube video]([https://youtu.be/RggW_nbjA-c])


