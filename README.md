# MACHINE-LEARNING-MODEL-IMPLEMENTATION

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: KANISHKA K

*INTERN ID*: CT04DH1742

*DOMAIN*: PYTHON PROGRAMMING

*DURATION*: 4 WEEKS

*MENTOR*: NEELA SANTOSH

*DESCRIPTION*: 

MACHINE LEARNING MODEL IMPLEMENTATON : SMS SPAM DETECTIO USING SCIKIT-LEARN

This project demonstrates a machine learning model implementation for detecting spam messages in SMS text using the scikit-learn library. The goal is to build a predictive classification model that automatically identifies whether a message is spam or ham (not spam) based on its content. This is a classic application of Natural Language Processing (NLP) and supervised learning.

DATASET OVERVIEW :

The dataset used in this project is the SMS Spam Collection Dataset, which contains 5,572 labeled text messages. Each message is tagged as either ‘spam’ or ‘ham’. The dataset is formatted as a tab-separated file (sms.tsv), with two main columns: the label and the message content.

PROJECT WORKFLOW :

The project is implemented in Python using a structured machine learning pipeline:

1.DATA LOADING

The dataset is loaded using pandas and parsed as tab-separated values. The columns are renamed for better understanding — ‘label’ (target variable) and ‘message’ (input text data).

2.DATA PREPROCESSING

Since machine learning models require numerical input, the text data is transformed using TF-IDF Vectorization. TF-IDF (Term Frequency-Inverse Document Frequency) helps identify how relevant a word is within a message, balancing word frequency across the dataset. This transformation converts the text into a sparse numerical matrix.

3.LABEL ENCODING

The categorical labels ham and spam are encoded to binary values — 0 and 1 respectively — using LabelEncoder from scikit-learn.

4.TRAIN-TEST SPLIT

The dataset is split into training and testing sets (typically 80-20). This ensures the model is trained on one subset and evaluated on another, unseen subset to assess its real-world performance.

5.MODEL BUILDING

A Logistic Regression classifier is selected for this binary classification task. It’s a widely used algorithm for its simplicity and efficiency in classification problems. The model is trained using the TF-IDF features extracted from the training set.

6.MODEL EVALUATION

The model is evaluated using a confusion matrix, accuracy score, and a classification report. The confusion matrix is visualized with seaborn for clarity. These metrics help us understand how many messages were correctly or incorrectly classified.

RESULTS :

The model performs with high accuracy, correctly identifying a large portion of spam and ham messages. The confusion matrix shows very few false positives or negatives, indicating that the model generalizes well on unseen data. This suggests the effectiveness of using TF-IDF with logistic regression for text classification tasks.

TOOLS AND TECHNOLOGIES USED :

Language: Python

Libraries: pandas, scikit-learn, matplotlib, seaborn

Algorithm: Logistic Regression

Feature Engineering: TF-IDF Vectorizer

Environment: Jupyter Notebook / VSCode

CONCLUSION :

This project effectively demonstrates how a machine learning model can be implemented using scikit-learn to solve a real-world problem. The approach is efficient, interpretable, and can be further enhanced using other NLP techniques, model tuning, or deep learning architectures. It serves as a foundational example for students and professionals exploring spam detection, NLP, or classification system.

*OUTPUT*:

1.DATA PREVIEW :

<img width="493" height="217" alt="Image" src="https://github.com/user-attachments/assets/072e79fd-277f-4270-b1fa-adde1d8faeda" />

2.ACCURACY OUTPUT :

<img width="533" height="412" alt="Image" src="https://github.com/user-attachments/assets/b5e99333-b5a7-4f5e-a7eb-c655db38ad11" />

3.CONFUSION MATRIX :

<img width="685" height="579" alt="Image" src="https://github.com/user-attachments/assets/36b7f74f-cfe2-46a7-a54c-28d22178c3f6" />
