# Disaster Tweet Classifiaction (NLP)
**The objective of this NLP github project is to showcase the thought proccess of data preprocessing in NLP and more importantly, approaches in modelling for NLP using classical machine learning algorithms and the SOTA transformer architectures with attention**

## Scoring Metric Choice
**Accuracy** is utilised as the scoring metric since from the Exploratory Data Analysis, the labels are approximately balanced. As such, accuracy is considered for the evaluation metric

## Exploratory Data Analysis + Data Preprocessing + Modelling Approaches 
**Exploratory Data Analysis** is performed to understand the characters and words in the sentences of the tweets. Furthermore, EDA is performed on the target (superrvised learning) so that an appropriate evaluation metric, **Accuracy**, is selected for proper evaluation.

**The notebook has numerically labelled the different approaches using various vectorizers, machine learning algorithms, bert algorithm for convenient reference**
1. **Data Preprocessing + Classical Machine learning Models**
Typical Data Preprocessing methods such as **1. Removing HTML tags, 2. removing URLs, 3. Removing Emojis** are implemented since intuitively urls and htmls do not contribute to the classification of the sequences. Furthermore, classical machine learning can have issues with emojis (not considered as words) and as such these emojis are removed. **Stemming** has been implemented, but unfortunately it has led to a decrease in predictive performance. **Lemmatization** could have been trialled as it is considered to be less aggressive as compared to stemming. **Stopwords Removal** could have been implemented. However, since TFIDFvectorizer is used later, this is not a huge concern as it normalizes the word. 

2. **CountVectorizer and TFIDF Vectorizer** has both been implemented. **TFIDF Vectorizer with logistic regression** has shown to have better performance as compared to countvectorizer with logistic regression. It is not surprising since count vectorizer is simply a raw count, whereas TFIDF is a normalized count of the words across the documents. As such, the significance of words that are overtly common such as stopwords are downplayed. 

3. **Cross Validation with non-tree based models and tree-based models**
Tree-based models have poorer predictive performance as compared to non-tree based models. It is not surprising when tfidf vectorizer and countvectorizer are used as it can be seen from the EDA that there are many columns/features/unique words. **The innate property of tree-base models is to split and having to split among 6000-7000 features is overwhelming for tree-based models, even if there is innate feature selection evident in tree-based models.**

4. **No Data Preprocessing + Transformer Architecture (BertForSequenceClassification vs BertModel + Custom Layers in Architecture)**
Bert Models have higher predictive performance as compared to classical machine learning since it is a SOTA model. 

Firstly, **no data preprocessing** is performed for bert since bert is bidirectional and is able to capture the semantic meaning of important stopwords such as "not". Bert Tokenizer with SOTA encoding (BPE,wordpiecce tokenizer) already has an innate stemming feature since tokens from the same words will be followed by "##". 

Secondly, The BertForSequenceClassification (with classification head) has been **fine-tuned** on the custom dataset and has acheived superior performance. 
The hyperparameters adheres to the best practice guidelines recommended by the BERT Authors
1. Batch Size - **16**,32
2. Epochs - 2, **3**, 4
3. Learning rate - 3e-4, 1e-4, **5e-5**, 3e-5

Overfitting is evident from the accuracy plots and loss plots

As such, **BertModel + Custom layers** has been introduced to add more dropout layers with higher probabilities to provide a greater regularisation effect. This is in effect recreating a **custom classification head** with stronger regularisation. However, there are more nn.linear layers which may also add more parameters and counter the effect of overfitting. Evidence of overfitting has similarly been reported.

**Overall**, Bert Architecture is recommended and either bert models report similar performance. Critically, **fine-tuning till the second epoch** suffices as there is always a drop in perfromance from the third epoch after the weight updates 

