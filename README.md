# Disaster Tweet Classifiaction (NLP)
**The objective of this NLP github project is to showcase the thought proccess of data preprocessing in NLP and more importantly, approaches in modelling for NLP using classical machine learning algorithms and the SOTA transformer architectures with attention**

## Scoring Metric Choice
**Accuracy** is utilised as the scoring metric since from the Exploratory Data Analysis, the labels are approximately balanced. As such, accuracy is considered for the evaluation metric

## Exploratory Data Analysis + Data Preprocessing + Modelling Approaches 
**Exploratory Data Analysis** is performed to understand the characters and words in the sentences of the tweets. Furthermore, EDA is performed on the target (superrvised learning) so that an appropriate evaluation metric, **Accuracy**, is selected for proper evaluation.

1. **Data Preprocessing + Classical Machine learning Models**
Typical Data Preprocessing methods such as **1. Removing HTML tags, 2. removing URLs, 3. Removing Emojis** are implemented since intuitively urls and htmls do not contribute to the classification of the sequences. Furthermore, not all emojis convey meaningful messages and can be random in meaning. As such, emojis are removed during text processing. Furthermore, **the text processing performed above has led to an increase of accuracy result or rather predictive performance when TFIDF Vectorizer is used coupled with Logistic Regression**

**Stemming** has been implemented, but unfortunately **it has led to a decrease in predictive performance.** The choice of SnowballStemmer from NLTK would have been more ideal as compared to PorterStemmer, especially when accuracy is a key consideration since it is an improvement of Porter. However, SnowballStemmer is slightly worse than the PorterStemmer in this case. **Lemmatization** has been trialled without POS tagging, although it would have been more ideal to have POS tagging for more appropriate. **It has led to a reduction of accruacy, but the reduction is not as much compared to Stemming.**

**Stopwords Removal** was not implemented since negation words, such as "not" can be important in text classification. However, considering the count vectorizer and tfidf which are counts or normalsiation respectively, **it is questionable how useful stopwords would be compared to transformer architecture. Nonetheless, stopwords were kept in this case.**

2. **CountVectorizer and TFIDF Vectorizer** has both been implemented. **TFIDF Vectorizer with logistic regression** has shown to have better performance as compared to countvectorizer with logistic regression. It is not surprising since count vectorizer is simply a raw count, whereas TFIDF is a normalized count of the words across the documents. As such, the significance of words that are overtly common such as stopwords are downplayed. **TFIDF Vectorizer has better predictive performance as compared to countvectorizer**

3. **Cross Validation with non-tree based models and tree-based models**
Tree-based models have poorer predictive performance as compared to non-tree based models. It is not surprising when tfidf vectorizer and countvectorizer are used as it can be seen from the EDA that there are many columns/features/unique words. **The innate property of tree-base models is to split and having to split among 6000-7000 features is overwhelming for tree-based models, even if there is innate feature selection evident in tree-based models.**

4. **Hyperparameter Tuning for Logistic Regression - NGrams, C hyperparameter, penalty**
Hyperparameter Tuning was performed on Logistic Regression. **It was found that all models suffered from overfitting as shown in model_results.csv.**. As such, the **C hyperparameter** has been trialled to check if overfitting could be reduced from the regularisation as well as trailling both **L2 and elasticnet (a combination of L1+L2)** in efforts to reduce the overfitting. **From the model_tuned_results.csv file, when C = 1.2, the validation score is slightly higher as compared to the lower C values, which is not surprising as C is inverse to regularisation strength. When C = 0.7, it can be seen that the validation score is only slightly lower, but the overfitting has been reduced to a greater extent. It is hence a trade off and the overfitting reduction can a worthwhile trade off.**

5. **No Data Preprocessing + Transformer Architecture (BertForSequenceClassification vs BertModel + Custom Layers in Architecture)**
Bert Models have higher predictive performance as compared to classical machine learning since it is a SOTA model. 

Firstly, **no data preprocessing** is performed for bert since bert is bidirectional and is able to capture the semantic meaning of important stopwords such as "not". Bert Tokenizer with SOTA encoding (BPE,wordpiecce tokenizer) already has an innate stemming feature since tokens from the same words will be followed by "##". 

Secondly, The BertForSequenceClassification (with classification head) has been **fine-tuned** on the custom dataset and has acheived superior performance. 
The hyperparameters adheres to the best practice guidelines recommended by the BERT Authors
1. Batch Size - **16**,32
2. Epochs - 2, **3**, 4
3. Learning rate - 3e-4, 1e-4, **5e-5**, 3e-5

Overfitting is evident from the accuracy plots and loss plots

As such, **BertModel + Custom layers** has been introduced to add more dropout layers with higher probabilities to provide a greater regularisation effect. This is in effect recreating a **custom classification head** with stronger regularisation. However, there are more nn.linear layers which may also add more parameters and counter the effect of overfitting. Evidence of overfitting has similarly been reported. **More experimentation when given time allowance and increased computational ease can help with trialling various model architecture. **

**Overall**, Bert Architecture is recommended and either bert models report similar performance. Critically, **fine-tuning till the second epoch** suffices as there is always a drop in perfromance from the third epoch after the weight updates 

**Hence, the BertForSequenceClassification is utilised for final modelling and it trains for 2 epochs. The final model and tokenizer are both saved under the models folder when the codes are runned**

