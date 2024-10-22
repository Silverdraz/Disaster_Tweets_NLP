"""
------------------------------------
Applying the data preprocessing, model building, model comparison and evaluation stages in the machine learning pipeline
"""

#Import statements
import pandas as pd #data processing, CSV file I/O (e.g. pd.read_csv)
import os #os file paths
import nltk #NLP Language Modelling 
from nltk.stem.wordnet import WordNetLemmatizer # NLTK Lemmatizer

# py files in src folder
import models #models script 
import data_preprocessing #data preprocessing script
import visualisation #visualisation script


#Global Constants
RAW_DATA_PATH = r"..\data\raw" #Path to raw data

def main():
    train_data, test_data = train_test_dfs()

    #Split into training and test (x & y)
    y_train= train_data["target"]
    x_train = train_data.drop(["target"],axis=1)
 
    #Baseline algorithm using Logistic Regression and CountVectorizer (Bag of Words)
    print("Performing evaluation using baseline Logistic Regression and CountVectorizer")
    models.count_baseline_algo(x_train,y_train)

    #Baseline algorithm using Logistic Regression and TFIDFVectorizer 
    print("Performing evaluation using baseline Logistic Regression and TFIDFVectorizer")
    models.tfidf_baseline_algo(x_train,y_train)

    #Preprocess the data by removing html tags, url links and emojis
    x_train = data_preprocessing.text_preprocessing(x_train)

    #Data Preprocessed text + Baseline algorithm using Logistic Regression +  CountVectorizer (Bag of Words)
    print("Performing evaluation using baseline Logistic Regression + CountVectorizer + Preprocessed text ")
    models.count_baseline_algo(x_train,y_train)


    #Data Preprocessed text + Baseline algorithm using Logistic Regression +  TFIDFVectorizer
    print("Performing evaluation using baseline Logistic Regression + TFIDFVectorizer + Preprocessed text ")
    models.tfidf_baseline_algo(x_train,y_train)

    #Initialise snowball stemmer to retrieve the stem form of the words
    stemmer = nltk.SnowballStemmer("english")

    #Create a new column for stemmed sentences to prevent overriding original, text column
    x_train = data_preprocessing.perform_stemming(x_train,stemmer)

    #Data Preprocessed text + Stemming + Baseline algorithm using Logistic Regression +  TFIDFVectorizer
    print("Performing evaluation using baseline Logistic Regression + TFIDFVectorizer + Preprocessed text + Stemming")
    models.tfidf_stem_baseline(x_train,y_train)
        
    #Create lemmatizer to perform lemmatization using relevant dictionary
    lemma = WordNetLemmatizer()
    nltk.download('wordnet')
    #Create a new column for lemmatized sentences to prevent overriding original, text column
    x_train = data_preprocessing.perform_lemmatization(x_train,lemma)

    #Data Preprocessed text + Lemmatization + Baseline algorithm using Logistic Regression +  TFIDFVectorizer
    print("Performing evaluation using baseline Logistic Regression + TFIDFVectorizer + Preprocessed text + Lemmatization")
    models.tfidf_lemmatizer_baseline(x_train,y_train)

    #Do model comparisons against the baseline Logistic Regression + iterative improvements (Preprocessed text)
    print("Model Comparison in progress")
    results_df, cv_validation = models.compare_models(x_train,y_train)
    visualisation.plot_visualisations(results_df,cv_validation)

    #Perform hyperparameter tuning on logistic regression
    print("Performing hyperparameter tuning on logistic regression")
    models.log_reg_tuning(x_train,y_train)


#Retrieve the data and split into train and test split
def train_test_dfs():
    """Retrieve the raw datas for model training

    Returns:
        train_data: dataframe for train dataset
        test_data: dataframe for test dataset
    """
    # Retrieve the raw train and raw test data
    train_data = pd.read_csv(os.path.join(RAW_DATA_PATH,f"train.csv"))
    test_data = pd.read_csv(os.path.join(RAW_DATA_PATH,f"test.csv"))
    return train_data, test_data



if __name__ == "__main__":
    main()
    


