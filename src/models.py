"""
Create models, such as baseline with iterative improvements and perform model comparison/evaluation.
Although some functions are essentially the same, they are named differently so that it is clearer and more explicit when comparing
models at the mlp.py module. Furthermore,(KIV) the recommendation is to use countvectorizer couple with MultinomialNB, which can be explored further 
"""

#Import statements
import pandas as pd #data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np #array data manipulation
import os #os file path
from sklearn.model_selection import KFold #KFold Cross Validation
from sklearn.model_selection import cross_val_score #cross validation score
from sklearn.pipeline import Pipeline #for pipeline 
from sklearn.model_selection import cross_validate #cross validation score
from sklearn.model_selection import GridSearchCV #grid search cv
from sklearn.feature_extraction.text import CountVectorizer #CountVectorizer BOW vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer #TfidfVectorizer TFIDF vectoriz

#Import Models for comparison
from sklearn.linear_model import LogisticRegression #logistic regression
from sklearn import svm #support vector Machine
from sklearn.ensemble import RandomForestClassifier #Random Forest
from sklearn.tree import DecisionTreeClassifier #Decision Tree
from xgboost import XGBClassifier #xgboost
from sklearn.naive_bayes import MultinomialNB #MultinomialNB

RESULTS_PATH = r"..\results" #Path to results data

def count_baseline_algo(x_train,y_train):
    """Logistic Regression is chosen as the baseline model using count vectorizer (without any additional feature engineering)

    Args:
        x_train: Dataframe of features
        y_train: ground-truth labels
    """    
    #create a baseline logistic regression with pipeline of bow (bag-of-words)
    pipe = Pipeline([("bow_vectorizer", CountVectorizer()),("model", LogisticRegression())])
    kfold = KFold(n_splits=5,shuffle=True,random_state=42)
    results = np.mean(cross_val_score(pipe, x_train["text"],y_train, cv=kfold))
    print(results)


def tfidf_baseline_algo(x_train,y_train):
    """Logistic Regression is chosen as the baseline model using TFIDF vectorizer (without any additional feature engineering)

    Args:
        x_train: Dataframe of features
        y_train: ground-truth labels
    """    
    #create a baseline logistic regression with pipeline of bow (bag-of-words)
    pipe = Pipeline([("tfidf_vectorizer", TfidfVectorizer()),("model", LogisticRegression())])
    kfold = KFold(n_splits=5,shuffle=True,random_state=42)
    results = np.mean(cross_val_score(pipe, x_train["text"], y_train, cv=kfold))
    print(results)

def tfidf_stem_baseline(x_train,y_train):
    """Logistic Regression is chosen as the baseline model using TFIDF vectorizer on stemmed text column

    Args:
        x_train: Dataframe of features with added stemmed column
        y_train: ground-truth labels
    """    
    #create a baseline logistic regression with pipeline of bow (bag-of-words)
    pipe = Pipeline([("tfidf_vectorizer", TfidfVectorizer()),("model", LogisticRegression())])
    kfold = KFold(n_splits=5,shuffle=True,random_state=42)
    results = np.mean(cross_val_score(pipe, x_train["text_stemmed"], y_train, cv=kfold))
    print(results)

def tfidf_lemmatizer_baseline(x_train,y_train):
    """Logistic Regression is chosen as the baseline model using TFIDF vectorizer on lemmatized text column

    Args:
        x_train: Dataframe of features with added stemmed column
        y_train: ground-truth labels
    """    
    #create a baseline logistic regression with pipeline of bow (bag-of-words)
    pipe = Pipeline([("tfidf_vectorizer", TfidfVectorizer()),("model", LogisticRegression())])
    kfold = KFold(n_splits=5,shuffle=True,random_state=42)
    results = np.mean(cross_val_score(pipe, x_train["text_lemmatized"], y_train, cv=kfold))
    print(results)

# Compare the various models for benchmarking  
def compare_models(x_train,y_train):
    """ Comparison among the common classification models to check which model is able to best extract the signals from the text
        and provide the highest predictive performance. The results are saved to model_results.csv.

        At method call, the x_train already has the text preprocessed for removal of urls, html tags, and emojis
        with text_stemmed and text_lemmatised column in addition to the text column

        Args:
            x_train: Feature dataframe with preprocessed text + stemmed text column + lemmatized text column
            y_train: labels for the dataset

        Returns:
            results_df: features dataframe with original text, text_stemmed, and text_lemmatised
            cv_validation: All validation scores for each fold
    """   
    #Store the results
    cv_validation = []
    cv_validation_mean = []
    cv_train_mean = []
    cv_std = []
    #For index of dataframe created later to store the results
    classifier_names = ['Radial Svm','Logistic Regression','Decision Tree',
                        'Multinomial Naive Bayes','Random Forest','XGBoost']
    models=[svm.SVC(kernel='rbf'),LogisticRegression(),DecisionTreeClassifier(),
            MultinomialNB(),RandomForestClassifier(),XGBClassifier()]
    
    #Iterate through the different models
    for model in models:
        pipe = Pipeline(
                    steps=[('tfidf_vectorizer', TfidfVectorizer()),("classifier", model)]
                )
        #Shuffle the dataset to prevent learning of order
        kfold = KFold(n_splits=5,shuffle=True,random_state=42)        
        results = cross_validate(pipe,x_train["text"],y_train,cv=kfold,return_train_score=True,scoring="accuracy")
        #Store the results train, test score to check for overfitting
        cv_train_mean.append(np.mean(results["train_score"]))
        cv_validation_mean.append(np.mean(results["test_score"]))
        cv_std.append(np.std(results["test_score"]))
        cv_validation.append(results["test_score"])
    #Create the dataframe
    results_df = pd.DataFrame({'mean train accuracy': cv_train_mean,'mean validation accuracy':cv_validation_mean,
                               'std accuracy':cv_std},index=classifier_names)
    #Save the results for reference comparison
    results_df.to_csv(os.path.join(RESULTS_PATH,f"model_results.csv"))
    #Return the results df for visualisation plotting
    return results_df, cv_validation
    

#Tune the model that has the highest accuracy
def log_reg_tuning(x_train,y_train):
    """ SVM is chosen at the model for prediction/inference. Add-on spline transformation to check if predictive performance is 
        enhanced when non-linearity is considered.

        Standard Scalar is required as spline has high degrees which will non-linearity increase the features distances. SVM uses 
        the features distacnes for classification. Otherwise, results are inaccurate
        Args:
            x_train: features dataframe
            y_train: ground-truth labels
    """   
    #Shuffle the dataset to prevent learning of order
    kfold = KFold(n_splits=5,shuffle=True,random_state=42)
    pipe = Pipeline(
                    steps=[('tfidf_vectorizer', TfidfVectorizer()),
                           ('log_reg', LogisticRegression())]
            )
    parameters = {'log_reg__C':[0.7,0.8,0.9,0.95,1,1.1,1.2],
                  "log_reg__penalty":("l2","elasticnet"),
                  'tfidf_vectorizer__ngram_range': [(1, 1), (1, 2), (1, 3)]}
    
    grid = GridSearchCV(estimator=pipe,param_grid=parameters,cv=kfold,return_train_score=True)
    grid.fit(x_train["text"],y_train)
    print("This is the best score")
    print(grid.best_score_)
    print("This is the best params")
    print(grid.best_params_)
    print(grid.cv_results_)
    tuned_params_df = pd.DataFrame.from_dict(grid.cv_results_)
    #grid_features = ["mean_train_score","mean_test_score"]
    #tuned_params_df = tuned_params_df[grid_features]
    tuned_params_df.to_csv(os.path.join(RESULTS_PATH,f"model_tuned_results.csv"))
