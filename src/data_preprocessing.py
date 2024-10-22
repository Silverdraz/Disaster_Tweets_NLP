"""
Preprocesses the dataframes, such as remove unnecessary html tags, remove url links, emojis. All data preprocessing are performed on the
text column of the dataframes.
"""

#Import statements
import re #regular expression library (string)

def text_preprocessing(x_train):
    """Remove html tags, urls and emojis which does not aid in the predictive performance of a disaster tweet

    Args:
        x_train: Pandas Dataframe of the features column

    Returns:
        x_train: Preprocessed text for the Pandas Series of the text column
    """    
    #Data Preprocessing
    #Proceed to drop columns such as "Unnamed:0" unique identifier/id columns
    x_train["text"] = x_train["text"].apply(lambda x : remove_html(x))
    x_train["text"] = x_train["text"].apply(lambda x : remove_URL(x))
    x_train["text"] = x_train["text"].apply(lambda x: remove_emoji(x))
    return x_train


def perform_stemming(x_train,stemmer):
    """Performs stemming on each sentence in the dataframe

    Args:
        x_train: Pandas Dataframe of the features column
        stemmer: NLTK stemmer to stem the words in the sentence

    Returns:
        x_train: Pandas Dataframe with new text_stemmed column consisting of stemmed words for the sentence
    """    
    #Create a new column to prevent overwritting of old column
    x_train['text_stemmed'] = x_train['text'].apply(lambda x: stemm_text(x,stemmer))
    return x_train

def perform_lemmatization(x_train,lemmatizer):
    """Performs lemmatisation on each sentence in the dataframe

    Args:
        x_train: Pandas Dataframe of the features column
        lemmatizer: NLTK lemmatizer to lemmatize the words in the sentence

    Returns:
        x_train: Pandas Dataframe with new text_lemmatized column consisting of lemmatized words for the sentence
    """    
    #Create a new column to prevent overwritting of old column
    x_train['text_lemmatized'] = x_train['text'].apply(lambda x: lemmatize_text(x,lemmatizer))
    return x_train


def remove_html(text):
    """Remove html tags from each sentence of the pandas dataframe

    Args:
        text: sentence for a row of the pandas dataframe

    Returns:
        Sentence for a row of the dataframe with html tags removed
    """    
    html=re.compile(r'<.*?>')
    return html.sub(r'',text)

def remove_URL(text):
    """Remove urls from each sentence of the pandas dataframe

    Args:
        text: sentence for a row of the pandas dataframe

    Returns:
        Sentence for a row of the dataframe with urls removed
    """    
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'',text)

def remove_emoji(text):
    """Remove emojis from each sentence of the pandas dataframe

    Args:
        text: sentence for a row of the pandas dataframe

    Returns:
        Sentence for a row of the dataframe with emojis removed
    """  
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


def stemm_text(text,stemmer):
    """ Perform stemming using provided stemmer arg on each sentence/row of the dataframe

    Args:
        text: sentence for a row of the pandas dataframe
        stemmer: NLTK stemmer to stem the words in the sentence

    Returns:
        text: Sentence with stemmed words
    """  
    text = ' '.join(stemmer.stem(word) for word in text.split(' '))
    return text

def lemmatize_text(text,lemmatizer):
    """ Perform lemmatization using provided lemmatizer arg on each sentence/row of the dataframe

    Args:
        text: sentence for a row of the pandas dataframe
        lemmatizer: NLTK lemmatizer to lemmatize the words in the sentence

    Returns:
        text: Sentence with lemmatized words
    """  
    text = ' '.join(lemmatizer.lemmatize(word) for word in text.split(' '))
    return text
