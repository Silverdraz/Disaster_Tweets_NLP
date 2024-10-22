
import pandas as pd
import os
from sklearn.model_selection import train_test_split #split data
import bert_tweet_dataset
import bert_engine
import bert_models
import torch
from torch.optim import AdamW #Optimizer for weights
import bert_visualisation

#Import statements for Modelling
from transformers import AutoTokenizer #Tokenizer model agnostic
from transformers import AutoModelForSequenceClassification  # Model with head for sequence classification (agnostic)
from transformers.optimization import get_linear_schedule_with_warmup #learning rate scheduler
from transformers import BertModel #Bert base model

#Global Constants
RAW_DATA_PATH = r"..\data\raw" #Path to raw data
SAVED_MODEL_PATH = r"..\models" #Path to model folder#
NUM_FINAL_EPOCHS = 2 #Number of epochs for the final model training

def main():
    train_data, test_data = train_test_dfs()

    #Retrieve the typical x_train (labels) and y_train (labels)
    df_train_labels = train_data["target"].values
    df_train_raw = train_data["text"].values
    train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(df_train_raw,
                                                                                        df_train_labels,
                                                                                        test_size=0.2)

    #Set up device agnostic code - either GPU or CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    #Create Bert Tokenizer (tokenizer model agnostic) and Bert Model for text classification
    tokenizer, BertHuggingFaceModel = create_tokenizer_model(device)

    train_loader, valid_loader = create_loaders(tokenizer,
                                                train_inputs,
                                                validation_inputs,
                                                train_labels,
                                                validation_labels)

    #Adam with Weight Decay (to prevent overfitting)
    optimizer = AdamW(BertHuggingFaceModel.parameters(),
                lr=5e-5,
                weight_decay=0.01)
    
    #Learning rate scheduler with warmup steps to allow for gradual learning, instead of a sudden change in direction
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(train_loader) * NUM_FINAL_EPOCHS)

    # Start training with help from engine.py
    results = bert_engine.train(model=BertHuggingFaceModel,
                                train_dataloader=train_loader,
                                test_dataloader=valid_loader,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                epochs=NUM_FINAL_EPOCHS,
                                device=device)
    print(results)
    bert_visualisation.plot_loss_acc_curves(results,"BertDefaultModel")

    #Save both the final model and tokenizer for inference
    BertHuggingFaceModel.save_pretrained(os.path.join(SAVED_MODEL_PATH,"final_model")) 
    tokenizer.save_pretrained(os.path.join(SAVED_MODEL_PATH,"final_tokenizer"))


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

def create_tokenizer_model(device):
    """Create bert tokenizer as well as the default Bert Model for Sequence
    Classification by HuggingFace

    Args:
        device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
        tokenizer: Bert tokenizer
        BertHuggingFaceModel: BertforSequenceClassification default Bert Model
    """
    #Create Bert Tokenizer (tokenizer model agnostic)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    # Create model with head for text classification (model agnostic)
    BertHuggingFaceModel = bert_models.BertHuggingFaceSeq()
    BertHuggingFaceModel.to(device)
    return tokenizer, BertHuggingFaceModel

def create_loaders(tokenizer,
                   train_inputs,
                   validation_inputs,
                   train_labels,
                   validation_labels):
    """Create train loaders and validation loaders for the model

    Args:
        tokenizer: Bert Model Tokenizer
        train_inputs: Numpy array of texts for train dataset
        validation_inputs: Numpy array of texts for validation dataset
        train_labels: Numpy array of labels for train dataset
        validation_labels: Numpy array of labels for train dataset

    Returns:
        tokenizer: Bert tokenizer
        BertHuggingFaceModel: BertforSequenceClassification default Bert Model
    """
    train_encodings = tokenizer(list(train_inputs), truncation=True, padding=True)
    valid_encodings = tokenizer(list(validation_inputs), truncation=True, padding=True)

    # Create a pytorch custom dataset using encodings and labels for batch loading using dataloader subsequently
    train_dataset = bert_tweet_dataset.TweetDataset(train_encodings, train_labels)
    valid_dataset = bert_tweet_dataset.TweetDataset(valid_encodings, validation_labels)

    #Dataloaders for batch loading instead of loading full dataset (memory issues)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=16, shuffle=False)
    return train_loader, valid_loader


if __name__ == "__main__":
    main()
    