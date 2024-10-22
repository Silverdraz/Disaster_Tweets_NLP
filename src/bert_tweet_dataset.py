"""
A custom dataset class that can be extended to all pytorch defined datasets. Specifically, encodings from the tokenizer are used
to create train datalodaer and validation dataloader
"""

#Import statements
import torch #General troch module
from torch.utils.data import Dataset #custom dataset 

class TweetDataset(Dataset):
    """ Custom dataset class for the encoded tweets"""
    def __init__(self,encodings,labels):
        """ Initialise the dataset

        Args:
            encodings: encodings of the text created by Bert Tokenizer
            labels: labels for the respective text
        """
        self.encodings = encodings
        self.labels = labels
    
    def __getitem__(self,idx):
        """ Initialise the dataset

        Args:
            idx: Respective index 
        
        Returns: 
            item: Respective encoding and label for the item.
        """
        item = {key: torch.tensor(val[idx]) for key,val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        """Need to overide this method as mentioned in docs."""
        number_rows = len(self.labels)
        return number_rows