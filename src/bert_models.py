"""
Defines the model architecture for bert_train.py and bert_final_training.py to call the models. All models even default 
BertForSequenceClassification are defined here to ensure consistency and that these models are located in a single modular script
"""

#Import statement
import torch.nn as nn #nerual network modules

#Improt statement for transformers (HuggingFace)
from transformers import BertModel #Bert base model
from transformers import AutoModelForSequenceClassification  # Model with head for sequence classification (agnostic)

def BertHuggingFaceSeq():
    """ Default Bert for Sequence Classification model from HuggingFace

    Returns:
        model: Default Bert for Sequence Classification model from HuggingFace
    """
    # Create model with head for text classification (model agnostic)
    model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased')
    return model

class BertGeneralHeadModel(nn.Module):
    """ Custom Bert Model Architecture to implement stronger regularisation using a param of 0.35
    Generalised Bert Model Architecture is first ensured, followed by implementing additional 
    custom layers
    """
    def __init__(self):
        super(BertGeneralHeadModel,self).__init__()
        #Bert base foundaitonal model
        self.bert = BertModel.from_pretrained("bert-base-uncased",return_dict=False)
        #Classifier Head
        self.classifier = nn.Sequential(
            nn.Dropout(0.35),
            nn.Linear(768,768),
            nn.Linear(768,512),
            nn.Linear(512,256),
            nn.Linear(256,2)
        )
        
    def forward(self,input_ids,attention_mask,labels=None):
        _, pooled_output = self.bert(input_ids=input_ids,attention_mask=attention_mask)
        logits = self.classifier(pooled_output)
        #Train,Valid will have labels
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))
            output = {}
            output["loss"] = loss
            output["logits"] = logits
            return output
        #Pred/Inference will not have labels
        else:
            return logits