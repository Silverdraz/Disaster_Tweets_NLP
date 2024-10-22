"""
Trains the model using the provided optimizer, loss functions, scheduler in a range of epochs. Handles the training for the model
"""

#Import statements
from tqdm.auto import tqdm # Import tqdm for progress bar

#Torch import statements
import torch #General torch library

def accuracy_fn(y_true,y_pred):
    """Creates accuracy function for reusability.

    Takes in predictions and actual labels. Calculate the accuracy for these inputs and
    return the accuracy as a number

    Args:
        y_true: list of ground-truth labels.
        y_pred: list of predicted labels.

    Returns:
        Accuracy as a score
    """
    correct = torch.eq(y_true,y_pred).sum().item()
    acc = (correct/len(y_pred)) * 100
    return acc


def train_step(model,
               dataloader,
               optimizer,
               scheduler,
               device):
    """Trains a HuggingFace model using Pytorch for a single epoch.

    Turns a target HuggingFace model to training mode and then
    runs through all of the required training steps (forward
    pass, loss calculation, optimizer step, scheduler for learning rate).

    Args:
        model: A HuggingFace PyTorch model to be trained.
        dataloader: A DataLoader instance for the model to be trained on.
        optimizer: A PyTorch optimizer to help minimize the loss function.
        scheduler: Scheduler for gradual learning (learning rate)
        device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
        A tuple of training loss and training accuracy metrics
        In the form (train_loss, train_accuracy). 
    """

    #Training
    train_loss, train_acc = 0, 0
    #Add a loop to loop through the training batches
    for batch_idx, batch in enumerate(dataloader):        
        ### Prepare data and use GPU
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        #Allow configs for learning (e.g. weight updates)
        model.train()

        #Forward pass
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss, logits = outputs['loss'], outputs['logits']
        #Returns (max,max_indices)
        _, y_pred_labels = torch.max(logits,1)
        
        #Calculate the training loss per batch
        train_loss += loss
        #Calculate the training accuracy per batch
        train_acc += accuracy_fn(y_true=labels,
                                 y_pred=y_pred_labels)
        
        #Optimizer zero grad
        optimizer.zero_grad()
        
        #Loss backward
        loss.backward()
        
        #Optimizer step
        optimizer.step()
        
        #Scheduler step (lr step) is after changing the weights
        scheduler.step()
        
    #Average train loss per batch in an epoch and Average train accuracy per batch in an epoch
    train_loss /= len(dataloader)
    train_loss = train_loss.cpu().detach().numpy()
    train_acc /= len(dataloader)
    return train_loss, train_acc

def test_step(model, 
              dataloader, 
              #loss_fn,
              device):
    """Tests a HuggingFace model using Pytorch for a single epoch.

    Turns a target HuggingFace model to "eval" mode and then performs
    a forward pass on a testing dataset.

    Args:
        model: A HuggingFace PyTorch model to be trained.
        dataloader: A DataLoader instance for the model to be trained on.
        device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
        A tuple of testing loss and testing accuracy metrics
        In the form (test_loss, test_accuracy). 
    """
    #Testing
    test_loss, test_acc = 0, 0
    #Set to eval mode to prevent auto diff.
    model.eval()
    with torch.inference_mode():
        for batch_idx, batch in enumerate(dataloader):
            ### Prepare data and use GPU
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            #Forward Pass (BatchEncoding data type, docs)
            outputs = model(input_ids,attention_mask=attention_mask,labels=labels)
            loss, logits = outputs['loss'], outputs['logits']
            #Returns (max,max_indices)
            _, y_pred_labels = torch.max(logits,1)
            
            test_loss += loss
            #Calculate the validation accuracy per batch
            test_acc += accuracy_fn(y_true=labels,
                                    y_pred=y_pred_labels)
            
        #Average test loss per batch in an epoch and Average test accuracy per batch in an epoch
        test_loss /= len(dataloader)
        test_loss = test_loss.cpu().detach().numpy()
        test_acc /= len(dataloader)
        return test_loss,test_acc


def train(model, 
          train_dataloader, 
          test_dataloader, 
          optimizer,
          scheduler,
          epochs,
          device):
    """Trains and tests a HuggingFace model using PyTorch model.

    Passes a target HugginfFace PyTorch models through train_step() and test_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.

    Args:
    model: A HuggingFace PyTorch model to be trained.
    train_dataloader: A DataLoader instance for the model to be trained on.
    test_dataloader: A DataLoader instance for the model to be tested on.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    scheduler: Scheduler for gradual learning (learning rate)
    epochs: An integer indicating how many epochs to train for.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A dictionary of training and testing loss as well as training and
    testing accuracy metrics as well as training and testinf f1 scoring metric. 
    Each metric has a value in a list for each epoch for graph plotting
    """
    # Create empty results dictionary
    results = {"train_loss_list": [],
               "train_accuracy_list": [],
               "test_loss_list": [],
               "test_accuracy_list": [],
    }
    
    # Make sure model on target device
    model.to(device)

    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                            dataloader=train_dataloader,
                                            optimizer=optimizer,
                                            scheduler=scheduler,
                                            device=device
                                        )
        test_loss, test_acc = test_step(model=model,
                                            dataloader=test_dataloader,
                                            device=device
                                        )

        # Print out what's happening
        print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"test_loss: {test_loss:.4f} | "
          f"train_acc: {train_acc:.4f} | "
          f"test_acc: {test_acc:.4f} | "
        )

        # Update results dictionary
        results["train_loss_list"].append(train_loss)
        results["train_accuracy_list"].append(train_acc)
        results["test_loss_list"].append(test_loss)
        results["test_accuracy_list"].append(test_acc)
    # Return the filled results at the end of the epochs
    return results


