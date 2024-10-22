"""
Provide plotting functions to allow plotting of loss curves and f1 scores curves. This script helps to abstract the details 
away from the main train and test modules
"""

#Import statements
import matplotlib.pyplot as plt #data viz
import os #os/path library
import numpy as np #numpy array library

#Global Constant to save the file at this location
SAVE_VIS_PATH = r"..\visualisations"

def plot_loss_acc_curves(results,model_string):
    """Retrieve the list of evaluation scores from the results. Acts as an abstract function for plotting

    Plots are saved in the visualisaitons folder with the model_string as a stem for naming

    Args:
        results: Dictionary of the results
        model_string: String for naming the plots

    """
    train_loss_list, val_loss_list = results["train_loss_list"], results["test_loss_list"]
    train_accuracy_list, val_accuracy_list = results["train_accuracy_list"], results["test_accuracy_list"]
    plot_loss_curve(train_loss_list,val_loss_list,model_string)
    plot_accuracy_curve(train_accuracy_list, val_accuracy_list,model_string)


def plot_loss_curve(train_loss_list,val_loss_list,model_string):
    """Plot the train loss and val loss

    Args:
        train_loss_list: List of the train loss values across epochs
        val_loss_list: List of the val loss values across epochs
        model_string: String for naming the plots
    """
    #Plot the loss curves for training loss and validation loss
    num_epochs = len(train_loss_list)
    plt.plot(np.arange(1,num_epochs+1),train_loss_list,label="training loss")
    plt.plot(np.arange(1,num_epochs+1),val_loss_list,label="validation loss")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.title("Training/Validation Loss across Epochs")
    plt.legend()
    #Save the figure and close it without manual closing
    plt.savefig(os.path.join(SAVE_VIS_PATH,f"{model_string}_loss.png"))
    plt.close()

def plot_accuracy_curve(train_accuracy_list,val_accuracy_list,model_string):
    """Plot the train accuracy and val accuracy

    Args:
        train_loss_list: List of the train accuracy values across epochs
        val_loss_list: List of the val accuracy values across epochs
        model_string: String for naming the plots
    """
    #Plot the accuracy curves for training accuracy and validation accuracy
    num_epochs = len(train_accuracy_list)
    plt.plot(np.arange(1,num_epochs+1),train_accuracy_list,label="training Acc")
    plt.plot(np.arange(1,num_epochs+1),val_accuracy_list,label="validation Acc")
    plt.xlabel("epochs")
    plt.ylabel("Acc score")
    plt.title("Training/Validation accuracy score across Epochs")
    plt.legend()
    #Save the figure and close it without manual closing
    plt.savefig(os.path.join(SAVE_VIS_PATH,f"{model_string}_acc.png"))
    plt.close()
