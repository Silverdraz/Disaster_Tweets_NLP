"""
Plot visualisations for a more intuitive visual inspection of the results
"""

#Import statements
import pandas as pd
import matplotlib.pyplot as plt
import os

VISUALISATION_PATH = r"..\visualisations" #Path to visualisations folder

def plot_visualisations(result_df,cv_validation):
    """Plot results on visualisations for visual inspection

    Args:
        results_df: Pandas Dataframe containing the results of the various model evaluations
        cv_validation: All validation scores for each fold for boxplot (mean and std dev)
    """        
    plot_boxplot(result_df,cv_validation)
    plot_barploth(result_df)

def plot_boxplot(results_df,cv_validation):
    """Plot results onto a boxplot for better visualisation

    Args:
        results_df: Pandas Dataframe containing the results of the various model evaluations and the classifier names (index)
        cv_validation: All validation scores for each fold for boxplot (mean and std dev)
    """    
    #Plot the cross validaiton results for visualisation, shwoing the std. deviation and mean
    plt.figure(figsize=(20,10))

    #Use the validation score for every fold across folds to allow for boxplot to calculate std and mean
    accuracy_df = pd.DataFrame(cv_validation,index=results_df.index)

    #Transpose of dataframe since boxplot is constructed from the columns axis
    boxplot_figure = accuracy_df.T.boxplot().get_figure()
    #Save the boxplot figure 
    boxplot_figure.savefig(os.path.join(VISUALISATION_PATH,'results_boxplot.jpg'))
    

def plot_barploth(results_df):
    """Plot results onto a boxplot for better visualisation

    Args:
        results_df: Pandas Dataframe containing the results of the various model evaluations
    """    
    #Create a new figure for this new horizontal barplot 
    plt.figure(figsize=(20,10))
    ax = results_df["mean validation accuracy"].sort_values().plot.barh(width=0.8)
    #Save the barplot figure in ax
    ax.figure.savefig(os.path.join(VISUALISATION_PATH,'results_barplot.jpg'))
