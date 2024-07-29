import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

file_path = 'data/processed/evaluation_reports_v2.1.csv'    # Path to the evaluation reports CSV file
data = pd.read_csv(file_path)                               # Load the data from the CSV file

data['data'] = data['model'].str.extract(r'^(base_|smote_pca_|smote_)') # Extract the data type from the 'model' column


data['model_type'] = data['model'].replace({'^base_': '', '^smote_pca_': '', '^smote_': ''}, regex=True)    # Extract the model type from the 'model' column
data['data'] = data['data'].str.rstrip('_')                                                                 # Remove the trailing underscore from the 'data' column

unique_models = data['model_type'].unique() # Get the unique model types

print(unique_models)
print(data.head())

if not os.path.exists('plots/individual'):  # Create the 'plots/individual' directory if it does not exist
    os.makedirs('plots/individual')
if not os.path.exists('plots/group'):       # Create the 'plots/group' directory if it does not exist
    os.makedirs('plots/group')

palette = {'base': '#B80C09', 'smote': '#0B4F6C', 'smote_pca': '#01BAEF'}   # Define the color palette for each data type

def plot_grouped_metric(metric):
    sorted_data = data.sort_values(by=metric, ascending=False) # Sort the data by the metric
    
    # Create a FacetGrid for the bar plot
    g = sns.catplot(
        x='model_type', y=metric, hue='data', data=sorted_data, 
        kind='bar', height=6, aspect=2, palette=palette
    )
    
    g.fig.suptitle(f'{metric.replace("_", " ").capitalize()} by Model Type') # Set the title of the plot
    g.set_xticklabels(rotation=45, ha='right')                               # Rotate the x-axis labels   
    g._legend.set_bbox_to_anchor((1, 0.5))                                   # Move the legend to the right side
    g.tight_layout()                                                         # Adjust the layout of the plot
    plt.savefig(f'plots/individual/{metric}.png')   # Save the plot to a file
    plt.close()                                     # Close the plot
    
def plot_grouped_metrics(metrics_dict):
    for group, metrics in metrics_dict.items(): # Iterate over the grouped metrics
        fig, axes = plt.subplots(len(metrics), 1, figsize=(14, 6 * len(metrics)))       # Create subplots for each metric
        fig.suptitle(f'{group.replace("_", " ").capitalize()} by Model Type', y=1.02)   # Set the title of the plot
        
        if not isinstance(axes, np.ndarray): axes = [axes]  # Convert the axes to an array if it is not already
            
        for ax, metric in zip(axes, metrics):                           # Iterate over the axes and metrics
            sorted_data = data.sort_values(by=metric, ascending=False)  # Sort the data by the metric

            sns.barplot(x='model_type', y=metric, hue='data', data=sorted_data, palette=palette, ax=ax) # Create a bar plot
            ax.set_title(f'{metric.replace("_", " ").capitalize()}')                                    # Set the title of the plot
            ax.set_xticks(ax.get_xticks())                                                              # Set the x-ticks
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')                           # Rotate the x-axis labels
            ax.legend(title='Data', bbox_to_anchor=(1, 1))                                              # Set the legend and move it to the right side
        
        fig.tight_layout()                      # Adjust the layout of the plot
        plt.savefig(f'plots/group/{group}.png') # Save the plot to file
        plt.close()                             # Close the plot

# List of metrics to plot
metrics = [
    '0_precision', '0_recall', '0_f1-score',
    '1_precision', '1_recall', '1_f1-score',
    'accuracy', 'macro avg_precision', 'macro avg_recall',
    'macro avg_f1-score', 'weighted avg_precision',
    'weighted avg_recall', 'weighted avg_f1-score', 'eval_time'
]

# List of metrics to plot in subplots
grouped_metrics = {
    'class_0_metrics': ['0_precision', '0_recall', '0_f1-score'],
    'class_1_metrics': ['1_precision', '1_recall', '1_f1-score'],
    'precision_metrics': ['macro avg_precision', 'weighted avg_precision'],
    'recall_metrics': ['macro avg_recall', 'weighted avg_recall'],
    'f1_score_metrics': ['macro avg_f1-score', 'weighted avg_f1-score']
}

# Plot each metric grouped by model type
for metric in metrics:
    plot_grouped_metric(metric)
    
# Plot grouped metrics
plot_grouped_metrics(grouped_metrics)
