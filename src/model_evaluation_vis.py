import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# Load the data
file_path = 'data/processed/evaluation_reports_v2.1.csv'
data = pd.read_csv(file_path)

# Extract the prefixes and create a new column 'data'
data['data'] = data['model'].str.extract(r'^(base_|smote_pca_|smote_)')

# Remove prefixes 'base_', 'smote_', and 'smote_pca_' from the model column
data['model_type'] = data['model'].replace({'^base_': '', '^smote_pca_': '', '^smote_': ''}, regex=True)
# Remove the last underscore from the 'data' column
data['data'] = data['data'].str.rstrip('_')

# Extract unique model names without the prefixes
unique_models = data['model_type'].unique()

print(unique_models)

# Display the first few rows of the dataframe
print(data.head())

# Create directory for plots if it doesn't exist
if not os.path.exists('plots/individual'):
    os.makedirs('plots/individual')
if not os.path.exists('plots/group'):
    os.makedirs('plots/group')

# Define a consistent color palette for 'data'
palette = {
    'base': '#B80C09',
    'smote': '#0B4F6C',
    'smote_pca': '#01BAEF'
}

# Function to create grouped bar plots for each metric using catplot
def plot_grouped_metric(metric):
    # Sort the data by the metric in descending order
    sorted_data = data.sort_values(by=metric, ascending=False)
    # Create a FacetGrid for the bar plot
    g = sns.catplot(
        x='model_type', y=metric, hue='data', data=sorted_data, 
        kind='bar', height=6, aspect=2, palette=palette
    )
    # Adjust the title and the x-axis
    g.fig.suptitle(f'{metric.replace("_", " ").capitalize()} by Model Type')
    g.set_xticklabels(rotation=45, ha='right')
    # Adjust the legend position
    g._legend.set_bbox_to_anchor((1, 0.5))
    # Save the plot
    g.tight_layout()
    plt.savefig(f'plots/individual/{metric}.png')
    plt.close()  # Close the plot to avoid displaying it in the loop
    
# Function to create subplots for grouped metrics
def plot_grouped_metrics(metrics_dict):
    for group, metrics in metrics_dict.items():
        fig, axes = plt.subplots(len(metrics), 1, figsize=(14, 6 * len(metrics)))
        fig.suptitle(f'{group.replace("_", " ").capitalize()} by Model Type', y=1.02)
        
        if not isinstance(axes, np.ndarray):
            axes = [axes]

        for ax, metric in zip(axes, metrics):
            sorted_data = data.sort_values(by=metric, ascending=False)
            sns.barplot(x='model_type', y=metric, hue='data', data=sorted_data, palette=palette, ax=ax)
            ax.set_title(f'{metric.replace("_", " ").capitalize()}')
            ax.set_xticks(ax.get_xticks())
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.legend(title='Data', bbox_to_anchor=(1, 1))
        
        fig.tight_layout()
        plt.savefig(f'plots/group/{group}.png')
        plt.close()

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
