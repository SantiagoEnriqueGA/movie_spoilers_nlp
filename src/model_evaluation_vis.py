import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

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
if not os.path.exists('plots'):
    os.makedirs('plots')

# Function to create grouped bar plots for each metric using catplot
def plot_grouped_metric(metric):
    # Sort the data by the metric in descending order
    sorted_data = data.sort_values(by=metric, ascending=False)
    
    # Create a FacetGrid for the bar plot
    g = sns.catplot(
        x='model_type', y=metric, hue='data', data=sorted_data, 
        kind='bar', height=6, aspect=2, palette='icefire'
    )
    
    # Adjust the title and the x-axis
    g.fig.suptitle(f'{metric.replace("_", " ").capitalize()} by Model Type')
    g.set_xticklabels(rotation=45, ha='right')
    
    # Adjust the legend position
    g._legend.set_bbox_to_anchor((1, 0.5))
    
    # Save the plot
    g.tight_layout()
    plt.savefig(f'plots/{metric}.png')
    plt.clsoe()  # Close the plot to avoid displaying it in the loop

# List of metrics to plot
metrics = [
    '0_precision', '0_recall', '0_f1-score',
    '1_precision', '1_recall', '1_f1-score',
    'accuracy', 'macro avg_precision', 'macro avg_recall',
    'macro avg_f1-score', 'weighted avg_precision',
    'weighted avg_recall', 'weighted avg_f1-score', 'eval_time'
]

# Plot each metric grouped by model type
for metric in metrics:
    plot_grouped_metric(metric)
