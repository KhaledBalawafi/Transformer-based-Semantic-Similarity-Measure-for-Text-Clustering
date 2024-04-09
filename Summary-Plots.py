import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Create a DataFrame from the given data
data = {
    "Evaluation Metric": ["ROUGE-1", "ROUGE-2", "ROUGE-L"],
    "Precision": [1.00, 0.97, 0.60],
    "Recall": [1.00, 0.97, 0.60],
    "F1 Score": [1.00, 0.97, 0.60]
}


#data = {
#    "Evaluation Metric": ["ROUGE-1", "ROUGE-2", "ROUGE-L"],
#    "Precision": [0.38, 0.67, 0.49],
#    "Recall": [0.25, 0.44, 0.32],
#    "F1 Score": [0.33, 0.58, 0.42]
#}


df = pd.DataFrame(data)

# Define custom colors for the plot
custom_palette = sns.color_palette("Set2")

# Set the style of the plot
sns.set(style="whitegrid")

# Create separate figures for each evaluation metric
fig, axes = plt.subplots(3, 1, figsize=(6, 10))

# Plot Precision
sns.barplot(x="Evaluation Metric", y="Precision", data=df, palette=custom_palette, ax=axes[0])
axes[0].set_title("Precision")
axes[0].set_ylabel("Score")
axes[0].set_ylim(0, 1.2)

# Plot Recall
sns.barplot(x="Evaluation Metric", y="Recall", data=df, palette=custom_palette, ax=axes[1])
axes[1].set_title("Recall")
axes[1].set_ylabel("Score")
axes[1].set_ylim(0, 1.2)

# Plot F1 Score
sns.barplot(x="Evaluation Metric", y="F1 Score", data=df, palette=custom_palette, ax=axes[2])
axes[2].set_title("F1 Score")
axes[2].set_xlabel("Evaluation Metric")
axes[2].set_ylabel("Score")
axes[2].set_ylim(0, 1.2)

# Adjust spacing between subplots
plt.tight_layout()

# Show the plots
plt.show()
