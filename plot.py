"""
@file: plot.py
@brief: This script is used to plot the training and validation losses of two models in one figure.
@details: The script loads the CSV files containing the training and validation
losses of two models and plots them in one figure.
"""
import pandas as pd
import matplotlib.pyplot as plt
import os

# Create a directory if it doesn't exist
folder_name = "results"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# Load the CSV files
df1 = pd.read_csv("results/Default_model/losses_default.csv")
df2 = pd.read_csv("results/Model_extra_hidden/losses_extra_hidden.csv")

# Plot the data
plt.figure(figsize=(10, 6))

# Plot training losses
plt.plot(df1["train_losses"], label="Model 1 Training Loss", color="blue")
plt.plot(df2["train_losses"], label="Model 2 Training Loss", color="green")

# Plot validation losses
plt.plot(
    df1["val_losses"], label="Model 1 Validation Loss", linestyle="--", color="orange"
)
plt.plot(
    df2["val_losses"], label="Model 2 Validation Loss", linestyle="--", color="red"
)

# Adding titles and labels
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig(os.path.join(folder_name, "default_extra_hidden_combined_loss_plot.png"))
