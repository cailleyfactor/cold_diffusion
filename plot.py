import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load the CSV files
df1 = pd.read_csv("losses2.csv")
df2 = pd.read_csv("losses3.csv")

# Step 3: Plot the data
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

plt.show()
