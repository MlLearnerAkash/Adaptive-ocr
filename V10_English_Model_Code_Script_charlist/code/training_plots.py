# import matplotlib.pyplot as plt

# # Data provided
# steps = [1, 3, 5, 7, 9, 11, 13, 15, 17, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 51]
# training_loss = [19.2, 15.05, 14.89, 14.69, 14.76, 14.76, 14.81, 17.46, 16.12, 15.33, 15.22, 15.09, 15.19, 14, 14.88, 14.89, 14.85, 14.82, 14.84, 14.9, 14.76, 16.46, 15.55, 15.09]
# validation_loss = [137, 14.89, 14.91, 14.73, 14.71, 14.79, 14.61, 16.06, 16, 15.41, 15.21, 15.11, 15.15, 15.07, 14.99, 14.93, 14.9, 14.93, 14.93, 14.82, 14.83, 14.889, 15.45, 15.22]

# # Create the plot
# plt.figure(figsize=(10, 6))
# plt.plot(steps, training_loss, marker='o', linestyle='-', label='Training Loss')
# plt.plot(steps, validation_loss, marker='s', linestyle='--', label='Validation Loss')

# # Add labels and title
# plt.xlabel("Steps")
# plt.ylabel("Loss")
# plt.title("Training and Validation Loss over Steps")
# plt.legend()

# # Insert annotation text on the plot (position in axis coordinates)
# plt.text(0.5, 0.95, "Learning rate 0.001 and adadelta optimizer",
#          transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', horizontalalignment='center')

# # Add grid for better readability
# plt.grid(True)

# # Display the graph
# plt.savefig("./font-2.png")


import matplotlib.pyplot as plt
import numpy as np

# Use a modern style for improved aesthetics
plt.style.use('seaborn-darkgrid')

# Data provided
steps = [1, 3, 5, 7, 9, 11, 13, 15, 17, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 51]
training_loss = [19.2, 15.05, 14.89, 14.69, 14.76, 14.76, 14.81, 17.46, 16.12, 15.33, 15.22, 15.09, 15.19, 14, 14.88, 14.89, 14.85, 14.82, 14.84, 14.9, 14.76, 16.46, 15.55, 15.09]
validation_loss = [137, 14.89, 14.91, 14.73, 14.71, 14.79, 14.61, 16.06, 16, 15.41, 15.21, 15.11, 15.15, 15.07, 14.99, 14.93, 14.9, 14.93, 14.93, 14.82, 14.83, 14.889, 15.45, 15.22]

# Create the figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Plot training and validation loss
ax.plot(steps, training_loss, marker='o', markersize=8, linestyle='-', linewidth=2,
        label='Training Loss', color='blue')
ax.plot(steps, validation_loss, marker='s', markersize=8, linestyle='--', linewidth=2,
        label='Validation Loss', color='orange')

# Add labels and a title with enhanced formatting
ax.set_xlabel("Steps", fontsize=14)
ax.set_ylabel("Loss(CTC-Loss)", fontsize=14)
ax.set_title("Training and Validation Loss over Steps", fontsize=16, weight='bold', pad=20)

# Insert annotation text with a bounding box for clarity
ax.text(0.5, 0.9, "Adam(10e-3)",
        transform=ax.transAxes, fontsize=12,
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
        horizontalalignment='center')

# Annotate outliers in the validation loss (if they are genuine data points)
# Here, step 1 and step 27 are annotated as outliers.
# (If step 27 is not an actual outlier, consider removing it.)
outlier_points = {1: 137, 27: 1515}
for step_val, loss_val in outlier_points.items():
    ax.annotate(f"highest val loss: {loss_val}",
                xy=(step_val, loss_val),
                xytext=(0, 15),
                textcoords="offset points",
                fontsize=10,
                color='red',
                arrowprops=dict(arrowstyle='->', color='red'))

# Set x-ticks at every 2 units
ax.set_xticks(np.arange(min(steps), max(steps) + 1, 2))

# Adjust tick parameters for readability
ax.tick_params(axis='both', which='major', labelsize=12)

# Add a legend with an appropriate font size
ax.legend(fontsize=12)

# Improve layout spacing and save the figure
plt.tight_layout()
plt.savefig("./font-2.png")
plt.show()

