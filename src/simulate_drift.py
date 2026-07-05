import pandas as pd
import numpy as np

# ---------------------------------
# Load Reference Dataset
# ---------------------------------

reference_data = pd.read_csv("data/reference.csv")

print("Reference Dataset Loaded")


# ---------------------------------
# Create Copy
# ---------------------------------

current_data = reference_data.copy()

print("Copy Created")


# ---------------------------------
# Add Gaussian Noise
# ---------------------------------

numerical_columns = current_data.columns.drop("target")

noise = np.random.normal(
    loc=0,
    scale=0.5,
    size=current_data[numerical_columns].shape
)

current_data[numerical_columns] += noise

print("Gaussian Noise Added")


# ---------------------------------
# Shift Important Features
# ---------------------------------

current_data["mean radius"] += 3

current_data["mean texture"] += 2

print("Feature Shift Applied")


# ---------------------------------
# Flip 10% Target Labels
# ---------------------------------

np.random.seed(42)

flip_indices = np.random.choice(
    current_data.index,
    size=int(0.10 * len(current_data)),
    replace=False
)

current_data.loc[
    flip_indices,
    "target"
] = 1 - current_data.loc[
    flip_indices,
    "target"
]

print("10% Labels Flipped")


# ---------------------------------
# Save Drifted Dataset
# ---------------------------------

current_data.to_csv(
    "data/current.csv",
    index=False
)

print("Current Dataset Saved Successfully")