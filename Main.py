# This file is a small exploratory script for EMG signal inspection.
# It loads the dataset, filters specific label/class combinations, and plots the signal.
# I would not remove anything here yet, because the script is still useful for debugging and visual validation.
# The duplicate NumPy import is redundant, but it is harmless and I left it in place to avoid changing working code.

import pandas as pd
import numpy as np
import seaborn as sns
from Visualization.plot_signal import plot_signal

# Duplicate import kept intentionally; it is not harmful, but it is redundant.
import numpy as np


def extract_signals(df):
    # Collect split signal segments for each subject/label/class pair.
    subject_signals = []

    # There are 8 EMG channels in the dataset.
    channel_cols = [f"channel{k + 1}" for k in range(8)]

    # Group rows by the subject label and class number.
    for (label, class_val), group in df.groupby(["label", "class"]):
        # Extract the time sequence for this subset.
        time = group["time"].to_numpy()

        # Detect and split new recording segments.
        split_indices = np.where(np.diff(time) < 0)[0] + 1
        time_splits = np.split(time, split_indices)

        # Split all 8 EMG channels using the same boundaries.
        values_splits = [
            np.split(group[col].to_numpy(), split_indices)
            for col in channel_cols
        ]

        # Store first series
        signal_values_1 = [val_split[0] for val_split in values_splits]

        # If reset, store second series
        signal_values_2 = (
            [val_split[1] for val_split in values_splits]
            if len(split_indices) > 0
            else []
        )

        # Store in dictionary
        subject_signals.append({
            'subject': label,
            'signal_values_1': signal_values_1,
            'signal_time_1': time_splits[0],
            'signal_values_2': signal_values_2,
            'signal_time_2': time_splits[1]
            if len(time_splits) > 1
            else np.array([]),
            'class': class_val,
        })

    return subject_signals

def extract_windows(signal_data, window_size=200, step_size=50):
    """
    Slices a 2D array of shape (samples, n_channels) into 3D windows:
    (n_windows, window_size, n_channels).
    """
    n_samples, n_channels = signal_data.shape
    windows = []
    
    for start in range(0, n_samples - window_size + 1, step_size):
        end = start + window_size
        windows.append(signal_data[start:end, :])
        
    return np.array(windows)


input_fileroute = "Project_Data_EE4C12_S&S_EMG.csv"

df = pd.read_csv(input_fileroute)

print(df.head())

signal_value = df.loc[(df["label"] == 1) & (df["class"] == 1), "channel1"]
signal_time = df.loc[(df["label"] == 1) & (df["class"] == 1), "time"]

print(df.loc[(df["label"] == 1) & (df["class"] == 1), "time"])

plot_signal(signal_time, signal_value)

# Extract segmented signals for all groups.
subject_signals = extract_signals(df)

# This loop checks whether the time values decrease, which would indicate a new segment.
value_previous = 0
for value in signal_time:
    if value < value_previous:
        print("a")
    value_previous = value

# Plot the first and second segments from each extracted signal group.
for subject_signal in subject_signals:
    plot_signal(subject_signal['signal_time_1'], subject_signal['signal_values_1'][0])
    plot_signal(subject_signal['signal_time_2'], subject_signal['signal_values_2'][0])
