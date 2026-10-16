# This file is a small exploratory script for EMG signal inspection.
# It loads the dataset, filters specific label/class combinations, and plots the signal.
# I would not remove anything here yet, because the script is still useful for debugging and visual validation.
# The duplicate NumPy import is redundant, but it is harmless and I left it in place to avoid changing working code.

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from Visualization.plot_signal import plot_signal



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

def extract_windows(signal_data, window_size=200, step_size=100):
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

def plot_8_channels(time, channel, title="8 Independent EMG Channels"):
    """
    Plots 8 channels on separate, vertically stacked axes sharing the same X-axis.
    channel_df should be a pandas DataFrame containing the 8 channel columns.
    """
    # Create a figure with 8 vertically stacked subplots sharing the X-axis
    fig, axs = plt.subplots(8, 1, sharex='all', figsize=(12, 14))
    fig.suptitle(title, fontsize=16)

    # Extract column names dynamically so it works with any filtered data
    cols = channel.columns

    # Loop through and plot each channel on its own axis
    for i in range(8):
        axs[i].plot(time, channel[cols[i]], color=f"C{i}")
        axs[i].set_ylabel(f"Ch {i+1}")
        axs[i].grid(True)

    plt.xlabel("Time (ms)")
    plt.tight_layout()
    plt.show()

def compute_window_features(window):
    """
    Extracts MAV, AR(4), and MNF features from a single window.
    Input shape:  (200, 8) -> 200 samples across 8 EMG channels
    Output shape: (48,)    -> 6 features per channel * 8 channels
    """
    window_features = []

    # Loop over each of the 8 EMG channels
    for ch in range(8):
        # Extract the 200 samples for the current channel
        sig = window[:, ch]

        # Mean Absolute Value (MAV)
        # Calculate the mean of the absolute values of sig
        mav = np.mean(np.abs(sig))
        window_features.append(mav)
        
        # Autoregressive model (AR)
        
        # Minimum Noise Fraction (MNF)
        

    return np.array(window_features)

## ===================================================================


input_fileroute = "Project_Data_EE4C12_S&S_EMG.csv"
df = pd.read_csv(input_fileroute)

channel_cols_all = [f"channel{i}" for i in range(1, 9)]
# plot_8_channels(time=df["time"], channel=channel_cols_all, title="Raw 8-channel Data")

# Filtering out Class 0 (noise) and Class 7 (not perfrmed by everyone):
df = df[df["class"] != 0].copy()
df = df[df["class"] != 7].copy()
print(df["class"].unique())

subject_signals = extract_signals(df)

all_windows = []
y_labels = []
subject_ids = []

for item in subject_signals:
    # --- Series 1 ---
    trial_1 = np.column_stack(item['signal_values_1'])
    if len(trial_1) >= 200:
        w_1 = extract_windows(trial_1, window_size=200, step_size=50)
        all_windows.append(w_1)
        y_labels.extend([item['class']] * len(w_1))
        subject_ids.extend([item['subject']] * len(w_1))

    # --- Series 2 (if present) ---
    if len(item['signal_values_2']) > 0:
        trial_2 = np.column_stack(item['signal_values_2'])
        if len(trial_2) >= 200:
            w_2 = extract_windows(trial_2, window_size=200, step_size=50)
            all_windows.append(w_2)
            y_labels.extend([item['class']] * len(w_2))
            subject_ids.extend([item['subject']] * len(w_2))

# Combine all extracted trial blocks into unified arrays
X_raw_windows = np.vstack(all_windows)
y_labels = np.array(y_labels)
subject_ids = np.array(subject_ids)

print(f"Raw windows shape:  {X_raw_windows.shape}")  # Expected: (N, 200, 8)
print(f"Labels shape:       {y_labels.shape}")       # Expected: (N,)
print(f"Subject IDs shape:  {subject_ids.shape}")   # Expected: (N,)


# features:

# Loop over every 200 ms window to extract features
X_features = np.array([compute_window_features(w) for w in X_raw_windows])

print(f"Feature matrix shape (MAV): {X_features.shape}")  # Expected: (N, 8)
print(f"Sample MAV vector (first window): \n{X_features[0]}")





# print(df.head())

# signal_value = df.loc[(df["label"] == 1) & (df["class"] == 1), "channel1"]
# signal_time = df.loc[(df["label"] == 1) & (df["class"] == 1), "time"]

# print(df.loc[(df["label"] == 1) & (df["class"] == 1), "time"])

# # plot_signal(signal_time, signal_value)

# # Extract segmented signals for all groups.



# # This loop checks whether the time values decrease, which would indicate a new segment.
# value_previous = 0
# for value in signal_time:
#     if value < value_previous:
#         print("a")
#     value_previous = value

# # Plot the first and second segments from each extracted signal group.
# for subject_signal in subject_signals:
#     plot_signal(subject_signal['signal_time_1'], subject_signal['signal_values_1'][0])
#     plot_signal(subject_signal['signal_time_2'], subject_signal['signal_values_2'][0])
