# This file is a small exploratory script for EMG signal inspection.
# It loads the dataset, filters specific label/class combinations, and plots the signal.
# I would not remove anything here yet, because the script is still useful for debugging and visual validation.
# The duplicate NumPy import is redundant, but it is harmless and I left it in place to avoid changing working code.
#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from Visualization.plot_signal import plot_signal
from scipy.linalg import solve_toeplitz
from scipy.signal import periodogram



# ------------------------------------------------------------------ settings
INPUT_FILE = "Project_Data_EE4C12_S&S_EMG.csv"
df = pd.read_csv(INPUT_FILE)
channel_cols_all = [f"channel{i}" for i in range(1, 9)]     # 1 column = 1 channel

# series 0 = first recording of a subject, series 1 = second (clock restarts)
backward_jump = df.groupby("label")["time"].diff() < 0
df["series"] = backward_jump.groupby(df["label"]).cumsum().astype(int)

# Sampling rate
dt = df.groupby(["label", "series"])["time"].diff().dropna()
FS = 1000 / dt.median()

WINDOW_MS = 200                                         # window length in ms
OVERLAP_MS = 100                                        # hop between windows in ms
WINDOW_SIZE = int(WINDOW_MS * FS / 1000)                # samples per window
STEP_SIZE = WINDOW_SIZE - int(OVERLAP_MS * FS / 1000)   # samples between window starts
AR_ORDER = 4
N_CHANNELS = len(channel_cols_all)                      #! check if works

print(f"FS = {FS:.1f} Hz | window = {WINDOW_SIZE} samples | step = {STEP_SIZE} samples")

#%%

new_run = (
    (df["label"] != df["label"].shift())
    | (df["series"] != df["series"].shift())
    | (df["class"] != df["class"].shift())
)
df["segment"] = new_run.cumsum()

# We remove unmarked data (0) and class 7
df = df[~df["class"].isin([0, 7])].copy()
#print(df["class"].unique())

# Check how many segments there are per person for each class 
n_seg = df.groupby(["label", "class"])["segment"].nunique().unstack()
#print(n_seg)
#! confirms that there are 2 gestures of the same class per serie.

seg_len = df.groupby("segment").size()
#print(seg_len.describe())

print(df['segment'][50:100])

#%%

#Each channel (8) has 36x6x4=864 segments.
def extract_segments(df):
    """One entry per segment: its subject, class, and the (samples, 8) signal."""
    segments = []
    for seg_id, g in df.groupby("segment"):
        segments.append({
            "segment": seg_id,
            "subject": g["label"].iloc[0],
            "class": g["class"].iloc[0],
            "signal": g[channel_cols_all].to_numpy(),
        })
    return segments


#new_df = extract_segments(df)

#%%

# Each segment gets window sliced according to the settings.
def extract_windows(signal_data, window_size, step_size):
    """
    (samples, channels) -> python list of windows
    Last window may be shorter.
    """
    n_samples = signal_data.shape[0]
    windows = []
    for start in range(0, n_samples, step_size):
        windows.append(signal_data[start:start + window_size, :])
        if start + window_size >= n_samples:
            break
    return windows


# gfrrf = new_df[0]
# onesegment = gfrrf['signal']
# windows = extract_windows(onesegment, WINDOW_SIZE, STEP_SIZE)
# print(windows[0])

#%%
# ------------------------------------------------------------------ features

# Each feature matrix of MAV has Nx8 entries, with N= number of windows. 
def compute_mav(window):
    """(n_samples, 8) -> (8,) one MAV per channel."""
    return np.mean(np.abs(window), axis=0)


def ar_coefficients(sig, order=AR_ORDER):
    """AR(order) coefficients via the Yule-Walker equations."""
    sig = sig - np.mean(sig)
    n = len(sig)
    # biased autocorrelation r[0..order]
    r = np.array([np.dot(sig[:n - k], sig[k:]) / n for k in range(order + 1)])
    if r[0] < 1e-30:                       # flat window -> no information
        return np.zeros(order)
    return solve_toeplitz(r[:order], r[1:order + 1])
 
 
def mean_frequency(sig, fs=FS):
    """MNF = sum(f * P(f)) / sum(P(f)) using the periodogram."""
    f, p = periodogram(sig - np.mean(sig), fs=fs)
    total = p.sum()
    return float((f * p).sum() / total) if total > 0 else 0.0
 
 
def compute_window_features(window):
    """
    Input : (window_size, 8)
    Output: (48,) = 6 features per channel x 8 channels
            order per channel: MAV, AR1, AR2, AR3, AR4, MNF
    """
    window_features = []
    for ch in range(N_CHANNELS):
        sig = window[:, ch]
        window_features.append(np.mean(np.abs(sig)))        # MAV
        window_features.extend(ar_coefficients(sig))         # AR(4)
        window_features.append(mean_frequency(sig))          # MNF
    return np.array(window_features)
 
 
FEATURE_NAMES = [
    f"ch{ch + 1}_{name}"
    for ch in range(N_CHANNELS)
    for name in ["MAV", "AR1", "AR2", "AR3", "AR4", "MNF"]
]



# plot_8_channels(time=df["time"], channel=channel_cols_all, title="Raw 8-channel Data")


#%%

segments = extract_segments(df)

mav_list, y_labels, subject_ids, segment_ids, window_lengths = [], [], [], [], []

for item in segments:
    windows = extract_windows(item["signal"], WINDOW_SIZE, STEP_SIZE)
    for w in windows:
        mav_list.append(compute_mav(w))
        y_labels.append(item["class"])
        subject_ids.append(item["subject"])
        segment_ids.append(item["segment"])
        window_lengths.append(len(w))

X_mav = np.array(mav_list)          # (N, 8)
y_labels = np.array(y_labels)       # (N,)
subject_ids = np.array(subject_ids)
segment_ids = np.array(segment_ids)
window_lengths = np.array(window_lengths)

print(f"MAV feature matrix: {X_mav.shape}")
print(f"Window lengths: {dict(zip(*np.unique(window_lengths, return_counts=True)))}")


all_windows = []
y_labels = []
subject_ids = []

mav_df = pd.DataFrame(X_mav, columns=[f"ch{i}" for i in range(1, 9)])
mav_df["class"] = y_labels

# average MAV per class and channel
mean_table = mav_df.groupby("class").mean()
print(mean_table)

plt.figure(figsize=(10, 4))
sns.heatmap(mean_table, annot=True, fmt=".1e", cmap="viridis")
plt.title("Mean MAV per class and channel")
plt.show()

# spread per channel and class
fig, axs = plt.subplots(2, 4, figsize=(16, 7), sharex=True)
for i, ax in enumerate(axs.flat):
    sns.boxplot(data=mav_df, x="class", y=f"ch{i + 1}", ax=ax, showfliers=False)
    ax.set_title(f"Channel {i + 1}")
plt.tight_layout()
plt.show()

#%%

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
