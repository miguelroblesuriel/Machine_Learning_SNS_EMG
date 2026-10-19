# This file is a small exploratory script for EMG signal inspection.
# It loads the dataset, filters specific label/class combinations, and plots the signal.
# I would not remove anything here yet, because the script is still useful for debugging and visual validation.
# The duplicate NumPy import is redundant, but it is harmless and I left it in place to avoid changing working code.
print('hello')
#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from Visualization.plot_signal import plot_signal
from scipy.linalg import solve_toeplitz
from scipy.signal import periodogram
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score



# ------------------------------------------------------------------ settings
INPUT_FILE = "Project_Data_EE4C12_S&S_EMG.csv"
df = pd.read_csv(INPUT_FILE)
channel_cols_all = [f"channel{i}" for i in range(1, 9)]     # 1 column = 1 channel

# series 0 = first recording of a subject, series 1 = second (clock restarts)
backward_jump = df.groupby("label")["time"].diff() < 0
df["series"] = backward_jump.groupby(df["label"]).cumsum().astype(int)

# Sampling rate
dt = df.groupby(["label", "series"])["time"].diff().dropna()



PERIOD_MS = 8
FS = 1000 / PERIOD_MS

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


#%%
# --------------------------------------------------------------------- functions
def thin_segment(g, period_ms=PERIOD_MS):
    """Keep the first row of every 8 ms time bin -> about one row per real sample."""
    t = g["time"].to_numpy()
    bins = ((t - t[0]) // period_ms).astype(int)
    first_row_of_bin = np.unique(bins, return_index=True)[1]
    return g[channel_cols_all].to_numpy()[first_row_of_bin]

#Each channel (8) has 36x6x4=864 segments.
def extract_segments(df):
    """One entry per segment: its subject, class, and the (samples, 8) signal."""
    segments = []
    for seg_id, g in df.groupby("segment"):
        segments.append({
            "segment": seg_id,
            "subject": g["label"].iloc[0],
            "class": g["class"].iloc[0],
            "signal": thin_segment(g),
        })
    return segments

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
 
def compute_ar(window):
    """(n_samples, 8) -> (32,) = 4 AR coeff. per 8 channels"""
    ar = np.concatenate([ar_coefficients(window[:, ch]) for ch in range(N_CHANNELS)])
    return ar
 
def mean_frequency(sig, fs=FS):
    """MNF = sum(f * P(f)) / sum(P(f)) using the periodogram."""
    f, p = periodogram(sig - np.mean(sig), fs=fs)
    total = p.sum()
    return float((f * p).sum() / total) if total > 0 else 0.0

def compute_mnf(window):
    """(n_samples, 8) -> (8,) one mean frequency (Hz) per channel."""
    return np.array([mean_frequency(window[:, ch], fs=FS) for ch in range(N_CHANNELS)])
 
 
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




# plot_8_channels(time=df["time"], channel=channel_cols_all, title="Raw 8-channel Data")


#%% 
# -------------------------------------- EXTRACTING SEGMENTS AND FEATURES

segments = extract_segments(df)

mav_list, ar_list, mnf_list, y_labels, subject_ids, segment_ids, window_lengths = [], [], [], [], [], [], []

for item in segments:
    windows = extract_windows(item["signal"], WINDOW_SIZE, STEP_SIZE)
    for w in windows:
        mav_list.append(compute_mav(w))
        ar_list.append(compute_ar(w))
        mnf_list.append(compute_mnf(w))
        y_labels.append(item["class"])
        subject_ids.append(item["subject"])
        segment_ids.append(item["segment"])
        window_lengths.append(len(w))

# (???????????????????????????????????)
# From what the code suggests is that we have repetitions of samples. So the data has a unique sampling rate of ~125Hz but is actually sampled at 1000Hz.
# so essencially we have repeating values which affect MAV and AR (AR more than MAV). Not sure what to do with this...
 
# w = windows[0]
# print(w.shape, (np.diff(w, axis=0) == 0).all(axis=1).mean())

# print(df["segment"].unique()[:10])
# seg = df[df["segment"] == df["segment"].unique()[0]]     # first segment that exists
# print(len(seg), seg["class"].iloc[0])

# seg = df[df["segment"] == 1]                      # pick a segment by its ID
# plt.plot(seg["channel1"].to_numpy()[500:700], ".")  # rows 500 to 700 of that segment
# plt.show()

#plt.plot(df.loc[df["label"] == 1, 'channel1'].to_numpy()) # plot signal (remove)


X_mav = np.array(mav_list)          # (N, 8)
X_ar = np.array(ar_list)
X_mnf = np.array(mnf_list)

y_labels = np.array(y_labels)       # (N,)
subject_ids = np.array(subject_ids)
segment_ids = np.array(segment_ids)
window_lengths = np.array(window_lengths)

print(f"MAV feature matrix: {X_mav.shape}")
print(f"AR feature matrix: {X_ar.shape}")
print(f"MNF feature matrix: {X_mnf.shape}")
#print(f"Window lengths: {dict(zip(*np.unique(window_lengths, return_counts=True)))}")


#%%
# --------------------------------------------------------------------- VISUALIZATION OF FEATURES

# ===================== MAV =====================
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

# ======================== AR =========================
X_ar3 = X_ar.reshape(len(X_ar), N_CHANNELS, AR_ORDER) # (N, 8, 4)

fig, axs = plt.subplots(2, 2, figsize=(14, 7))
vmax = np.abs(X_ar3).mean()   # placeholder, replaced below
tables = []
for k in range(AR_ORDER):
    t = pd.DataFrame(X_ar3[:, :, k], columns=[f"ch{i}" for i in range(1, 9)])
    t["class"] = y_labels
    tables.append(t.groupby("class").mean())

vmax = max(np.abs(t.values).max() for t in tables)   # same colour scale for all four

for k, ax in enumerate(axs.flat):
    sns.heatmap(tables[k], annot=True, fmt=".2f", cmap="coolwarm",
                center=0, ax=ax)
    ax.set_title(f"Mean AR{k + 1} per class and channel")
plt.tight_layout()
plt.show()

# ====================== MNF ========================================
mnf_df = pd.DataFrame(X_mnf, columns=[f"ch{i}" for i in range(1, 9)])
mnf_df["class"] = y_labels

mean_table = mnf_df.groupby("class").mean()
plt.figure(figsize=(10, 4))
sns.heatmap(mean_table, annot=True, fmt=".1f", cmap="viridis")
plt.title("Mean MNF (Hz) per class and channel")
plt.show()

fig, axs = plt.subplots(2, 4, figsize=(16, 7), sharex=True)
for i, ax in enumerate(axs.flat):
    sns.boxplot(data=mnf_df, x="class", y=f"ch{i + 1}", ax=ax, showfliers=False)
    ax.set_title(f"Channel {i + 1}")
plt.tight_layout()
plt.show()


#%%

from sklearn.ensemble import RandomForestClassifier

# 1. Combine all existing feature matrices horizontally
X_combined = np.hstack((X_mav, X_ar, X_mnf))

# 2. Train a baseline Random Forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_combined, y_labels)

# 3. Create labels to identify each feature in the chart
mav_labels = [f"MAV_ch{i}" for i in range(1, 9)]
ar_labels = [f"AR{k}_ch{i}" for i in range(1, 9) for k in range(1, 5)]
mnf_labels = [f"MNF_ch{i}" for i in range(1, 9)]
feature_names = mav_labels + ar_labels + mnf_labels

# 4. Extract and plot the importances
importances = rf.feature_importances_
importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
importance_df = importance_df.sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10, 10))
sns.barplot(data=importance_df, x="Importance", y="Feature", palette="viridis")
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.show()

#%%
# ------------------------------------ SVM TRAINING

X_combined = np.hstack((X_mav, X_ar, X_mnf))

X_train, X_test, y_train, y_test = train_test_split(
    X_combined, 
    y_labels, 
    test_size=0.2, 
    random_state=42, 
    stratify=y_labels
)

scaler = StandardScaler()

# Fit the scaler ONLY on the training data, then transform it
X_train_scaled = scaler.fit_transform(X_train)

# Transform the test data using the parameters learned strictly from the training set
X_test_scaled = scaler.transform(X_test)


# ================ Linear SVM for Baseline ========================
svm_linear = SVC(kernel='linear', C=1.0, random_state=42)
svm_linear.fit(X_train_scaled, y_train)

y_pred_linear = svm_linear.predict(X_test_scaled)

print("--- Linear SVM Results ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_linear):.4f}")
print(classification_report(y_test, y_pred_linear))


# ================ RBF SVM ====================
svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_rbf.fit(X_train_scaled, y_train)

y_pred_rbf = svm_rbf.predict(X_test_scaled)

print("--- RBF SVM Results ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rbf):.4f}")
print(classification_report(y_test, y_pred_rbf))

# ================== Poly SVM =============================
svm_poly = SVC(kernel='poly', degree=3, C=1.0, coef0=1.0, random_state=42)
svm_poly.fit(X_train_scaled, y_train)

y_pred_poly = svm_poly.predict(X_test_scaled)

print("--- Polynomial SVM (Degree 3) Results ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_poly):.4f}")
print(classification_report(y_test, y_pred_poly))

#%%

# FS PRECOMPUTATION
periods = []
for seg_id, g in df[df["class"].isin([2, 3, 4, 5, 6])].groupby("segment"):
    vals = g[channel_cols_all].to_numpy()
    t = g["time"].to_numpy()
    changed = np.any(vals[1:] != vals[:-1], axis=1)
    periods.append(np.diff(t[1:][changed]))     # time between consecutive new values (ms)
periods = np.concatenate(periods)


#%%
