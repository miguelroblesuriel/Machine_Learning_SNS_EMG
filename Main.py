# irjfirjfeirjf

import pandas as pd
import numpy as np
import seaborn as sns
from Visualization.plot_signal import plot_signal


import numpy as np


def extract_signals(df):
    subject_signals = []
    channel_cols = [f"channel{k + 1}" for k in range(8)]

    for (label, class_val), group in df.groupby(["label", "class"]):
        time = group["time"].to_numpy()

        split_indices = np.where(np.diff(time) < 0)[0] + 1

        time_splits = np.split(time, split_indices)

        values_splits = [
            np.split(group[col].to_numpy(), split_indices)
            for col in channel_cols
        ]

        signal_values_1 = [val_split[0] for val_split in values_splits]
        signal_values_2 = (
            [val_split[1] for val_split in values_splits]
            if len(split_indices) > 0
            else []
        )

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

input_fileroute = "Project_Data_EE4C12_S&S_EMG.csv"
df = pd.read_csv(input_fileroute)
print(df.head())
signal_value = df.loc[(df["label"] == 1) & (df["class"] == 1), "channel1"]
signal_time = df.loc[(df["label"] == 1) & (df["class"] == 1), "time"]
print(df.loc[(df["label"] == 1) & (df["class"] == 1), "time"])
plot_signal(signal_time, signal_value)
subject_signals = extract_signals(df)
value_previous =0
for value in signal_time:
    if value < value_previous:
        print("a")
    value_previous = value
for subject_signal in subject_signals:
    plot_signal(subject_signal['signal_time_1'], subject_signal['signal_values_1'][0])
    plot_signal(subject_signal['signal_time_2'], subject_signal['signal_values_2'][0])
