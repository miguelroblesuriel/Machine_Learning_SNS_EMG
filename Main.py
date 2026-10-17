# This file is a small exploratory script for EMG signal inspection.
# It loads the dataset, filters specific label/class combinations, and plots the signal.
# I would not remove anything here yet, because the script is still useful for debugging and visual validation.
# The duplicate NumPy import is redundant, but it is harmless and I left it in place to avoid changing working code.

import pandas as pd
import numpy as np
import seaborn as sns
from sympy import true
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

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
        times_1 = time_splits[0]
        times_2 = time_splits[1]
        signal_matrix_1 = signal_values_1
        signal_matrix_2 = signal_values_2
        cutoff_idx = len(times_1)
        value_previous = times_1[0]
        for i, time_1 in enumerate(times_1):
            if time_1 > value_previous + 1000:
                cutoff_idx = i
                break
            value_previous = time_1

        tiempo_1_1 = times_1[:cutoff_idx]
        tiempo_1_2 = times_1[cutoff_idx:]

        señal_1_1 = [channel[:cutoff_idx] for channel in signal_matrix_1]
        señal_1_2 = [channel[cutoff_idx:] for channel in signal_matrix_1]

        cutoff_idx = len(times_2)
        value_previous = times_2[0]
        for i, time_2 in enumerate(times_2):
            if time_2 > value_previous + 1000:
                cutoff_idx = i
                break
            value_previous = time_2

        tiempo_2_1 = times_2[:cutoff_idx]
        tiempo_2_2 = times_2[cutoff_idx:]

        señal_2_1 = [channel[:cutoff_idx] for channel in signal_matrix_2]
        señal_2_2 = [channel[cutoff_idx:] for channel in signal_matrix_2]

        # Store in dictionary
        subject_signals.append({
            'subject': label,
            'signal_values_1': señal_1_1,
            'signal_time_1': tiempo_1_1,
            'signal_values_2': señal_2_1,
            'signal_time_2': tiempo_2_1
            if len(time_splits) > 1
            else np.array([]),
            'class': class_val,
        })

        subject_signals.append({
            'subject': label,
            'signal_values_1': señal_1_2,
            'signal_time_1': tiempo_1_2,
            'signal_values_2': señal_2_2,
            'signal_time_2': tiempo_2_2
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


def normalize_and_pad(signal_list, mean, std, target_len):
    n_samples = len(signal_list)
    n_features = mean.shape[0]

    padded_arr = np.zeros((n_samples, n_features, target_len))

    for i, sample in enumerate(signal_list):
        sample_arr = np.array(sample)

        norm_sample = (sample_arr - mean[:, np.newaxis]) / std[:, np.newaxis]

        length = norm_sample.shape[1]
        copy_len = min(length, target_len)

        padded_arr[i, :, :copy_len] = norm_sample[:, :copy_len]

    return padded_arr

def extract_data(subject_subset, target_signals, target_classes, subject_signals):
    for subject in subject_subset:
        sub_df = subject_signals[subject_signals['subject'] == subject]

        for clase in sub_df['class'].unique():
            class_df = sub_df[sub_df['class'] == clase]

            for _, row in class_df.iterrows():
                target_signals.append(row['signal_values_1'])
                target_classes.append(clase)

                target_signals.append(row['signal_values_2'])
                target_classes.append(clase)

def split_by_subjects(subject_signals, training_ratio, validation_ratio):
    training_signals = []
    testing_signals = []
    validation_signals = []
    training_classes = []
    testing_classes = []
    validation_classes = []
    subject_signals = pd.DataFrame(subject_signals)
    subjects = subject_signals['subject'].unique()
    random_subjects = np.random.permutation(subjects)

    total = len(random_subjects)
    train_end = int(total * training_ratio)
    val_end = train_end + int(total * validation_ratio)

    train_subjects = random_subjects[:train_end]
    val_subjects = random_subjects[train_end:val_end]
    test_subjects = random_subjects[val_end:]

    extract_data(train_subjects, training_signals, training_classes, subject_signals)
    extract_data(val_subjects, validation_signals, validation_classes, subject_signals)
    extract_data(test_subjects, testing_signals, testing_classes, subject_signals)

    all_train_data = np.hstack([np.array(sample) for sample in training_signals])
    print(all_train_data[0])

    global_mean = np.mean(all_train_data, axis=1)
    print(global_mean)
    global_std = np.std(all_train_data, axis=1)
    global_std[global_std == 0] = 1.0

    all_signals = training_signals + testing_signals + validation_signals
    max_len = max(np.array(sample).shape[1] for sample in all_signals)

    training_signals = normalize_and_pad(training_signals, global_mean, global_std, max_len)
    testing_signals = normalize_and_pad(testing_signals, global_mean, global_std, max_len)
    validation_signals = normalize_and_pad(validation_signals, global_mean, global_std, max_len)

    return (
        training_signals,
        testing_signals,
        validation_signals,
        np.array(training_classes),
        np.array(testing_classes),
        np.array(validation_classes)
    )

def extract_from_df(df, signals_list, classes_list):
    # Ajusta esto según cómo guardes las señales y clases en tus columnas
    for _, row in df.iterrows():
        signals_list.append(row['signal'])  # Cambia 'signal' por el nombre de tu columna
        classes_list.append(row['class'])
    return signals_list,classes_list

def split_total_samples(subject_signals, training_ratio, validation_ratio):
    training_signals = []
    testing_signals = []
    validation_signals = []
    training_classes = []
    testing_classes = []
    validation_classes = []

    subject_signals = pd.DataFrame(subject_signals)

    shuffled_data = subject_signals.sample(frac=1, random_state=42).reset_index(drop=True)

    total_samples = len(shuffled_data)
    train_end = int(total_samples * training_ratio)
    val_end = train_end + int(total_samples * validation_ratio)

    train_df = shuffled_data.iloc[:train_end]
    val_df = shuffled_data.iloc[train_end:val_end]
    test_df = shuffled_data.iloc[val_end:]

    training_signals, training_classes = extract_from_df(train_df, training_signals, training_classes)
    validation_signals, validation_classes = extract_from_df(val_df, validation_signals, validation_classes)
    testing_signals, testing_classes = extract_from_df(test_df, testing_signals, testing_classes)

    all_train_data = np.hstack([np.array(sample) for sample in training_signals])
    print(all_train_data[0])

    global_mean = np.mean(all_train_data, axis=1)
    print(global_mean)
    global_std = np.std(all_train_data, axis=1)
    global_std[global_std == 0] = 1.0

    all_signals = training_signals + testing_signals + validation_signals
    max_len = max(np.array(sample).shape[1] for sample in all_signals)

    training_signals = normalize_and_pad(training_signals, global_mean, global_std, max_len)
    testing_signals = normalize_and_pad(testing_signals, global_mean, global_std, max_len)
    validation_signals = normalize_and_pad(validation_signals, global_mean, global_std, max_len)

    return (
        training_signals,
        testing_signals,
        validation_signals,
        np.array(training_classes),
        np.array(testing_classes),
        np.array(validation_classes)
    )


input_fileroute = "Project_Data_EE4C12_S&S_EMG.csv"

df = pd.read_csv(input_fileroute)

print(df.head())

signal_value = df.loc[(df["label"] == 1) & (df["class"] == 1), "channel1"]
signal_time = df.loc[(df["label"] == 1) & (df["class"] == 1), "time"]

print(df.loc[(df["label"] == 1) & (df["class"] == 1), "time"])

plot_signal(signal_time, signal_value)

# Extract segmented signals for all groups.
subject_signals = extract_signals(df)
for subject_signal in subject_signals:
    if subject_signal["subject"] == 1:
        print(subject_signal["class"])
        plot_signal(subject_signal["signal_time_1"], subject_signal["signal_values_1"][6])




training_x, testing_x, validation_x, training_y, testing_y, validation_y = split_by_subjects(subject_signals,0.6,0.2)
print(training_x[0])
print(len(training_y))

training_x2, testing_x2, validation_x2, training_y2, testing_y2, validation_y2 = split_by_subjects(subject_signals,0.6,0.2)
print(len(training_x2))
print(len(training_y2))

n_samples_train = training_x.shape[0]
training_x_flat = training_x.reshape(n_samples_train, -1)

n_samples_val = validation_x.shape[0]
validation_x_flat = validation_x.reshape(n_samples_val, -1)

n_samples_test = testing_x.shape[0]
testing_x_flat = testing_x.reshape(n_samples_test, -1)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(training_x_flat, training_y)
val_pred = rf_model.predict(validation_x_flat)
val_accuracy = accuracy_score(validation_y, val_pred)
print(f"Precisión en Validación: {val_accuracy * 100:.2f}%\n")
test_pred = rf_model.predict(testing_x_flat)
print("Reporte de clasificación final (Testing):")
print(classification_report(testing_y, test_pred))

n_samples_train2 = training_x2.shape[0]
training_x_flat2 = training_x2.reshape(n_samples_train2, -1)
print(len(training_x_flat2))

n_samples_val2 = validation_x2.shape[0]
validation_x_flat2 = validation_x2.reshape(n_samples_val2, -1)

n_samples_test2 = testing_x2.shape[0]
testing_x_flat2 = testing_x2.reshape(n_samples_test2, -1)

rf_model2 = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model2.fit(training_x_flat2, training_y2)
val_pred2 = rf_model2.predict(validation_x_flat2)
val_accuracy2 = accuracy_score(validation_y2, val_pred2)
print(f"Precisión en Validación: {val_accuracy2 * 100:.2f}%\n")
test_pred2 = rf_model2.predict(testing_x_flat2)
print("Reporte de clasificación final (Testing):")
print(classification_report(testing_y2, test_pred2))