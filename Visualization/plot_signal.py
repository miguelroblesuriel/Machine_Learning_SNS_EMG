import matplotlib.pyplot as plt
def plot_signal(signal_time, signal_value,  title="EMG Signal over Time"):
    plt.figure(figsize=(10, 4))
    plt.plot(signal_time, signal_value, color="tab:blue", linewidth=1.5)

    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Signal Value")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()