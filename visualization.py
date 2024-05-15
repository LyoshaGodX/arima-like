import matplotlib.pyplot as plt


def plot_timeseries(x, y, title, ylabel):
    plt.figure(figsize=(12, 6))
    plt.plot(x, y, label=ylabel)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel(ylabel)
    plt.grid(True)
    labels = [x[i] if i%12 == 0 else "" for i in range(len(x))]  # Generate labels
    plt.xticks(range(len(x)), labels, rotation=45, fontsize=8)  # Set x-ticks
    plt.legend()
    plt.tight_layout()


def plot_residuals(x, residuals, title):
    labels = [str(i) for i in range(1998, 1999 + len(x) // 12)]
    ticks = range(0, len(x), 12)

    # Ensure the number of labels matches the number of ticks
    if len(ticks) > len(labels):
        ticks = ticks[:len(labels)]
    elif len(labels) > len(ticks):
        labels = labels[:len(ticks)]

    plt.figure(figsize=(12, 6))
    plt.plot(residuals)
    plt.title(title)
    plt.xticks(ticks, labels, rotation=45, fontsize=8)
    plt.grid(True)
    plt.show()
