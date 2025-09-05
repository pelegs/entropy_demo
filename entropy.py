import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from pandas._libs import interval
from tqdm import tqdm

particles: int = 10000
buckets: int = 100
steps: int = 1000

data = np.zeros(shape=(steps, particles), dtype=int)
data[0] = np.ones(particles) * np.random.randint(buckets)
# data[0] = np.random.randint(0, buckets, size=particles)

histograms = np.zeros(shape=(steps, buckets + 1))
# for explanation on bins, see here:
# https://stackoverflow.com/a/53017960/6194420
bins = np.arange(0, buckets + 1.5)
histograms[0], _ = np.histogram(data[0], bins)

for t in tqdm(range(1, steps)):
    data[t] = data[t - 1] + np.random.randint(-1, 2, size=particles)
    for i, idx in enumerate(data[t]):
        if idx >= buckets:
            data[t, i] = buckets - 1
        if idx < 0:
            data[t, i] = 0
    histograms[t], _ = np.histogram(data[t], bins)


fig, ax = plt.subplots()
bars = ax.bar(bins[:-1], histograms[0])
ax.set_title(label="Distribution of particles, t=0", fontsize=25)
ax.set_ylim(top=particles)  # set safe limit to ensure that all data is visible.
ax.set_xlabel("Bucket", fontsize=20)
ax.set_ylabel("# of particles in bucket", fontsize=20)


def animate(frame_number):
    ax.set_title(f"Distribution of particles, t={frame_number}", fontsize=25)
    for rect, h in zip(bars, histograms[frame_number, :]):
        rect.set_height(h)
    return bars


ani = animation.FuncAnimation(fig, animate, steps, interval=0, repeat=False, blit=False)
plt.show()
