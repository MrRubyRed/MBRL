import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle
import numpy as np

H_depths = [1, 5, 10];
colors = ["r", "g", "b"];
axs = []
patches = []
labels = []

for H_depth, color in zip(H_depths, colors):
    data = pickle.load(open("norm_errs_H" + str(H_depth) + ".pkl", "rb"));
    labels.append("horizon = " + str(H_depth));
    patches.append(mpatches.Patch(color=color, label=labels[-1]));
    axs.append(sns.tsplot(data, color=color));
    axs[-1].legend([patches[-1]]);

#plt.legend(handles=patches);
plt.legend();
plt.title("L2 norm of errors as a function of timestep");
plt.xlabel("Timestep");
plt.ylabel("L2 difference between true and estimated state");
plt.show();
