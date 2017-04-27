import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle
import numpy as np

H_depths = [1, 5, 10];
colors = ["r", "g", "b"];
datasets = [];
ax = None;
labels = []

for H_depth, color in zip(H_depths, colors):
    data = pickle.load(open("norm_errs_H" + str(H_depth) + ".pkl", "rb"));
    datasets.append(data);
    labels.append("horizon = " + str(H_depth));

bigdata = np.zeros((datasets[0].shape[0], datasets[0].shape[1], len(datasets)));
for ii, data in enumerate(datasets):
    bigdata[:, :, ii] = data;

sns.set(font_scale=1.5);
sns.tsplot(data=bigdata, condition=labels, legend=True);
plt.title("L2 norm of errors as a function of timestep");
plt.xlabel("Timestep");
plt.ylabel("L2 difference between true and estimated state");
plt.show();
