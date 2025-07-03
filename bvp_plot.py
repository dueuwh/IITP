import os
import numpy as np
import matplotlib.pyplot as plt

base_path = "C:/Users/U/Desktop/BCML/IITP/IITP_old/IITP_emotions/data/senior/sychro/results/rppg_toolbox/bvp/CHROM/"
bvps = [name for name in os.listdir(base_path) if "bvp" in name]

for bvp in bvps:
    load_bvp = np.load(os.path.join(base_path, bvp))
    plt.plot(load_bvp)
    plt.show()
