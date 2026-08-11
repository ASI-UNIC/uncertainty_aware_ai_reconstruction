import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":
    os.makedirs('data/attractors', exist_ok=True)

    for case_index in [0, 1, 2, 3]:
        df = pd.read_csv(f"data/mean_strips/case_{case_index}.csv")
        pressure = np.array(df.iloc[:, 0])
        
        x = pressure[:-1]
        y = pressure[1:]

        plt.figure()
        plt.scatter(x, y, s=5)
        plt.xlabel("p(t)")
        plt.ylabel("p(t + dt)")
        plt.title(f"Case {case_index}")
        plt.show()