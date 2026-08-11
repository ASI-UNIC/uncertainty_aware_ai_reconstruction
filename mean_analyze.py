import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    for case_index in [0, 1, 2, 3]:
        df = pd.read_csv(f"data/strips/case_{case_index}.csv")
        mean, std = df.mean(1), df.std(1)
        percentage_deviated = std / mean * 100

        plt.plot(percentage_deviated)
        plt.title(f"Case {case_index}")
        plt.xlabel("Features")
        plt.ylabel("Percentage Deviated (%) [std/mean * 100%]")
        plt.savefig(f"outputs/data_analysis/percentage_deviated_case_{case_index}.png")
        plt.clf()
        plt.close()
