import pandas as pd

if __name__ == "__main__":
    for case_index in [0, 1, 2, 3]:
        df = pd.read_csv(f"data/strips/case_{case_index}.csv")
        df = df.mean(1)
        df.to_csv(f"data/mean_strips/case_{case_index}.csv", index=False)
