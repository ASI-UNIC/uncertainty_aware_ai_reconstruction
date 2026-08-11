from dataset import get_dataloaders
import pandas as pd
import os

if __name__ == "__main__":
    os.makedirs("data/strips", exist_ok=True)

    for case_index in [0, 1, 2, 3]:
        (
            train_loader,
            global_min,
            global_max,
            global_mean,
            global_std,
        ) = get_dataloaders(
            data_dir="./data",
            batch_size=1,
            input_dim=384,
            seq_len=2,
            step_size=2,
            prediction_horizon=1,
            normalize=False,
            train_ratio=0.8,
            task="interpolation",
            num_workers=0,
            sparsity=1,
            interpolation_method="cubic",
            start_dim=0,
            case_index=case_index,
            get_strips=True,
        )

        data_list = train_loader.dataset.data_list
        pd.DataFrame(data_list[0]).transpose().to_csv(
            f"data/strips/case_{case_index}.csv", index=False
        )
