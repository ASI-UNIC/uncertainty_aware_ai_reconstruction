import numpy as np
from dataset import get_dataloaders
import matplotlib.pyplot as plt


def calculate_rolling_mean_standard(data_list, window_size=5):
    """
    Calculates the rolling mean using a standard Python list comprehension.

    The rolling mean starts once the window_size is reached.
    The first (window_size - 1) elements will not have a rolling mean value.

    Args:
        data_list (list): The list of numbers.
        window_size (int): The size of the moving window (default is 5).

    Returns:
        list: A list of rolling means.
    """

    if window_size <= 0 or window_size > len(data_list):
        raise ValueError(
            "Window size must be a positive integer and less than or equal to the list length."
        )

    # Rolling mean calculation:
    # 1. Iterate from the (window_size - 1) index up to the end of the list.
    # 2. For each index 'i', calculate the average of the slice
    #    from 'i - window_size + 1' up to 'i' (which gives a window of 'window_size' items).
    rolling_means = [
        sum(data_list[i - window_size + 1 : i + 1]) / window_size
        for i in range(window_size - 1, len(data_list), window_size)
    ]

    return rolling_means


def calculate_moving_average(data_list, window_size=5):
    # 2. Create the Weights Array (1/n for each element)
    weights = np.ones(window_size) / window_size
    # weights will be [0.2, 0.2, 0.2, 0.2, 0.2]

    # 3. Perform Convolution
    sma_numpy = np.convolve(data_list, weights, mode="valid")
    return sma_numpy


if __name__ == "__main__":
    start_dim = int(74112 / 2)

    all_rolling_mean = {}
    all_moving_average = {}

    for window_size in [5, 10, 15, 20]:
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
                input_dim=1,
                seq_len=2,
                step_size=2,
                prediction_horizon=1,
                normalize=False,
                train_ratio=0.8,
                task="interpolation",
                num_workers=0,
                sparsity=1,
                interpolation_method="cubic",
                start_dim=start_dim,
                case_index=case_index,
            )

            sequences = np.array(train_loader.dataset.sequences[0])
            sequences = sequences.reshape([-1, sequences.shape[-1]])

            # calculate the mean of sequences
            rolling_means = calculate_rolling_mean_standard(
                sequences[:, 0], window_size
            )
            moving_average = calculate_moving_average(sequences[:, 0], window_size)

            all_rolling_mean[case_index] = rolling_means
            all_moving_average[case_index] = moving_average

        for case_index in all_rolling_mean:
            plt.plot(all_rolling_mean[case_index], label=f"case_{case_index}")
        plt.legend()
        plt.title("Rolling Mean with window size of {}".format(window_size))
        plt.savefig("outputs/data_analysis/rolling_mean_{}.png".format(window_size))
        plt.clf()
        plt.close()

        for case_index in all_moving_average:
            plt.plot(all_moving_average[case_index], label=f"case_{case_index}")
        plt.legend()
        plt.title("Moving Average with window size of {}".format(window_size))
        plt.savefig("outputs/data_analysis/moving_average_{}.png".format(window_size))
        plt.clf()
        plt.close()
