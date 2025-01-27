import matplotlib.pyplot as plt
import pandas as pd

from coolchic.eval.hypernet import plot_hypernet_rd

if __name__ == "__main__":
    results = pd.read_csv("kodak_results.csv")
    for kodim_name in [f"kodim{i:02d}" for i in range(1, 25)]:
        plot_hypernet_rd(kodim_name, results)
    plt.show()
