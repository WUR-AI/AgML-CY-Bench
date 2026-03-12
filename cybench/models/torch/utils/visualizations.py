import pandas as pd
import matplotlib.pyplot as plt

# test_mses = pd.Series(fit_info["val_loss"]).dropna()
# plt.plot(range(1, len(fit_info["train_loss"]) + 1), fit_info["train_loss"], label=f"train MSE ({fit_info['train_loss'][-1]:.4})")
# plt.plot(test_mses.index + 1, test_mses, label=f"test MSE  ({test_mses.values[-1]:.4})")
# plt.hlines(y=2.55, xmin=0, xmax=len(fit_info["train_loss"]), label="test MSE of RollAvgModel", color="tab:grey", linestyle="dashed")
# plt.hlines(y=1.75, xmin=0, xmax=len(fit_info["train_loss"]), label="test MSE of LightGBM", color="tab:red", linestyle="dashed")
# plt.legend()
# plt.xlabel("Epochs")
# plt.ylabel("MSE")
# plt.ylim((min(fit_info["train_loss"]) - 0.1, 3))
# plt.title \
#     (f'{model.name} Training progress | {cfg.dataset.country} ({cfg.dataset.crop}) testing on {"_".join(str(x) for x in years_split[1])}')
# # plt.grid(axis="y")
# # plt.savefig("US_maize_benchmark_CNN.png", dpi=300)
# plt.show()