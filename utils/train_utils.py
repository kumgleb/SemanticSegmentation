import numpy as np
import matplotlib.pyplot as plt


def train_monitor(losses_train, losses_train_mean, losses_val):

        fig, ax = plt.subplots(1, 2, figsize=(16, 8))

        iters = np.arange(len(losses_train))
        n_vals = len(losses_val)
        step = int(len(losses_train) / n_vals)
        val_steps = np.linspace(step, step*n_vals, n_vals)

        for i in range(2):
            ax[i].plot(iters, losses_train, linewidth=1.5, alpha=0.6,
                       c='tab:blue', label='train loss')
            ax[i].plot(iters, losses_train_mean, linewidth=2, alpha=1,
                       c='tab:blue', label='avg10 train loss')
            ax[i].plot(val_steps, losses_val, linewidth=2, alpha=1,
                       c='tab:red', label='val loss')
            ax[i].set_ylabel('CrossEntropy loss')
            ax[i].set_xlabel('Iteration')
            ax[i].legend()
            ax[i].grid()
            if i == 1:
                ax[i].set_yscale('log')
        plt.show()

