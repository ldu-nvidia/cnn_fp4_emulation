import matplotlib.pyplot as plt
import numpy as np

def log_2d_histogram(wandb, wandb_key, layer_indices, values, title, ylabel, epoch):
    """
    Plots and logs a 2D histogram of values vs layer index to wandb.
    """
    layer_indices, values = np.array(layer_indices), np.array(values)
    if len(values) == 0:
        return  # Nothing to plot
    H, xedges, yedges = np.histogram2d(layer_indices, values, bins=[50, 100])
    plt.figure(figsize=(12, 6))
    plt.imshow(H.T, aspect='auto', origin='lower',
               extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    plt.xlabel("Layer Index")
    plt.ylabel(ylabel)
    plt.title(f"{title} at Epoch {epoch}")
    plt.colorbar(label="Counts")
    wandb.log({f"{wandb_key}_Epoch_{epoch}": wandb.Image(plt)})
    plt.close()
