import matplotlib.pyplot as plt
import numpy as np
import os
import plotly.graph_objects as go

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


def plot_grid_heatmaps(tensor, layer_names, stat_names, args, type):
    out_path = "plots/" + args.task + "_" + type + "_heatmap.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    steps = tensor.shape[0]
    fig, axs = plt.subplots(len(stat_names), 1, figsize=(15, 10 * len(stat_names)), squeeze=False)
    for i, stat in enumerate(stat_names):
        ax = axs[i, 0]
        im = ax.imshow(tensor[:, :, i].T, aspect='auto', cmap='viridis', origin='lower')
        ax.set_title(stat)
        ax.set_xticks(np.arange(steps))
        ax.set_yticks(np.arange(len(layer_names)))
        ax.set_yticklabels(layer_names)
        plt.xlabel("Every "+str(args.logf)+" Steps")
        fig.colorbar(im, ax=ax)
    plt.savefig(out_path)
    plt.close()

def plot_interactive_3d(tensor, layer_names, stat_names, args, type):
    out_path="plots/" + args.task + "_" + type + "_" + "interactive_3d.html"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    steps, layers, stats = tensor.shape
    fig = go.Figure()
    for i in range(stats):
        z = tensor[:, :, i].T
        fig.add_trace(go.Surface(z=z, name=stat_names[i]))
    fig.update_layout(title="Layer Stats 3D", scene=dict(
        xaxis_title="Step",
        yaxis_title="Layer",
        zaxis_title="Stat Value",
        yaxis=dict(tickmode='array', tickvals=list(range(len(layer_names))), ticktext=layer_names)
    ))
    fig.write_html(out_path)