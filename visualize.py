import matplotlib.pyplot as plt
import numpy as np
import os
import plotly.graph_objects as go

def plot_grid_heatmaps(tensor, layer_names, stat_names, args, type):
    os.makedirs("plots/heatmaps/", exist_ok=True)
    out_path = "plots/heatmaps/" + args.task + "_" + type + ".png"
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
    os.makedirs("plots/heatmaps", exist_ok=True)
    out_path="plots/heatmaps/" + args.task + "_" + type + ".html"
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