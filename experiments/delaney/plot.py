import argparse

import numpy as np
import os
import sys

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from saliency.visualizer.smiles_visualizer import SmilesVisualizer


def visualize(dir_path):
    parent_dir = os.path.dirname(dir_path)
    saliency_vanilla = np.load(os.path.join(dir_path, "saliency_vanilla.npy"))
    saliency_smooth = np.load(os.path.join(dir_path, "saliency_smooth.npy"))
    saliency_bayes = np.load(os.path.join(dir_path, "saliency_bayes.npy"))

    visualizer = SmilesVisualizer()
    os.makedirs(os.path.join(parent_dir, "result_vanilla"), exist_ok=True)
    os.makedirs(os.path.join(parent_dir, "result_smooth"), exist_ok=True)
    os.makedirs(os.path.join(parent_dir, "result_bayes"), exist_ok=True)

    test_idx = np.load(os.path.join(dir_path, "test_idx.npy"))
    answer = np.load(os.path.join(dir_path, "answer.npy"))
    output = np.load(os.path.join(dir_path, "output.npy"))

    smiles_all = np.load(os.path.join(parent_dir, "smiles.npy"))

    def calc_range(saliency):
        vmax = float('-inf')
        vmin = float('inf')
        for v in saliency:
            vmax = max(vmax, np.max(v))
            vmin = min(vmin, np.min(v))
        return vmin, vmax

    v_range_vanilla = calc_range(saliency_vanilla)
    v_range_smooth = calc_range(saliency_smooth)
    v_range_bayes = calc_range(saliency_bayes)

    def get_scaler(v_range):
        def scaler(saliency_):
            saliency = np.copy(saliency_)
            minv, maxv = v_range
            if maxv == minv:
                saliency = np.zeros_like(saliency)
            else:
                pos = saliency >= 0.0
                saliency[pos] = saliency[pos]/maxv
                nega = saliency < 0.0
                saliency[nega] = saliency[nega]/(np.abs(minv))
            return saliency
        return scaler

    scaler_vanilla = get_scaler(v_range_vanilla)
    scaler_smooth = get_scaler(v_range_smooth)
    scaler_bayes = get_scaler(v_range_bayes)

    def color(x):
        if x > 0:
            # Red for positive value
            return 1., 1. - x, 1. - x
        else:
            # Blue for negative value
            x *= -1
            return 1. - x, 1. - x, 1.

    for i, id in enumerate(test_idx):
        smiles = smiles_all[id]
        out = output[i]
        ans = answer[i]
        # legend = "t:{}, p:{}".format(ans, out)
        legend = ''
        ext = '.png'  # '.svg'
        # visualizer.visualize(
        #     saliency_vanilla[id], smiles, save_filepath=os.path.join(parent_dir, "result_vanilla", str(id) + ext),
        #     visualize_ratio=1.0, legend=legend, scaler=scaler_vanilla, color_fn=color)
        # visualizer.visualize(
        #     saliency_smooth[id], smiles, save_filepath=os.path.join(parent_dir, "result_smooth", str(id) + ext),
        #     visualize_ratio=1.0, legend=legend, scaler=scaler_smooth, color_fn=color)
        visualizer.visualize(
            saliency_bayes[id], smiles, save_filepath=os.path.join(parent_dir, "result_bayes", str(id) + ext),
            visualize_ratio=1.0, legend=legend, scaler=scaler_bayes, color_fn=color)


def plot_result(prediction, answer, save_filepath='result.png'):
    plt.scatter(prediction, answer, marker='.')
    plt.plot([-100, 100], [-100, 100], c='r')
    max_v = max(np.max(prediction), np.max(answer))
    min_v = min(np.min(prediction), np.min(answer))
    plt.xlim([min_v-0.1, max_v+0.1])
    plt.xlabel("prediction")
    plt.ylim([min_v-0.1, max_v+0.1])
    plt.ylabel("ground truth")
    plt.savefig(save_filepath)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Regression with own dataset.')
    parser.add_argument('--dirpath', '-d', type=str, default='./results/M_30_3_32_32')
    args = parser.parse_args()
    path = args.dirpath
    n_split = 5
    output = []
    answer = []
    for i in range(n_split):
        suffix = str(i) + "-" + str(n_split)
        output.append(np.load(os.path.join(path, suffix, "output.npy")))
        answer.append(np.load(os.path.join(path, suffix, "answer.npy")))
    output = np.concatenate(output)
    answer = np.concatenate(answer)

    plot_result(output, answer, save_filepath=os.path.join(path, "result.png"))
    for i in range(n_split):
        suffix = str(i) + "-" + str(n_split)
        print(suffix)
        visualize(os.path.join(path, suffix))


if __name__ == '__main__':
    main()

