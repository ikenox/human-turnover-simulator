#!/usr/bin/env python

import argparse

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

from lib.util import save_and_close_plt


def main():

    parser = argparse.ArgumentParser(description='Plot 3D scatter graphs of human turnover simulations from csv.')
    parser.add_argument('file', type=str,
                       help='csv file inputted')

    args = parser.parse_args()

    df = pd.read_csv(args.file)

    fig = plt.figure()
    ax = Axes3D(fig)

    ax.set_xlabel(r"$\alpha_S$")
    ax.set_ylabel(r"$\alpha_L$")
    ax.set_zlabel(r"Skewness of $\log_{10}\tau$")

    ax.set_xlim(0.45, 0.8)
    ax.set_ylim(0.45, 0.8)
    ax.set_zlim(2.3, -2.3)

    d_wake = df[df.stage == 'wake']
    d_sleep_disturb = df[(df.stage == 'sleep') & (df.alpha_l > 0.55)]
    d_sleep_undisturb = df[(df.stage == 'sleep') & (df.alpha_l <= 0.55)]


    ax.scatter3D(d_sleep_disturb['alpha_s'], d_sleep_disturb['alpha_l'], d_sleep_disturb['skew'], "o", color="#00FF00", s=2)
    ax.scatter3D(d_sleep_undisturb['alpha_s'], d_sleep_undisturb['alpha_l'], d_sleep_undisturb['skew'], "o", color="#0000FF", s=2)
    ax.scatter3D(d_wake['alpha_s'], d_wake['alpha_l'], d_wake['skew'], "o", color="gold", s=2)

    save_and_close_plt(plt, '3dscat')

if __name__ == '__main__':
    main()