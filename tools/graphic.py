# Based on work: https://int-i.github.io/python/2021-11-07/matplotlib-google-benchmark-visualization/

from argparse import ArgumentParser
from itertools import groupby
from cycler import cycler
from random import randint
import json
import math
import operator
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd


def generate_color(size):
    colors = []

    for i in range(size):
        colors.append('#%06X' % randint(0, 0xFFFFFF))

    colors = sorted(set(colors), key=colors.index)  # Remove all same elements
    return colors


def extract_label_from_benchmark(benchmark):
    bench_full_name = benchmark['name']
    bench_name = bench_full_name.split('/')[0]  # Remove all after /
    if (bench_name.startswith('BM_')):  # Remove if string start with BM_
        return bench_name[3:]  # Remove BM_
    else:
        return bench_name


def extract_size_from_benchmark(benchmark):
    bench_name = benchmark['name']
    return bench_name.split('/')[1]  # Remove all before /


if __name__ == "__main__":
    plt.rcParams['figure.figsize'] = [21, 12]
    mpl.rcParams['axes.prop_cycle'] = cycler(color=generate_color(200))

    parser = ArgumentParser()
    parser.add_argument('path', help='benchmark result json file')
    args = parser.parse_args()

    with open(args.path) as file:
        benchmark_result = json.load(file)
        benchmarks = benchmark_result['benchmarks']
        elapsed_times = groupby(benchmarks, extract_label_from_benchmark)
        data1 = None
        data2 = None

        for key, group in elapsed_times:
            benchmark = list(group)
            x = list(map(extract_size_from_benchmark, benchmark))
            y1 = list(map(operator.itemgetter('bytes_per_second'), benchmark))
            y2 = list(map(operator.itemgetter('items_per_second'), benchmark))

            if data1 is None:
                data1 = pd.DataFrame({'size': x, key: y1})
            else:
                data1[key] = y1
            
            if data2 is None:
                data2 = pd.DataFrame({'size': x, key: y2})
            else:
                data2[key] = y2

        df1 = pd.melt(data1, id_vars=['size'], var_name='algorithm', value_name='bytes_per_second')
        df2 = pd.melt(data2, id_vars=['size'], var_name='algorithm', value_name='items_per_second')
        
        print(df1)
        
        sns.set_theme()

        fig, ax = plt.subplots(2, 1)

        fig.set_size_inches(16, 9)
        fig.set_dpi(160)

        sns.lineplot(data=df1, x='size', y='bytes_per_second', hue='algorithm', ax=ax[0])
        sns.lineplot(data=df2, x='size', y='items_per_second', hue='algorithm', ax=ax[1])

        ax[0].set_title('Bytes per second')
        ax[1].set_title('Items per second')

        ax[0].set_xlabel('Array size')
        ax[1].set_xlabel('Array size')

        ax[0].set_ylabel('Giga byte per second (GB/s)')
        ax[1].set_ylabel('Giga items per second (GI/s)')

        fig.tight_layout()

        plt.show()

        plt.savefig('benchmark.png', bbox_inches='tight', dpi=300)

