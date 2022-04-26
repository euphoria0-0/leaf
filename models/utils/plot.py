import os
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import argparse
import time
import numpy as np
from scipy.interpolate import interp1d, make_interp_spline


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='celeba_s')
    parser.add_argument('--max', type=int, default=500)
    parser.add_argument('--std', action='store_true', default=False, help='show std')
    parser.add_argument('--offset', action='store_true', default=False, help='show random offset')
    parser.add_argument('--smooth', action='store_true', default=False, help='smoothing plot')
    args = parser.parse_args()
    return args


# setting
args = get_args()
data_path = f'./results/{args.data}/'
save_path = f'./results/{args.data}_plots/'
os.makedirs(save_path, exist_ok=True)


# files
columns = {}
df_raw = pd.DataFrame()
df_stats = pd.DataFrame()

# raw data file
files = os.listdir(data_path)[::-1]
for file in files:
    model = file[:-16].replace('S_','')

    if model in columns.keys():
        columns[model] += 1
    else:
        columns[model] = 1
    for f in os.listdir(data_path+file):
        if 'stat.csv' in f:
            data = pd.read_csv(data_path+file+'/'+f, delimiter=',')
            data.sort_values(by='round_number', inplace=True)
            accuracies = data.groupby('round_number', as_index=False).mean()
            if 'round_number' not in df_raw.columns:
                df_raw['round_number'] = accuracies['round_number']
            df_raw[model+str(columns[model])] = accuracies['accuracy']
            #stds = data.groupby('round_number', as_index=False).std()

# statistics file
col = 'Random'
cols = [x for x in df_raw.columns if col in x]
df_stats[col] = df_raw[cols].mean(axis=1)
if args.std:
    df_stats[col+'_std'] = df_raw[cols].std(axis=1)

for col in columns.keys():
    if 'Random' in col: continue
    cols = [x for x in df_raw.columns if col in x]
    if args.offset:
        df_stats[col] = df_raw[cols].mean(axis=1) - df['Random']
    else:
        df_stats[col] = df_raw[cols].mean(axis=1)
    if args.std:
        df_stats[col+'_std'] = df_raw[cols].std(axis=1)
df_stats.index = df_raw['round_number']
df_stats.to_csv(save_path+'df.csv')

print(columns)

# max x axis
df_stats = df_stats[:args.max]

# plotting
sns.set(rc={'figure.figsize': (16,9)}, style='white')
num_cols = len(columns)-1 if args.offset else len(columns)
color_palette_name = 'Spectral' if num_cols > 10 else 'tab10'
colors = sns.color_palette(color_palette_name, num_cols)


fig, ax = plt.subplots()
for i, column in enumerate(columns):
    if args.offset and 'Random' in args.offset:
        continue

    x, y = df_stats.index, df_stats[column]
    #print(x,y)

    if args.smooth:
        model = make_interp_spline(x, y, check_finite=False, k=3)
        x = np.linspace(1, args.max, 50)
        y = model(x)

    ax.plot(x, y, label=column, color=colors[i])
    if args.std:
        ax.fill_between(x, (y - df_stats[column + '_std']), (y + df_stats[column + '_std']), alpha=.2)


#df.plot(figsize=(16,9))
ax.legend()
if args.offset:
    ax.hlines(y=0, xmin=0, xmax=len(df_stats), linewidth=2, color='black')
else:
    ax.set_ylim(0.54,)
ax.set_title(f'{args.data.upper()}', fontsize=25)
ax.set_xlabel('Number of communication rounds', fontsize=15)


std = '_std' if args.std else ''
offset = '_offset' if args.offset else ''
mx = '_'+str(args.max) if args.max != 500 else ''
tm = time.strftime('%Y%m%d-%H%M%S', time.localtime())
fig_name = f'Plot{std}{offset}{mx}_{tm}'.replace('.','_')
print(save_path+fig_name)
plt.savefig(save_path+fig_name, bbox_inches='tight')
