# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: intercluster-py3.12
#     language: python
#     name: intercluster-py3.12
# ---

# %%
import sys
sys.path.insert(0, '..')

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.collections import LineCollection
from matplotlib.ticker import MaxNLocator, FormatStrFormatter, AutoMinorLocator
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import seaborn as sns
from data.preprocessing import *
from intercluster.plotting import *
from intercluster import *
from intercluster.decision_trees import *
from intercluster.decision_sets import *
from intercluster.decision_sets.mining import *
from intercluster.decision_sets.objectives import *

# %load_ext autoreload
# %autoreload 2


# %%
# Which experiment output files this notebook loads (shared by every section below).
EXP_REF = "_conf_00"
DATASETS = ["climate", "anuran", "protein", "yeast", "mnist", "fashion"]

# %%
# This assumes tex is installed in your system,
# if not you may simply remove most of this aside from font.size
# To get tex working on linux run the following:
# `sudo apt-get install texlive-latex-extra texlive-fonts-recommended dvipng cm-super`
plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "font.family": "serif",
    "font.serif": [],
    "text.usetex": True,
    "pgf.rcfonts": False,
    "font.size": 32
})

palette = sns.color_palette("husl", 8)
cmap = ListedColormap(palette)

cmap = ListedColormap(sns.color_palette("tab10", 10))

# Per-model color/hatch, keyed by module name as it appears in the JSON
# (after the objective segment is stripped, for PEC variants). Every model
# also gets a distinct hatch pattern as a second visual channel.
color_dict = {
    'Decision-Tree': cmap(0),
    'ExKMC': cmap(1),
    #'WRA': cmap(6),
    'IDS': cmap(2),
    'CBA': cmap(3),
    'CN2': cmap(4),
    'dscluster; ensemble': cmap(9),
    # PEC's scaled-greedy ablation gets its own hue (kept visually distinct
    # from PEC itself); bicriteria-plot cells additionally apply `_muted` on
    # top to keep it deliberately understated there.
    'dscluster; ensemble; lazy-greedy': cmap(5),
    #'dscluster; ensemble; weighted': cmap(2),
    'Reference': 'black',
}

hatch_dict = {
    'Decision-Tree': '',
    'ExKMC': '',
    'WRA': '',
    'CBA': '',
    'CN2': '',
    'IDS': '',
    'dscluster; ensemble': '//',
    'dscluster; ensemble; weighted': '',
    'dscluster; ensemble; lazy-greedy': '\\',
}

title_dict = {
    'dscluster; ensemble' : r'\texttt{PEC}',
    'dscluster; ensemble; weighted' : r'\texttt{PEC Weighted}',
    'Decision-Tree': r'\texttt{Decision-Tree}',
    'ExKMC': r'\texttt{ExKMC}',
    'WRA': r'\texttt{WRA}',
    'CBA': r'\texttt{CBA}',
    'CN2': r'\texttt{CN2}',
    'IDS': r'\texttt{IDS}',
}

objective_name_dict = {
    'coverage-cost': r'\textit{k-Means}',
    'coverage-mistake': r'\textit{Mistakes}',
    'coverage-pairwise-distance': r'\textit{Pairwise Distance}',

}


# %%
cmap

# %% [markdown]
# # Experiment Plotting:
# Gathers results and produces plots/tables for the paper. Requires the experiments in `experiments/` to have been run first.

# %%
objective_cost_reward_dict = {
    'coverage-mistake': {'reward': 'cluster-coverage', 'cost': 'mistakes'},
    'total-coverage-mistake': {'reward': 'total-coverage', 'cost': 'mistakes'},
    'coverage-cost': {'reward': 'cluster-coverage', 'cost': 'rule-clustering-cost'},
    'total-coverage-cost': {'reward': 'total-coverage', 'cost': 'rule-clustering-cost'},
    'coverage-pairwise-distance': {'reward': 'cluster-coverage', 'cost': 'rule-pairwise-distance'},
    'total-coverage-pairwise-distance': {'reward': 'total-coverage', 'cost': 'rule-pairwise-distance'},
    'coverage-mistake-weighted': {'reward': 'cluster-coverage', 'cost': 'mistakes'},
    'total-coverage-mistake-weighted': {'reward': 'total-coverage', 'cost': 'mistakes'},
    'coverage-cost-weighted': {'reward': 'cluster-coverage', 'cost': 'rule-clustering-cost'},
    'total-coverage-cost-weighted': {'reward': 'total-coverage', 'cost': 'rule-clustering-cost'},
    'coverage-pairwise-distance-weighted': {'reward': 'cluster-coverage', 'cost': 'rule-pairwise-distance'},
    'total-coverage-pairwise-distance-weighted': {'reward': 'total-coverage', 'cost': 'rule-pairwise-distance'},
}


# %%
# Helpers for picking "nice" (multiple of 1/2/5) axis limits with exactly 3 ticks.
def _nice_step(x: float) -> float:
    """Return a 'nice' step size near x using the 1-2-5 rule."""
    if not np.isfinite(x) or x <= 0:
        return 1.0
    exp = np.floor(np.log10(x))
    f = x / (10 ** exp)
    if f <= 1:
        nf = 1
    elif f <= 2:
        nf = 2
    elif f <= 5:
        nf = 5
    else:
        nf = 10
    return nf * (10 ** exp)


def nice_lim_for_3_ticks(raw_min: float, raw_max: float, *, clip=(0.0, 1.0)):
    """Choose ymin/ymax so that 3 evenly spaced ticks are also 'nice'.

    We enforce: ticks = [ymin, ymin+step, ymin+2*step], so both ends and the midpoint
    are multiples of a 'nice' step.
    """
    lo_clip, hi_clip = clip

    raw_min = max(raw_min, lo_clip)
    raw_max = min(raw_max, hi_clip)

    if not (np.isfinite(raw_min) and np.isfinite(raw_max)):
        return lo_clip, hi_clip, np.array([lo_clip, (lo_clip + hi_clip) / 2, hi_clip])

    if raw_max <= raw_min:
        # Degenerate case: build a small nice range around raw_min
        step = _nice_step(abs(raw_min) * 0.1 + 1e-3)
        ymin = np.floor(raw_min / step) * step
        ymax = ymin + 2 * step
    else:
        # With 3 ticks, there are 2 intervals. Pick a 'nice' step near (range / 2)
        step = _nice_step((raw_max - raw_min) / 2)
        ymin = np.floor(raw_min / step) * step
        ymax = np.ceil(raw_max / step) * step

        # Ensure the span fits exactly 2*step so that the midpoint is also nice.
        # Expand outward in step increments as needed.
        span_steps = int(np.ceil((ymax - ymin) / step))
        if span_steps <= 0:
            span_steps = 1
        if span_steps % 2 == 1:
            span_steps += 1  # make it even

        ymax = ymin + span_steps * step

        # If we expanded too much, try shifting up one step (without losing coverage)
        # to stay closer to data while keeping the 2*step structure.
        while (ymin + step) <= raw_min and (ymax + step) <= hi_clip:
            ymin += step
            ymax += step

    ymin = max(ymin, lo_clip)
    ymax = min(ymax, hi_clip)

    # Recompute ticks (exactly 3)
    ticks = np.array([ymin, ymin + (ymax - ymin) / 2, ymax])
    return ymin, ymax, ticks

def _scalar(v):
    """Stochastic modules (Decision-Tree, IDS) store per-r values as
    {'mean', 'std', 'values'}; deterministic modules store a bare float/int.
    Normalize to a scalar (the mean); None (JSON-encoded NaN) -> NaN.
    """
    if isinstance(v, dict):
        return v.get('mean', np.nan)
    return v if v is not None else np.nan


def _values(v):
    """Raw per-trial values dict for stochastic modules; None otherwise."""
    if isinstance(v, dict):
        return v.get('values', None)
    return None


def _std(v):
    """Raw std across trial repeats (Decision-Tree/IDS only); NaN otherwise.
    Used only in Input Sensitivity, where the spread itself (not the average)
    is the quantity being measured -- see `_se` for everywhere else.
    """
    if isinstance(v, dict):
        return v.get('std', np.nan)
    return np.nan


def _se(v):
    """Standard error of the mean (std / sqrt(n_trials)); NaN for deterministic
    modules. Used everywhere a plotted band supports a comparison between
    models' average performance (Max Rules, Bicriteria, Confidence) -- raw std
    would overstate noise there. See `_std` for the one section (Input
    Sensitivity) where spread itself, not SE, is the right statistic.
    """
    if isinstance(v, dict):
        values = v.get('values', None)
        std = v.get('std', np.nan)
        if not values:
            return np.nan
        return std / np.sqrt(len(values))
    return np.nan


# Floor for on-screen error-bar visibility (same [0,1]-ish units as x/y/z),
# tuned to exceed the marker's own radius. Keep in sync with the other
# bicriteria plot cell.
MIN_VISIBLE_ERR = 0.025

def _visible_err(err):
    """Floor nonzero error up to MIN_VISIBLE_ERR; exact zeros stay 0."""
    err = np.asarray(err, dtype=float)
    return np.where(err > 0, np.maximum(err, MIN_VISIBLE_ERR), 0.0)


def _muted(color, amount=1/3):
    """Blend `color` toward white by `amount` (0 = color, 1 = white); used to
    keep PEC's scaled-greedy ablation visually understated vs. PEC itself."""
    r, g, b = mcolors.to_rgb(color)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)


# %% [markdown]
# ### Alphas
#
# Plot and select values from the `alphas.py` experiment.
#

# %%
# Load experiment data
# NOTE: This assumes you have computed the experiment for all datasets and objectives.
# If any are missing, simply comment them out from the DATASETS and objective_names lists.

objective_names = ['coverage-cost', 'coverage-mistake', 'coverage-pairwise-distance']

dataset_alpha_experiment_dict = {}
for dataset in DATASETS:
    fname = "../data/experiments/" + dataset + "/alphas/exp" + EXP_REF + ".json"
    with open(fname, 'r') as f:
        experiment_dict = json.load(f)
    dataset_alpha_experiment_dict[dataset] = experiment_dict

# Collect modules from the last experiment dict
dscluster_modules = list(experiment_dict['modules'].keys())

# %%
# Collect experiment data:

alpha_objective_dict = {
    dataset: {objective: {} for objective in objective_names} for dataset in dataset_alpha_experiment_dict.keys()
}
for dataset, experiment_dict in dataset_alpha_experiment_dict.items():
    fixed_parameters = experiment_dict['fixed-parameters']
    for objective in objective_names:
        selected_dscluster_module = [
            m for m in dscluster_modules if objective == m.split(';')[1].strip()
        ][0] # there should only be one per objective!!
        reward = objective_cost_reward_dict[objective]['reward']
        cost = objective_cost_reward_dict[objective]['cost']

        alpha_vals = np.array(list(experiment_dict['modules'][selected_dscluster_module][cost].keys()), dtype=float)
        z = alpha_vals
        x = np.array([experiment_dict['modules'][selected_dscluster_module]['weighted-avg-length'][str(l)] for l in z])

        rl = np.array([experiment_dict['modules'][selected_dscluster_module]['sum-rule-length'][str(l)] for l in z])
        y1 = np.array([experiment_dict['modules'][selected_dscluster_module][reward][str(l)] for l in z])
        y2 = np.array([experiment_dict['modules'][selected_dscluster_module][cost][str(l)] for l in z]) + z * rl
        lambda_vals = np.array([experiment_dict['modules'][selected_dscluster_module]['lambda'][str(l)] for l in z])
        y = y1 - lambda_vals * y2

        # Compute best alpha using elbow method
        best_alpha_idx = compute_elbow(x, y, increasing = True)
        
        # Check to see if there are any larger values after best_alpha_idx
        for idx in range(best_alpha_idx + 1, len(z)):
            if y[idx] >= y[best_alpha_idx] * 0.999:  # within 1% of best value
                best_alpha_idx = idx
        
        best_alpha = z[best_alpha_idx]

        alpha_objective_dict[dataset][objective] = {
            'alpha_vals': z,
            'x': x,
            'y': y / fixed_parameters['n'],
            'selected_alpha_idx': best_alpha_idx,
            'selected_alpha': best_alpha,
        }

# %%
# Plot results:

fig, axs = plt.subplots(len(objective_names), len(dataset_alpha_experiment_dict), figsize=(34, 14))

function_name_dict = {0: 'K', 1: 'M', 2: 'P'}

for i, (dataset, objective_result_dict) in enumerate(alpha_objective_dict.items()):
    for j, (objective, module_result_dict) in enumerate(objective_result_dict.items()):
        # Gridlines:
        axs[j,i].grid(which='major', linestyle='-', linewidth=0.8, alpha = 0.5)
        axs[j,i].grid(which='minor', linestyle=':', linewidth=0.8, alpha = 0.5)
        axs[j,i].minorticks_on()

        # Plot the main line:
        x = module_result_dict['x']
        y = module_result_dict['y']
        z = module_result_dict['alpha_vals']
        selected_idx = module_result_dict['selected_alpha_idx']
        selected_alpha = module_result_dict['selected_alpha']

        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm = plt.Normalize(z.min(), z.max()) # Normalize color map
        lc = LineCollection(segments, cmap='coolwarm', norm=norm)
        lc.set_array(z)
        lc.set_linewidth(6)
        line = axs[j,i].add_collection(lc)

        # Plot selected point:
        axs[j,i].scatter(
            x[selected_idx],
            y[selected_idx],
            color='black',
            marker='o',
            s=200,
            zorder=5,
            label=rf'Best $\alpha$ = {selected_alpha:.2g}'
        )

        # Set x and y ticks to make the plot look nice
        x_min = np.min(x)
        x_max = np.max(x)
        x_std = np.std(x)
        y_min = np.min(y)
        y_max = np.max(y)
        y_std = np.std(y)

        x_lo, x_hi, xticks = nice_lim_for_3_ticks(x_min - 0.001 * x_std, x_max + 0.001 * x_std, clip=(1.0, 6.0))
        y_lo, y_hi, yticks = nice_lim_for_3_ticks(y_min - 0.001 * y_std, y_max + 0.001 * y_std)

        axs[j,i].set_xlim(x_lo, x_hi)
        axs[j,i].set_xticks(xticks)
        axs[j,i].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        axs[j,i].set_ylim(y_lo, y_hi)
        axs[j,i].set_yticks(yticks)
        axs[j,i].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        # Increase tick-label padding away from the axes to reduce overlap at origin
        axs[j,i].tick_params(axis='x', which='major', pad=10)
        axs[j,i].tick_params(axis='y', which='major', pad=10)

        # Individual colorbar per subplot
        cbar = fig.colorbar(line, ax=axs[j,i], fraction=0.046, pad=0.02)

        # Make color bar ticks compact
        cbar.locator = MaxNLocator(nbins=4)
        cbar.formatter.set_powerlimits((0, 0))  # always scientific (10^k)
        cbar.update_ticks()
        cbar.ax.yaxis.get_offset_text().set(size=18)

        # Y-label with objective
        if i == 0:
            axs[j,i].set_ylabel(
                rf"$\bar{{f}}_{function_name_dict[j]}$", rotation=0, labelpad=40, fontsize = 36
            )

        # Title with dataset name
        if j == 0:
            if dataset == "kddcup":
                axs[j,i].set_title(rf"$KDDCup$")
            else:
                axs[j,i].set_title(rf"${dataset.capitalize()}$")


fig.supxlabel(r"$\textup{weighted-avg-length}$", y=0.05, x = 0.525)
plt.tight_layout()

plt.savefig(
    "../figures/experiments/alphas.pdf",
    bbox_inches='tight',
    dpi=300
)

# %% [markdown]
# # Max Rules Experiment
# The following plots results from the `max_rules` experiments.

# %%
# Load experiment data
# NOTE: Loading is defensive -- datasets whose max_rules.py output hasn't been
# generated yet (data/experiments/<dataset>/max_rules/exp<ref>.json missing)
# are skipped with a warning rather than raising.

objective_names = ['coverage-cost', 'coverage-mistake', 'coverage-pairwise-distance']

# Explicit comparison set -- objective-agnostic, unlike PEC's own modules below.
comparison_modules = {'Decision-Tree', 'ExKMC', 'IDS', 'CBA', 'CN2'}

# PEC's distorted-greedy (main model) and scaled-greedy (selection-algorithm
# ablation, keyed by max_rules.py's "; lazy-greedy" suffix) modules are kept
# separate from comparison_modules: each is objective-specific, so bar-plot
# cells must pick the one matching the current objective rather than treat
# them as a fixed set (merging scaled-greedy in previously leaked a dataset's
# coverage-cost-tuned module into the other objectives' panels).
dscluster_modules = set()
scaled_dscluster_modules = set()
dataset_experiment_dict = {}
for dataset in DATASETS:
    fname = "../data/experiments/" + dataset + "/max_rules/exp" + EXP_REF + ".json"
    if not os.path.exists(fname):
        print(f"[max_rules] skipping {dataset}: {fname} not found")
        continue
    with open(fname, 'r') as f:
        experiment_dict = json.load(f)
    dataset_experiment_dict[dataset] = experiment_dict
    dscluster_modules.update(
        [m for m in experiment_dict['modules'].keys() if ('dscluster' in m) and ('lazy-greedy' not in m)]
    )
    scaled_dscluster_modules.update(
        [m for m in experiment_dict['modules'].keys() if ('dscluster' in m) and ('lazy-greedy' in m) and ('weighted' not in m)]
    )
    missing = [m for m in comparison_modules if m not in experiment_dict['modules']]
    if missing:
        print(f"[max_rules] warning: {dataset} is missing modules {missing}")

dscluster_modules = list(dscluster_modules)
scaled_dscluster_modules = list(scaled_dscluster_modules)

# PEC scaled-greedy comes from a separate script (max_rules_pec_lazy.py, merged
# via max_rules_combine.py), so a dataset may legitimately be missing it for
# some objective -- flag rather than silently drop the comparison bar later.
for dataset, experiment_dict in dataset_experiment_dict.items():
    missing_scaled = [
        objective for objective in objective_names
        if not any(
            objective == m.split(';')[1].strip() and m in experiment_dict['modules']
            for m in scaled_dscluster_modules
        )
    ]
    if missing_scaled:
        print(f"[max_rules] note: {dataset} has no PEC scaled-greedy results for objective(s) {missing_scaled}")

# %%
lambda_val_dict = {}
for dataset, experiment_dict in dataset_experiment_dict.items():
    if dataset not in lambda_val_dict:
        lambda_val_dict[dataset] = {}
    for module in experiment_dict['modules'].keys():
        if 'dscluster' not in module:
            continue

        lambda_val_dict[dataset][module] = list(experiment_dict['modules'][module]['lambda'].values())[0]

# %%
# Lambda values for each dataset and objective:
pd.DataFrame(lambda_val_dict) 

# %%
alpha_val_dict = {}
for dataset, experiment_dict in dataset_experiment_dict.items():
    alpha_val_dict[dataset] = experiment_dict['fixed-parameters']['alpha']

# %%
# Alpha values for each dataset and objective:
pd.DataFrame(alpha_val_dict)


# %% [markdown]
# ### Bar Plots

# %%
# Collect experiment data for bar plots. obj_err is ±1 SE; see `_se`.

def _module_bar_series(module_dict, objective_name, x, reward, cost, lambd, alpha):
    """Returns (obj_values, obj_err) arrays over r (in `x` order).

    Prefers the module's stored true objective score, `module_dict['objective'][objective_name]`
    (written by max_rules.py) since it's the actual g - lambda*h being optimized. Falls back to
    reconstructing `reward - lambda*(cost + alpha*length)` only for cached exp*.json files
    predating that field.
    """
    stored = module_dict.get('objective', {}).get(objective_name)
    if stored is not None:
        obj_values = np.array([_scalar(stored[r]) for r in x])
        obj_err = np.nan_to_num(np.array([_se(stored[r]) for r in x]), nan=0.0)
        return obj_values, obj_err

    reward_vals = [module_dict[reward][r] for r in x]
    cost_vals = [module_dict[cost][r] for r in x]
    length_vals = [module_dict['sum-rule-length'][r] for r in x]
    obj_values = np.array([
        _scalar(rv) - lambd * (_scalar(cv) + alpha * _scalar(lv))
        for rv, cv, lv in zip(reward_vals, cost_vals, length_vals)
    ])
    obj_err = np.zeros_like(obj_values)
    for idx, (rv, cv, lv) in enumerate(zip(reward_vals, cost_vals, length_vals)):
        rv_t, cv_t, lv_t = _values(rv), _values(cv), _values(lv)
        if rv_t is not None and cv_t is not None and lv_t is not None:
            combined_t = np.array(rv_t) - lambd * (np.array(cv_t) + alpha * np.array(lv_t))
            obj_err[idx] = np.std(combined_t) / np.sqrt(len(combined_t))
    return obj_values, obj_err


bar_dict = {
    dataset: {objective: {} for objective in objective_names} for dataset in dataset_experiment_dict.keys()
}
for dataset, experiment_dict in dataset_experiment_dict.items():
    fixed_parameters = experiment_dict['fixed-parameters']
    for objective in objective_names:
        selected_dscluster_module = [
            m for m in dscluster_modules if objective == m.split(';')[1].strip()
        ][0] # there should only be one per objective!!
        reward = objective_cost_reward_dict[objective]['reward']
        cost = objective_cost_reward_dict[objective]['cost']
        alpha = fixed_parameters['alpha'][selected_dscluster_module]
        lambd = max(list(experiment_dict['modules'][selected_dscluster_module]['lambda'].values()))

        x = np.array(list(experiment_dict['modules'][selected_dscluster_module][reward].keys()))
        idxs = np.where(x.astype(int) <= min(np.max(x.astype(int)), np.min(x.astype(int) + 6)))[0][::2]

        # Compute objective values:
        for cmod in comparison_modules:
            if cmod not in experiment_dict['modules']:
                continue
            obj_values, obj_err = _module_bar_series(
                experiment_dict['modules'][cmod], objective, x, reward, cost, lambd, alpha
            )
            bar_dict[dataset][objective][cmod] = (
                x[idxs], obj_values[idxs] / fixed_parameters['n'], obj_err[idxs] / fixed_parameters['n']
            )

        # DSCluster Module (distorted-greedy, the paper's main model):
        obj_values, obj_err = _module_bar_series(
            experiment_dict['modules'][selected_dscluster_module], objective, x, reward, cost, lambd, alpha
        )
        bar_dict[dataset][objective][selected_dscluster_module] = (
            x[idxs], obj_values[idxs] / fixed_parameters['n'], obj_err[idxs] / fixed_parameters['n']
        )

        # Scaled-greedy counterpart (optional, see missing_scaled check above).
        # Scored at the SAME alpha/lambda as distorted-greedy above (it was fit
        # at that identical lambda_star), for an apples-to-apples comparison.
        scaled_matches = [
            m for m in scaled_dscluster_modules
            if objective == m.split(';')[1].strip() and m in experiment_dict['modules']
        ]
        if scaled_matches:
            selected_scaled_module = scaled_matches[0] # there should only be one per objective!!
            obj_values, obj_err = _module_bar_series(
                experiment_dict['modules'][selected_scaled_module], objective, x, reward, cost, lambd, alpha
            )
            bar_dict[dataset][objective][selected_scaled_module] = (
                x[idxs], obj_values[idxs] / fixed_parameters['n'], obj_err[idxs] / fixed_parameters['n']
            )

# %%
# Plot results. Error bars: ±1 SE (see `_se`); none for deterministic modules.

# Lower bound for the objective-value y-axis, so subplots don't waste space.
BAR_Y_MIN = 0.0

fig,ax = plt.subplots(len(objective_names), len(dataset_experiment_dict), figsize=(36, 12))

module_order = [
    'Decision-Tree',
    'ExKMC',
    #'WRA',
    'IDS',
    'CBA',
    'CN2',
    'dscluster; ensemble; lazy-greedy',
    'dscluster; ensemble',
]

function_name_dict = {0: 'K', 1: 'M', 2: 'P'}

for i, (dataset, objective_result_dict) in enumerate(bar_dict.items()):
    for j, (objective, module_result_dict) in enumerate(objective_result_dict.items()):
        # plot bars
        # PEC scaled-greedy is always placed immediately to the left of
        # standard (distorted-greedy) PEC, both here and in the legend below.
        module_order[-2] = f'dscluster; {objective}; ensemble; lazy-greedy'
        module_order[-1] = f'dscluster; {objective}; ensemble'

        obj_min = np.inf
        obj_max = -np.inf
        for k, module in enumerate(module_order):
            if module not in module_result_dict:
                continue
            x, obj_values, obj_err = module_result_dict[module]
            if 'dscluster' in module:
                # Strip the objective segment so 4-part scaled-greedy names
                # resolve to the same style-dict key as everywhere else.
                mod_name = module.split(';')[0].strip() + "; " + "; ".join(
                    part.strip() for part in module.split(';')[2:]
                )
            else:
                mod_name = module
            hatch = hatch_dict.get(mod_name, '')

            # Plot bar for a single module:
            width = 0.225
            ax[j,i].bar(
                x.astype(int) + k * width,
                obj_values,
                yerr=obj_err,
                width=width,
                label=mod_name,
                color=_muted(color_dict.get(mod_name, 'grey')),
                alpha = 1.0,
                hatch=hatch,
                edgecolor='black',
                capsize=3,
                error_kw={'elinewidth': 1.2, 'ecolor': 'black'},
            )

            if np.min(obj_values) < obj_min:
                obj_min = np.min(obj_values)
            if np.max(obj_values) > obj_max:
                obj_max = np.max(obj_values)

        # Set y limits and ticks
        obj_rng = obj_max - obj_min
        pad = 0.01 * obj_rng
        y_lo, y_hi, yticks = nice_lim_for_3_ticks(obj_min - pad, obj_max + pad, clip=(BAR_Y_MIN, 1.0))
        ax[j,i].set_ylim(y_lo, y_hi)
        ax[j,i].set_yticks(yticks)
        ax[j,i].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        # Set x ticks and labels
        ax[j,i].set_xticks(x.astype(int) + (len(module_result_dict)-1) * 0.2 / 2)
        if j != len(objective_names) - 1:
            ax[j,i].set_xticklabels([])
        else:
            ax[j,i].set_xticklabels([str(int(val)) for val in x.astype(int)])

        # Y-label with objective
        if i == 0:
            ax[j,i].set_ylabel(
                rf"$\bar{{f}}_{function_name_dict[j]}$", rotation=0, labelpad=40, fontsize = 36
            )

        # Title with dataset name
        if j == 0:
            ax[j,i].set_title(rf"${dataset.capitalize()}$")

        # Gridlines:
        ax[j,i].grid(which='major', linestyle='-', linewidth=0.8, alpha = 0.5)
        ax[j,i].axhline(0.0, color="black", linewidth=2.0, alpha=0.7)


fig.supxlabel(r"Number of Rules, $\ell$", y=0.05, x = 0.525)
plt.tight_layout()

plt.savefig(
    "../figures/experiments/objectives.pdf",
    bbox_inches='tight',
    dpi=300
)

# %%
# Create a separate legend for the bar plot with hatch patterns
fig, ax = plt.subplots(figsize=(12, 2))

dataset = list(dataset_experiment_dict.keys())[0]
objective = objective_names[0]
module_order = [
    'Decision-Tree',
    'ExKMC',
    #'WRA',
    'IDS',
    'CBA',
    'CN2',
    'dscluster; ensemble; lazy-greedy',
    'dscluster; ensemble',
]

# Local override -- 'dscluster; ensemble; lazy-greedy' isn't in the global
# title_dict, same pattern as the Bicriteria-plot legend cell above.
legend_labels = dict(title_dict) | {
    'dscluster; ensemble; lazy-greedy': r'\texttt{ScaledGreedy}',
}

legend_elements = []
for mod in module_order:
    # module_order entries are already short-form keys, no stripping needed
    # (unlike the plotting cell above, which reads real JSON module names).
    mod_name = mod

    legend_elements.append(
        mpatches.Patch(
            facecolor=_muted(color_dict[mod_name]),
            alpha = 1.0,
            edgecolor='black',
            linewidth=1.5,
            hatch=hatch_dict.get(mod_name, ''),
            label = legend_labels.get(mod_name, mod_name)
        )
    )

ax.legend(handles=legend_elements, ncol=7, loc='center', frameon=False,
          handlelength=2, handleheight=2)
ax.axis('off')

plt.savefig(
    "../figures/experiments/objectives_legend.pdf",
    bbox_inches='tight',
    dpi=300
)

plt.show()

# %% [markdown]
# ### Bar Plots (IDS Alt)
#
# Same layout as the Max Rules bar plots, but IDS is tuned to directly maximize the PEC objective (via `ids_lambda_search_alt.py`) instead of held-out AUC. Only `coverage-cost` has been run so far.

# %%
# Load experiment data for the max_rules_ids_alt.py / max_rules_combine_ids_alt.py comparison.
# NOTE: Loading is defensive, same as the main max_rules loading cell above.

ids_alt_objective_names = objective_names[:1]  # only coverage-cost (k-means) has been run

dscluster_modules_ids_alt = set()
scaled_dscluster_modules_ids_alt = set()
dataset_experiment_dict_ids_alt = {}
for dataset in DATASETS:
    fname = "../data/experiments/" + dataset + "/max_rules/exp" + EXP_REF + "_ids_alt.json"
    if not os.path.exists(fname):
        print(f"[max_rules_ids_alt] skipping {dataset}: {fname} not found")
        continue
    with open(fname, 'r') as f:
        experiment_dict = json.load(f)
    dataset_experiment_dict_ids_alt[dataset] = experiment_dict
    dscluster_modules_ids_alt.update(
        [m for m in experiment_dict['modules'].keys() if ('dscluster' in m) and ('lazy-greedy' not in m)]
    )
    scaled_dscluster_modules_ids_alt.update(
        [m for m in experiment_dict['modules'].keys() if ('dscluster' in m) and ('lazy-greedy' in m) and ('weighted' not in m)]
    )

dscluster_modules_ids_alt = list(dscluster_modules_ids_alt)
scaled_dscluster_modules_ids_alt = list(scaled_dscluster_modules_ids_alt)

# %%
# Collect experiment data for the IDS-alt bar plot -- identical logic to the main Max Rules
# bar_dict collection cell above, restricted to ids_alt_objective_names.

bar_dict_ids_alt = {
    dataset: {objective: {} for objective in ids_alt_objective_names} for dataset in dataset_experiment_dict_ids_alt.keys()
}
for dataset, experiment_dict in dataset_experiment_dict_ids_alt.items():
    fixed_parameters = experiment_dict['fixed-parameters']
    for objective in ids_alt_objective_names:
        selected_dscluster_module = [
            m for m in dscluster_modules_ids_alt if objective == m.split(';')[1].strip()
        ][0]  # there should only be one per objective!!
        reward = objective_cost_reward_dict[objective]['reward']
        cost = objective_cost_reward_dict[objective]['cost']
        alpha = fixed_parameters['alpha'][selected_dscluster_module]
        lambd = max(list(experiment_dict['modules'][selected_dscluster_module]['lambda'].values()))

        x = np.array(list(experiment_dict['modules'][selected_dscluster_module][reward].keys()))
        idxs = np.where(x.astype(int) <= min(np.max(x.astype(int)), np.min(x.astype(int) + 6)))[0][::2]

        for cmod in comparison_modules:
            if cmod not in experiment_dict['modules']:
                continue
            obj_values, obj_err = _module_bar_series(
                experiment_dict['modules'][cmod], objective, x, reward, cost, lambd, alpha
            )
            bar_dict_ids_alt[dataset][objective][cmod] = (
                x[idxs], obj_values[idxs] / fixed_parameters['n'], obj_err[idxs] / fixed_parameters['n']
            )

        obj_values, obj_err = _module_bar_series(
            experiment_dict['modules'][selected_dscluster_module], objective, x, reward, cost, lambd, alpha
        )
        bar_dict_ids_alt[dataset][objective][selected_dscluster_module] = (
            x[idxs], obj_values[idxs] / fixed_parameters['n'], obj_err[idxs] / fixed_parameters['n']
        )

        scaled_matches = [
            m for m in scaled_dscluster_modules_ids_alt
            if objective == m.split(';')[1].strip() and m in experiment_dict['modules']
        ]
        if scaled_matches:
            selected_scaled_module = scaled_matches[0]  # there should only be one per objective!!
            obj_values, obj_err = _module_bar_series(
                experiment_dict['modules'][selected_scaled_module], objective, x, reward, cost, lambd, alpha
            )
            bar_dict_ids_alt[dataset][objective][selected_scaled_module] = (
                x[idxs], obj_values[idxs] / fixed_parameters['n'], obj_err[idxs] / fixed_parameters['n']
            )

# %%
# Plot results -- identical styling to the main Max Rules bar plot above, restricted to the
# single coverage-cost (k-means) row this experiment has data for. Figure height is scaled
# down proportionally (1 row instead of 3) to keep the same per-row size as the main plot.

fig, ax = plt.subplots(
    len(ids_alt_objective_names), len(dataset_experiment_dict_ids_alt),
    figsize=(36, 6), squeeze=False,
)

module_order = [
    'Decision-Tree',
    'ExKMC',
    #'WRA',
    'IDS',
    'CBA',
    'CN2',
    'dscluster; ensemble; lazy-greedy',
    'dscluster; ensemble',
]

function_name_dict = {0: 'K', 1: 'M', 2: 'P'}

for i, (dataset, objective_result_dict) in enumerate(bar_dict_ids_alt.items()):
    for j, (objective, module_result_dict) in enumerate(objective_result_dict.items()):
        # plot bars
        module_order[-2] = f'dscluster; {objective}; ensemble; lazy-greedy'
        module_order[-1] = f'dscluster; {objective}; ensemble'

        obj_min = np.inf
        obj_max = -np.inf
        for k, module in enumerate(module_order):
            if module not in module_result_dict:
                continue
            x, obj_values, obj_err = module_result_dict[module]
            if 'dscluster' in module:
                mod_name = module.split(';')[0].strip() + "; " + "; ".join(
                    part.strip() for part in module.split(';')[2:]
                )
            else:
                mod_name = module
            hatch = hatch_dict.get(mod_name, '')

            # Plot bar for a single module:
            width = 0.225
            ax[j,i].bar(
                x.astype(int) + k * width,
                obj_values,
                yerr=obj_err,
                width=width,
                label=mod_name,
                color=_muted(color_dict.get(mod_name, 'grey')),
                alpha = 1.0,
                hatch=hatch,
                edgecolor='black',
                capsize=3,
                error_kw={'elinewidth': 1.2, 'ecolor': 'black'},
            )

            if np.min(obj_values) < obj_min:
                obj_min = np.min(obj_values)
            if np.max(obj_values) > obj_max:
                obj_max = np.max(obj_values)

        # Set y limits and ticks
        obj_rng = obj_max - obj_min
        pad = 0.01 * obj_rng
        y_lo, y_hi, yticks = nice_lim_for_3_ticks(obj_min - pad, obj_max + pad, clip=(BAR_Y_MIN, 1.0))
        ax[j,i].set_ylim(y_lo, y_hi)
        ax[j,i].set_yticks(yticks)
        ax[j,i].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        # Set x ticks and labels
        ax[j,i].set_xticks(x.astype(int) + (len(module_result_dict)-1) * 0.2 / 2)
        if j != len(ids_alt_objective_names) - 1:
            ax[j,i].set_xticklabels([])
        else:
            ax[j,i].set_xticklabels([str(int(val)) for val in x.astype(int)])

        # Y-label with objective
        if i == 0:
            ax[j,i].set_ylabel(
                rf"$\bar{{f}}_{function_name_dict[j]}$", rotation=0, labelpad=40, fontsize = 36
            )

        # Title with dataset name
        if j == 0:
            ax[j,i].set_title(rf"${dataset.capitalize()}$")

        # Gridlines:
        ax[j,i].grid(which='major', linestyle='-', linewidth=0.8, alpha = 0.5)
        ax[j,i].axhline(0.0, color="black", linewidth=2.0, alpha=0.7)


fig.supxlabel(r"Number of Rules, $\ell$", y=0.05, x = 0.525)
plt.tight_layout()

plt.savefig(
    "../figures/experiments/objectives_ids_alt.pdf",
    bbox_inches='tight',
    dpi=300
)

# %% [markdown]
# ### Bar Plots (Alpha Zero)
#
# Same as the Max Rules bar plots, but PEC is fit with `alpha_val=0` (no rule-length penalty) instead of the elbow-selected alpha. Only PEC's bars differ; comparison models are unaffected. Shares the main Bar Plots legend above.

# %%
# Load experiment data (PEC fit with alpha=0). Loading is defensive, as above.

dscluster_modules_alpha_zero = set()
scaled_dscluster_modules_alpha_zero = set()
dataset_experiment_dict_alpha_zero = {}
for dataset in DATASETS:
    fname = "../data/experiments/" + dataset + "/max_rules/exp" + EXP_REF + "_alpha_zero.json"
    if not os.path.exists(fname):
        print(f"[max_rules_alpha_zero] skipping {dataset}: {fname} not found")
        continue
    with open(fname, 'r') as f:
        experiment_dict = json.load(f)
    dataset_experiment_dict_alpha_zero[dataset] = experiment_dict
    dscluster_modules_alpha_zero.update(
        [m for m in experiment_dict['modules'].keys() if ('dscluster' in m) and ('lazy-greedy' not in m)]
    )
    scaled_dscluster_modules_alpha_zero.update(
        [m for m in experiment_dict['modules'].keys() if ('dscluster' in m) and ('lazy-greedy' in m) and ('weighted' not in m)]
    )
    missing = [m for m in comparison_modules if m not in experiment_dict['modules']]
    if missing:
        print(f"[max_rules_alpha_zero] warning: {dataset} is missing modules {missing}")

dscluster_modules_alpha_zero = list(dscluster_modules_alpha_zero)
scaled_dscluster_modules_alpha_zero = list(scaled_dscluster_modules_alpha_zero)

for dataset, experiment_dict in dataset_experiment_dict_alpha_zero.items():
    missing_scaled = [
        objective for objective in objective_names
        if not any(
            objective == m.split(';')[1].strip() and m in experiment_dict['modules']
            for m in scaled_dscluster_modules_alpha_zero
        )
    ]
    if missing_scaled:
        print(f"[max_rules_alpha_zero] note: {dataset} has no PEC scaled-greedy results for objective(s) {missing_scaled}")

# %%
# Collect experiment data for the alpha-zero bar plot -- identical logic to the main Max Rules
# bar_dict collection cell above, sourced from dataset_experiment_dict_alpha_zero.

bar_dict_alpha_zero = {
    dataset: {objective: {} for objective in objective_names} for dataset in dataset_experiment_dict_alpha_zero.keys()
}
for dataset, experiment_dict in dataset_experiment_dict_alpha_zero.items():
    fixed_parameters = experiment_dict['fixed-parameters']
    for objective in objective_names:
        selected_dscluster_module = [
            m for m in dscluster_modules_alpha_zero if objective == m.split(';')[1].strip()
        ][0]  # there should only be one per objective!!
        reward = objective_cost_reward_dict[objective]['reward']
        cost = objective_cost_reward_dict[objective]['cost']
        alpha = fixed_parameters['alpha'][selected_dscluster_module]
        lambd = max(list(experiment_dict['modules'][selected_dscluster_module]['lambda'].values()))

        x = np.array(list(experiment_dict['modules'][selected_dscluster_module][reward].keys()))
        idxs = np.where(x.astype(int) <= min(np.max(x.astype(int)), np.min(x.astype(int) + 6)))[0][::2]

        for cmod in comparison_modules:
            if cmod not in experiment_dict['modules']:
                continue
            obj_values, obj_err = _module_bar_series(
                experiment_dict['modules'][cmod], objective, x, reward, cost, lambd, alpha
            )
            bar_dict_alpha_zero[dataset][objective][cmod] = (
                x[idxs], obj_values[idxs] / fixed_parameters['n'], obj_err[idxs] / fixed_parameters['n']
            )

        obj_values, obj_err = _module_bar_series(
            experiment_dict['modules'][selected_dscluster_module], objective, x, reward, cost, lambd, alpha
        )
        bar_dict_alpha_zero[dataset][objective][selected_dscluster_module] = (
            x[idxs], obj_values[idxs] / fixed_parameters['n'], obj_err[idxs] / fixed_parameters['n']
        )

        scaled_matches = [
            m for m in scaled_dscluster_modules_alpha_zero
            if objective == m.split(';')[1].strip() and m in experiment_dict['modules']
        ]
        if scaled_matches:
            selected_scaled_module = scaled_matches[0]  # there should only be one per objective!!
            obj_values, obj_err = _module_bar_series(
                experiment_dict['modules'][selected_scaled_module], objective, x, reward, cost, lambd, alpha
            )
            bar_dict_alpha_zero[dataset][objective][selected_scaled_module] = (
                x[idxs], obj_values[idxs] / fixed_parameters['n'], obj_err[idxs] / fixed_parameters['n']
            )

# %%
# Plot results -- identical styling to the main Max Rules bar plot above (all three objectives),
# sourced from the alpha=0 PEC fits instead of the elbow-selected-alpha fits.

fig,ax = plt.subplots(len(objective_names), len(dataset_experiment_dict_alpha_zero), figsize=(36, 12))

module_order = [
    'Decision-Tree',
    'ExKMC',
    #'WRA',
    'IDS',
    'CBA',
    'CN2',
    'dscluster; ensemble; lazy-greedy',
    'dscluster; ensemble',
]

function_name_dict = {0: 'K', 1: 'M', 2: 'P'}

for i, (dataset, objective_result_dict) in enumerate(bar_dict_alpha_zero.items()):
    for j, (objective, module_result_dict) in enumerate(objective_result_dict.items()):
        # plot bars
        module_order[-2] = f'dscluster; {objective}; ensemble; lazy-greedy'
        module_order[-1] = f'dscluster; {objective}; ensemble'

        obj_min = np.inf
        obj_max = -np.inf
        for k, module in enumerate(module_order):
            if module not in module_result_dict:
                continue
            x, obj_values, obj_err = module_result_dict[module]
            if 'dscluster' in module:
                mod_name = module.split(';')[0].strip() + "; " + "; ".join(
                    part.strip() for part in module.split(';')[2:]
                )
            else:
                mod_name = module
            hatch = hatch_dict.get(mod_name, '')

            # Plot bar for a single module:
            width = 0.225
            ax[j,i].bar(
                x.astype(int) + k * width,
                obj_values,
                yerr=obj_err,
                width=width,
                label=mod_name,
                color=_muted(color_dict.get(mod_name, 'grey')),
                alpha = 1.0,
                hatch=hatch,
                edgecolor='black',
                capsize=3,
                error_kw={'elinewidth': 1.2, 'ecolor': 'black'},
            )

            if np.min(obj_values) < obj_min:
                obj_min = np.min(obj_values)
            if np.max(obj_values) > obj_max:
                obj_max = np.max(obj_values)

        # Set y limits and ticks
        obj_rng = obj_max - obj_min
        pad = 0.01 * obj_rng
        y_lo, y_hi, yticks = nice_lim_for_3_ticks(obj_min - pad, obj_max + pad, clip=(BAR_Y_MIN, 1.0))
        ax[j,i].set_ylim(y_lo, y_hi)
        ax[j,i].set_yticks(yticks)
        ax[j,i].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        # Set x ticks and labels
        ax[j,i].set_xticks(x.astype(int) + (len(module_result_dict)-1) * 0.2 / 2)
        if j != len(objective_names) - 1:
            ax[j,i].set_xticklabels([])
        else:
            ax[j,i].set_xticklabels([str(int(val)) for val in x.astype(int)])

        # Y-label with objective
        if i == 0:
            ax[j,i].set_ylabel(
                rf"$\bar{{f}}_{function_name_dict[j]}$", rotation=0, labelpad=40, fontsize = 36
            )

        # Title with dataset name
        if j == 0:
            ax[j,i].set_title(rf"${dataset.capitalize()}$")

        # Gridlines:
        ax[j,i].grid(which='major', linestyle='-', linewidth=0.8, alpha = 0.5)
        ax[j,i].axhline(0.0, color="black", linewidth=2.0, alpha=0.7)


fig.supxlabel(r"Number of Rules, $\ell$", y=0.05, x = 0.525)
plt.tight_layout()

plt.savefig(
    "../figures/experiments/objectives_alpha_zero.pdf",
    bbox_inches='tight',
    dpi=300
)

# %% [markdown]
# ### Rule Provenance
#
# Breakdown of PEC's selected rules by mining source (Decision-Tree, Random-Forest, CAR) at the smallest rule budget (`r = k`). One row per objective, one column per dataset.

# %%
cmap

# %%
source_order = ['decision_tree', 'random_forest', 'car']
source_label_dict = {'decision_tree': 'DT', 'random_forest': 'RF', 'car': 'CAR'}
source_color_dict = dict(zip(source_order, sns.color_palette("husl", 3)))


fig, ax = plt.subplots(
    len(objective_names), len(dataset_experiment_dict), figsize=(28, 12), squeeze=False
)

for i, (dataset, experiment_dict) in enumerate(dataset_experiment_dict.items()):
    for j, objective in enumerate(objective_names):
        selected_dscluster_module = [
            m for m in dscluster_modules if objective == m.split(';')[1].strip()
        ][0]  # there should only be one per objective!!

        try:
            rule_source_counts_by_r = experiment_dict['modules'][selected_dscluster_module]['rule-source-counts']
            r_min = min(rule_source_counts_by_r.keys(), key=int)  # r = k, the smallest max_rules budget
            counts = rule_source_counts_by_r[r_min] or {}
    
            heights = [counts.get(src, 0) for src in source_order]
            ax[j, i].bar(
                [source_label_dict[src] for src in source_order],
                heights,
                color=[_muted(source_color_dict[src]) for src in source_order],
                edgecolor='black',
                alpha = 1.0
            )
        except:
            continue
        ax[j, i].grid(which='major', axis='y', linestyle='-', linewidth=0.8, alpha=0.5)

        if i == 0:
            ax[j, i].set_ylabel(objective_name_dict.get(objective, objective))
        if j == 0:
            ax[j, i].set_title(rf"${dataset.capitalize()}$")

        ax[j,i].set_ylim(0, sum(list(counts.values())))

fig.supxlabel("Rule Source")
fig.supylabel("Number of Rules", x=0.005)
plt.tight_layout()

plt.savefig(
    "../figures/experiments/rule_source.pdf",
    bbox_inches='tight',
    dpi=300
)

plt.show()

# %% [markdown]
# ### Bicriteria Plots
#
# 3D scatter over all three objectives (coverage, cost, rule length). Coverage (`x`) is normalized by `n`; cost (`y`) is normalized per-objective (baseline k-means cost for `coverage-cost`, `n` for `coverage-mistake`, `n choose 2` for `coverage-pairwise-distance`); rule length (`z`) is shown unnormalized. Points come from `lambda.py`: each is one swept lambda value at a fixed rule budget, not a rule-budget sweep.

# %%
# Load experiment data for the bicriteria/3D scatter plots, from `lambda.py`'s output
# (each point = one swept lambda value at a fixed rule budget, not a rule-budget sweep).
# Loading is defensive, as above.

dataset_lambda_experiment_dict = {}
for dataset in DATASETS:
    fname = "../data/experiments/" + dataset + "/lambda/exp" + EXP_REF + ".json"
    if not os.path.exists(fname):
        print(f"[lambda] skipping {dataset}: {fname} not found")
        continue
    with open(fname, 'r') as f:
        dataset_lambda_experiment_dict[dataset] = json.load(f)


# %%
# Collect experiment data for the bicriteria / 3D scatter plots.
#
# obj1 = reward (coverage), obj2 = cost, obj3 = summed rule length.
# lambda.py fits two PEC variants per objective -- 'lazy-greedy' (valid across
# the whole grid, shown as "ScaledGreedy") and 'distorted-greedy' (the paper's
# main model, valid only for lambda >= lambda*) -- both kept under their own
# scatter_dict key. Each variant's own (sorted) lambda values and the index of
# lambda* within them are carried along so the plot can draw direction-of-
# change arrows and a lambda* star for both.
#
# scatter_dict (3D plot) normalizes each axis on principled, per-axis terms
# rather than a joint min-max to [0, 1]:
#   x = obj1 / n                        (coverage, as a fraction of all points)
#   y = obj2 / cost_normalizer, where cost_normalizer is:
#       - coverage-cost:             the baseline k-means solution's raw SSE
#                                     cost. `baseline['clustering-cost']` is
#                                     stored averaged-and-normalized (divided
#                                     by n, since every point is covered
#                                     exactly once by k-means -- see
#                                     ClusteringCost.__call__), so it is
#                                     multiplied back by n here to match
#                                     RuleClusteringCost's raw (unaveraged)
#                                     units.
#       - coverage-mistake:          n
#       - coverage-pairwise-distance: n choose 2 = n * (n - 1) / 2
#   z = obj3                            (summed rule length, unnormalized)
#
# scatter_dict_2d is unrelated to the above and keeps its own, separate
# normalization: it collapses obj2/obj3 into PEC's actual weighted-cost term,
# obj4 = obj2 + alpha*obj3 (using each dataset/objective's selected alpha),
# computed from raw obj2/obj3 and min-max scaled to [0, 1] jointly across
# modules, with x = obj1 min-max scaled the same way.
#
# Uncertainty: only Decision-Tree/IDS are stochastic here. Their SE is combined
# from per-trial values using the same linear combination as the mean, then
# scaled the same way (SE scales linearly under an affine transform); all
# other modules get 0.

scatter_dict = {
    dataset: {objective: {} for objective in objective_names} for dataset in dataset_lambda_experiment_dict.keys()
}
scatter_dict_2d = {
    dataset: {objective: {} for objective in objective_names} for dataset in dataset_lambda_experiment_dict.keys()
}

def _trial_se(values_list):
    """Standard error of a per-trial 'values' list; 0.0 if values_list is None
    (deterministic)."""
    if values_list is None or len(values_list) == 0:
        return 0.0
    return float(np.std(values_list) / np.sqrt(len(values_list)))


def _module_error_series(module_dict, keys, reward, cost, alpha):
    """Standard error across trials for obj1 (reward), obj2 (cost), obj3
    (sum-rule-length), and obj4 = obj2 + alpha*obj3, aligned to `keys` order.
    0.0 wherever the underlying entry is deterministic (no 'values' list)."""
    obj1_err, obj2_err, obj3_err, obj4_err = [], [], [], []
    for k in keys:
        rv_t = _values(module_dict[reward][k])
        cv_t = _values(module_dict[cost][k])
        lv_t = _values(module_dict['sum-rule-length'][k])
        obj1_err.append(_trial_se(rv_t))
        obj2_err.append(_trial_se(cv_t))
        obj3_err.append(_trial_se(lv_t))
        if rv_t is not None and cv_t is not None and lv_t is not None:
            obj4_err.append(float(np.std(np.array(cv_t) + alpha * np.array(lv_t)) / np.sqrt(len(lv_t))))
        else:
            obj4_err.append(0.0)
    return np.array(obj1_err), np.array(obj2_err), np.array(obj3_err), np.array(obj4_err)


for dataset, experiment_dict in dataset_lambda_experiment_dict.items():
    for objective in objective_names:
        candidate_modules = [
            m for m in experiment_dict['modules'].keys()
            if 'dscluster' in m and objective == m.split(';')[1].strip()
        ]
        scaled_module = [m for m in candidate_modules if m.strip().endswith('lazy-greedy')][0]
        distorted_module = [m for m in candidate_modules if m.strip().endswith('distorted-greedy')][0]

        # lambda* is keyed in fixed-parameters by the module name with the
        # algorithm suffix stripped, e.g. 'dscluster; coverage-cost; ensemble'.
        base_module_name = distorted_module.rsplit(';', 1)[0].strip()
        lambda_star = experiment_dict['fixed-parameters']['lambda_star'][base_module_name]
        alpha_val = experiment_dict['fixed-parameters']['alpha'][base_module_name]

        reward = objective_cost_reward_dict[objective]['reward']
        cost = objective_cost_reward_dict[objective]['cost']

        # Each dscluster variant is indexed by its own lambda grid, sorted
        # ascending -- 'distorted-greedy' is only valid/recorded for
        # lambda >= lambda*, so its key set is a strict subset of
        # 'scaled-greedy's, and the sort order is what lets the plotting cell
        # draw arrows from smaller to larger lambda.
        scaled_lam_dict = experiment_dict['modules'][scaled_module]['lambda']
        scaled_keys = sorted(scaled_lam_dict.keys(), key=lambda k: scaled_lam_dict[k])
        distorted_lam_dict = experiment_dict['modules'][distorted_module]['lambda']
        distorted_keys = sorted(distorted_lam_dict.keys(), key=lambda k: distorted_lam_dict[k])

        # Gather raw (unnormalized) objective values (and their trial-to-trial
        # std, for stochastic modules) per module first, so that normalization
        # constants can be computed jointly across all modules.
        raw_values = {}
        raw_errs = {}
        raw_lambdas = {}
        modules_to_process = [cmod for cmod in comparison_modules if cmod in experiment_dict['modules']]
        for mod in modules_to_process:
            obj1 = np.array([_scalar(experiment_dict['modules'][mod][reward][k]) for k in scaled_keys])
            obj2 = np.array([_scalar(experiment_dict['modules'][mod][cost][k]) for k in scaled_keys])
            obj3 = np.array([_scalar(experiment_dict['modules'][mod]['sum-rule-length'][k]) for k in scaled_keys])
            raw_values[mod] = (obj1, obj2, obj3)
            raw_errs[mod] = _module_error_series(experiment_dict['modules'][mod], scaled_keys, reward, cost, alpha_val)

        for mod, keys, lam_dict in [
            (scaled_module, scaled_keys, scaled_lam_dict),
            (distorted_module, distorted_keys, distorted_lam_dict),
        ]:
            obj1 = np.array([_scalar(experiment_dict['modules'][mod][reward][k]) for k in keys])
            obj2 = np.array([_scalar(experiment_dict['modules'][mod][cost][k]) for k in keys])
            obj3 = np.array([_scalar(experiment_dict['modules'][mod]['sum-rule-length'][k]) for k in keys])
            raw_values[mod] = (obj1, obj2, obj3)
            raw_errs[mod] = _module_error_series(experiment_dict['modules'][mod], keys, reward, cost, alpha_val)
            raw_lambdas[mod] = np.array([lam_dict[k] for k in keys])

        # Min-max normalization constants, shared across modules so scaled
        # values remain directly comparable within this (dataset, objective).
        obj1_lo = min(v[0].min() for v in raw_values.values())
        obj1_hi = max(v[0].max() for v in raw_values.values())
        obj2_lo = min(v[1].min() for v in raw_values.values())
        obj2_hi = max(v[1].max() for v in raw_values.values())
        obj3_lo = min(v[2].min() for v in raw_values.values())
        obj3_hi = max(v[2].max() for v in raw_values.values())

        def _minmax(v, lo, hi):
            rng = hi - lo
            return (v - lo) / rng if rng > 0 else np.zeros_like(v)

        def _minmax_err(err, lo, hi):
            rng = hi - lo
            return err / rng if rng > 0 else np.zeros_like(err)

        # Per-axis normalization for the 3D scatter (scatter_dict): coverage by n,
        # cost by an objective-specific constant, rule length left unnormalized.
        # See the "Collect experiment data" comment above for the rationale.
        n = experiment_dict['fixed-parameters']['n']
        if objective == 'coverage-cost':
            # baseline['clustering-cost'] is k-means' own cost, stored averaged
            # (no-op here -- k-means covers every point exactly once) and
            # normalized (divided by n, since coverage = n). Multiplying back
            # by n recovers the raw SSE, matching RuleClusteringCost's units.
            cost_normalizer = experiment_dict['baseline']['KMeans']['clustering-cost'] * n
        elif objective == 'coverage-mistake':
            cost_normalizer = n
        elif objective == 'coverage-pairwise-distance':
            cost_normalizer = n * (n - 1) / 2
        else:
            raise ValueError(f"No cost normalizer defined for objective '{objective}'.")

        for mod, (obj1, obj2, obj3) in raw_values.items():
            x = obj1 / n
            y = obj2 / cost_normalizer
            z = obj3
            obj1_err, obj2_err, obj3_err, _ = raw_errs[mod]
            x_err = obj1_err / n
            y_err = obj2_err / cost_normalizer
            z_err = obj3_err
            if mod == distorted_module:
                lam = raw_lambdas[mod]
                lambda_star_idx = int(np.argmin(np.abs(lam - lambda_star)))
                scatter_dict[dataset][objective]['dscluster; ensemble'] = {
                    'x': x, 'y': y, 'z': z,
                    'x_err': x_err, 'y_err': y_err, 'z_err': z_err,
                    'lam': lam, 'lambda_star_idx': lambda_star_idx,
                }
            elif mod == scaled_module:
                lam = raw_lambdas[mod]
                lambda_star_idx = int(np.argmin(np.abs(lam - lambda_star)))
                scatter_dict[dataset][objective]['dscluster; ensemble; lazy-greedy'] = {
                    'x': x, 'y': y, 'z': z,
                    'x_err': x_err, 'y_err': y_err, 'z_err': z_err,
                    'lam': lam, 'lambda_star_idx': lambda_star_idx,
                }
            else:
                scatter_dict[dataset][objective][mod] = {
                    'x': x, 'y': y, 'z': z,
                    'x_err': x_err, 'y_err': y_err, 'z_err': z_err,
                    'lam': None, 'lambda_star_idx': None,
                }

        # obj4 = obj2 + alpha * obj3, collapsed back to a single weighted-cost
        # axis; normalized jointly across modules, independent of obj2/obj3's
        # own normalization above.
        obj4_raw = {mod: v[1] + alpha_val * v[2] for mod, v in raw_values.items()}
        obj4_lo = min(v.min() for v in obj4_raw.values())
        obj4_hi = max(v.max() for v in obj4_raw.values())

        for mod, (obj1, obj2, obj3) in raw_values.items():
            x = _minmax(obj1, obj1_lo, obj1_hi)
            y4 = _minmax(obj4_raw[mod], obj4_lo, obj4_hi)
            obj1_err, _, _, obj4_err = raw_errs[mod]
            x_err = _minmax_err(obj1_err, obj1_lo, obj1_hi)
            y4_err = _minmax_err(obj4_err, obj4_lo, obj4_hi)
            if mod == distorted_module:
                lam = raw_lambdas[mod]
                lambda_star_idx = int(np.argmin(np.abs(lam - lambda_star)))
                scatter_dict_2d[dataset][objective]['dscluster; ensemble'] = {
                    'x': x, 'y4': y4, 'x_err': x_err, 'y4_err': y4_err,
                    'lam': lam, 'lambda_star_idx': lambda_star_idx,
                }
            elif mod == scaled_module:
                lam = raw_lambdas[mod]
                lambda_star_idx = int(np.argmin(np.abs(lam - lambda_star)))
                scatter_dict_2d[dataset][objective]['dscluster; ensemble; lazy-greedy'] = {
                    'x': x, 'y4': y4, 'x_err': x_err, 'y4_err': y4_err,
                    'lam': lam, 'lambda_star_idx': lambda_star_idx,
                }
            else:
                scatter_dict_2d[dataset][objective][mod] = {
                    'x': x, 'y4': y4, 'x_err': x_err, 'y4_err': y4_err,
                    'lam': None, 'lambda_star_idx': None,
                }

# %% [markdown]
# `x` = obj1 / n stays on a shared `[0, 1]` scale across every panel. `y` = obj2 / cost_normalizer and `z` = obj3 (unnormalized) are not comparable in scale across datasets (n and the cost normalizers vary by orders of magnitude), so each panel gets its own `y`/`z` limits and "nice" 3-tick labels, computed from that panel's own data. Kept simple otherwise -- one fixed viewing angle, matching color/marker encoding used elsewhere -- to stay legible in print.

# %%
# Plot results: 3D scatter over (obj1, obj2, obj3). Each point gets a drop-line
# to the z=0 floor plus a faint projection marker, since exact positions are
# hard to read once a point is lifted off the floor in 3D.
#
# 'distorted-greedy' PEC (paper's main model) is a bold, opaque diamond;
# 'scaled-greedy' is a small, faint, edge-less circle -- deliberately
# understated so the two stay legible where they nearly overlap. Both get
# direction-of-change arrows between consecutive lambda values (solid vs.
# dashed) and a star at lambda*.
#
# Uncertainty: Decision-Tree/IDS get thin per-axis error segments; all other
# modules' err arrays are 0, so no segment is drawn for them.

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers the '3d' projection)
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch

# mplot3d recomputes zorder every draw from camera distance, ignoring static
# zorder= kwargs. ax.computed_zorder = False restores normal 2D-style zorder
# (set per-axis below, right after each ax is created).

# Beta's subscript names the objective's cost term, not just "beta".
beta_subscript_dict = {
    'coverage-cost': 'K',
    'coverage-mistake': 'M',
    'coverage-pairwise-distance': 'P',
}

class _Arrow3D(FancyArrowPatch):
    """A FancyArrowPatch whose 3D endpoints are (re)projected to 2D at draw
    time. Unlike quiver -- whose arrowhead is built from data-space vectors,
    so its on-screen size gets distorted once x, y, z no longer share a
    common scale -- FancyArrowPatch's `mutation_scale` sizes the head in
    points, a fixed physical unit independent of any axis's data range.
    """
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return min(zs)

# Single parameter controlling arrowhead size, in points -- constant on
# screen in every panel regardless of that panel's y/z data ranges.
ARROW_HEAD_SIZE = 14

def _draw_lambda_arrows(ax, x, y, z, lam, color, *, dashed, alpha, linewidth, zorder):
    """Direction-of-change arrows between consecutive (sorted-by-lambda)
    points, one FancyArrowPatch (shaft + head) per segment -- see _Arrow3D
    for why this keeps the head a constant size across panels.
    """
    if lam is None or len(lam) <= 1:
        return
    order = np.argsort(lam)
    xs, ys, zs = x[order], y[order], z[order]
    for k in range(len(xs) - 1):
        arrow = _Arrow3D(
            [xs[k], xs[k + 1]], [ys[k], ys[k + 1]], [zs[k], zs[k + 1]],
            mutation_scale=ARROW_HEAD_SIZE, lw=linewidth, arrowstyle='-|>',
            shrinkA=0, shrinkB=0, color=color, alpha=alpha,
            linestyle='dashed' if dashed else 'solid', zorder=zorder,
        )
        ax.add_artist(arrow)

def _panel_axis_ticks(module_result_dict, key, err_key, exclude_keys=()):
    """3 'nice' ticks spanning [0, max(value + error)] across every module in
    this panel except those in `exclude_keys`, via the shared
    nice_lim_for_3_ticks helper (unclipped above, since y/z are no longer
    bounded to [0, 1])."""
    included = {m: pts for m, pts in module_result_dict.items() if m not in exclude_keys}
    values = np.concatenate([pts[key] for pts in included.values()])
    errs = np.concatenate([pts[err_key] for pts in included.values()])
    raw_max = float(np.nanmax(values + errs)) if values.size else 0.0
    return nice_lim_for_3_ticks(0.0, raw_max, clip=(0.0, np.inf))

def _panel_tick_labels(ticks):
    """Format 'nice' ticks with just enough decimals for their step size."""
    step = ticks[1] - ticks[0]
    decimals = 0 if step >= 1 or step <= 0 else max(0, int(np.ceil(-np.log10(step))))
    return [f"{t:.{decimals}f}" for t in ticks]

def _visible_err_frac(err, axis_range, frac=0.03):
    """Like _visible_err, but the floor is a fraction of this panel's own
    axis range instead of the fixed [0,1]-scale MIN_VISIBLE_ERR -- y and z no
    longer share that scale across panels."""
    err = np.asarray(err, dtype=float)
    floor = frac * axis_range if axis_range > 0 else 0.0
    return np.where(err > 0, np.maximum(err, floor), 0.0)

# Pastel (pre-lightened) tab10 colors, used only in this 3D scatter -- alpha-based
# translucency doesn't composite consistently in mplot3d (compositing order depends
# on per-draw camera distance), so colors are pre-lightened instead.
pastel_color_dict = {k: _muted(v, amount=0.35) for k, v in color_dict.items()}

# One figure per objective, 3 columns x 2 rows over datasets, so each panel
# stays legible and each objective saves to its own output file.
n_cols = 3
n_datasets = len(scatter_dict)
n_rows = int(np.ceil(n_datasets / n_cols))

for objective in objective_names:
    fig, axs = plt.subplots(
        n_rows, n_cols,
        figsize=(6.0 * n_cols, 5.0 * n_rows), subplot_kw={'projection': '3d'}, squeeze=False,
    )

    for idx, dataset in enumerate(scatter_dict.keys()):
        row, col = divmod(idx, n_cols)
        ax = axs[row, col]
        ax.computed_zorder = False

        module_result_dict = scatter_dict[dataset][objective]

        # x = coverage / n is bounded to [0, 1] for every dataset, so it keeps
        # a fixed, shared scale. y (cost) and z (rule length) are not bounded
        # this way -- their scale depends on this dataset's n and cost
        # normalizer -- so this panel gets its own "nice" 3-tick range for
        # each, computed from its own data (+ error) before anything is drawn.
        #
        # ScaledGreedy (lazy-greedy) is excluded from the y range specifically:
        # its lambda=0 point can have a far larger cost than every other
        # algorithm, and letting it set the axis scale would squeeze
        # everything else into a sliver near the floor. The y limit is instead
        # sized to the other algorithms, and any ScaledGreedy point above it
        # is simply clipped off the top of the panel.
        x_lo, x_hi = 0.0, 1.0
        y_lo, y_hi, y_ticks = _panel_axis_ticks(
            module_result_dict, 'y', 'y_err', exclude_keys={'dscluster; ensemble; lazy-greedy'}
        )
        z_lo, z_hi, z_ticks = _panel_axis_ticks(module_result_dict, 'z', 'z_err')
        y_range = y_hi - y_lo
        z_range = z_hi - z_lo

        for module, pts in module_result_dict.items():
            x, y, z = pts['x'], pts['y'], pts['z']
            x_err, y_err, z_err = pts['x_err'], pts['y_err'], pts['z_err']
            lam = pts['lam']
            lambda_star_idx = pts['lambda_star_idx']
            is_scaled = module.endswith('lazy-greedy')
            # Scaled-greedy is further muted on top of the pastel base to stay
            # understated relative to distorted-greedy PEC, the main model.
            color = _muted(pastel_color_dict.get(module, 'grey'), amount=1/3) if is_scaled else pastel_color_dict.get(module, 'grey')

            # Floor drop-line + projection marker (drawn first, low zorder,
            # so the actual data points render on top of them).
            for xi, yi, zi in zip(x, y, z):
                ax.plot([xi, xi], [yi, yi], [0, zi], color=color, alpha=0.5, linewidth=1.0, zorder=1)
                ax.scatter(xi, yi, 0, color=color, s=15, alpha=0.5, edgecolor='none', zorder=1)

            # Per-axis error segments (only nonzero for Decision-Tree/IDS).
            # x stays on the shared [0,1] scale (global _visible_err floor);
            # y/z get a floor relative to this panel's own range instead.
            x_err_disp = _visible_err(x_err)
            y_err_disp = _visible_err_frac(y_err, y_range)
            z_err_disp = _visible_err_frac(z_err, z_range)
            for xi, yi, zi, xe, ye, ze in zip(x, y, z, x_err_disp, y_err_disp, z_err_disp):
                if xe > 0:
                    ax.plot([xi - xe, xi + xe], [yi, yi], [zi, zi], color=color, alpha=0.5, linewidth=1.5, zorder=2)
                if ye > 0:
                    ax.plot([xi, xi], [yi - ye, yi + ye], [zi, zi], color=color, alpha=0.5, linewidth=1.5, zorder=2)
                if ze > 0:
                    ax.plot([xi, xi], [yi, yi], [zi - ze, zi + ze], color=color, alpha=0.5, linewidth=1.5, zorder=2)

            if is_scaled:
                # Faint, edge-less markers -- deliberately understated so this
                # secondary curve doesn't compete with distorted-greedy.
                ax.scatter(
                    x, y, z,
                    color=color, marker='o', s=80, alpha=0.9,
                    edgecolor='k', depthshade=False, zorder=2,
                )
                _draw_lambda_arrows(
                    ax, x, y, z, lam, color,
                    dashed=True, alpha=0.9, linewidth=2.0, zorder=1,
                )

                if lambda_star_idx is not None:
                    ax.scatter(
                        x[lambda_star_idx], y[lambda_star_idx], z[lambda_star_idx],
                        marker='*', s=600, color=color, edgecolor='black',
                        alpha=0.9, linewidth=1.0, depthshade=False, zorder=5,
                    )
            else:
                ax.scatter(
                    x, y, z,
                    label=module,
                    color=color,
                    marker='o',
                    s=240,
                    edgecolor='k',
                    alpha=0.9,
                    depthshade=False,
                    zorder=3,
                )

                _draw_lambda_arrows(
                    ax, x, y, z, lam, color,
                    dashed=False, alpha=0.9, linewidth=2.0, zorder=2,
                )

                if lambda_star_idx is not None:
                    ax.scatter(
                        x[lambda_star_idx], y[lambda_star_idx], z[lambda_star_idx],
                        marker='*', s=600, color=color, edgecolor='black',
                        linewidth=1.2, depthshade=False, zorder=6, alpha=0.9
                    )

        # Small margins proportional to each axis's own range (x keeps a
        # fixed absolute margin since it's always [0, 1]).
        y_margin = 0.05 * y_range if y_range > 0 else 0.05
        z_margin = 0.05 * z_range if z_range > 0 else 0.05
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(y_lo - y_margin, y_hi + y_margin)
        ax.set_zlim(z_lo - z_margin, z_hi + z_margin)
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks(y_ticks)
        ax.set_zticks(z_ticks)
        ax.set_xticklabels(['0.0', '0.5', '1.0'])
        ax.set_yticklabels(_panel_tick_labels(y_ticks))
        ax.set_zticklabels(_panel_tick_labels(z_ticks))
        ax.tick_params(axis='both', which='major', labelsize=16, pad=0)
        # Zoom the cube slightly within its own bounding box to claw back
        # some of the whitespace 3D subplots otherwise reserve for rotation.
        ax.set_box_aspect(None, zoom=1.12)

        ax.view_init(elev=22, azim=-60)

        if dataset == "kddcup":
            ax.set_title(rf"$KDDCup$", pad=6)
        else:
            ax.set_title(rf"${dataset.capitalize()}$", pad=6, fontsize=28)

        ax.set_ylabel(rf"$\bar{{\beta}}_{{{beta_subscript_dict.get(objective, '')}}}$", labelpad=6, fontsize=22)
        ax.set_xlabel(r"$\bar{g}$", labelpad=2, fontsize=22)
        # set_zlabel places the label far from the z-axis regardless of rotation
        # settings (and clips under usetex); text2D in axes-fraction coords pins
        # it next to the z-axis instead.
        ax.set_zlabel("")
        ax.text2D(
            1.12, 0.58, r"$s$",
            transform=ax.transAxes, fontsize=22, ha='left', va='center',
        )

        # Light, uncluttered panes/gridlines so points stay legible in print
        ax.xaxis.pane.set_alpha(0.05)
        ax.yaxis.pane.set_alpha(0.05)
        ax.zaxis.pane.set_alpha(0.05)
        ax.grid(True, linestyle=':', linewidth=0.6, alpha=0.5)

    # Hide any trailing unused panels (n_datasets not a multiple of n_cols).
    for idx in range(n_datasets, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axs[row, col].set_axis_off()

    # 3D subplots reserve extra layout margin for rotation, which makes
    # tight_layout() leave excess whitespace -- pack the grid manually instead.
    fig.subplots_adjust(left=0.02, right=0.90, top=0.90, bottom=0.05, wspace=0.15, hspace=0.2)

    plt.savefig(
        f"../figures/experiments/bicriteria_3d_{objective}.pdf",
        bbox_inches='tight',
        dpi=300
    )
    plt.show()

# %%
# Create a separate legend for the scatter plot with marker styles
fig, ax = plt.subplots(figsize=(34, 2.2))

module_order = [
    'Decision-Tree',
    'ExKMC',
    #'WRA',
    'IDS',
    'CBA',
    'CN2',
    'dscluster; ensemble; lazy-greedy',
    'dscluster; ensemble',
]

legend_labels = dict(title_dict) | {
    'dscluster; ensemble; lazy-greedy': r'\texttt{ScaledGreedy}',
}

legend_elements = []
for cmod in module_order:
    color = _muted(color_dict[cmod])
    legend_elements.append(
        mlines.Line2D(
            [], [],
            color=color,
            marker='o',
            markersize=30,
            markeredgecolor='k',
            markeredgewidth=1.5,
            alpha=0.9,
            linestyle='None',
            linewidth=2.0,
            label=legend_labels.get(cmod, cmod)
        )
    )

# lambda* markers -- one per PEC variant (lambda* is a grid point for both),
# each in that variant's own color so it's clear which curve it belongs to.
legend_elements.append(
    mlines.Line2D(
        [], [],
        color='white',
        marker='*',
        markersize=30,
        markeredgecolor='k',
        markeredgewidth=1.2,
        linestyle='None',
        label=r'$\lambda^*$'
    )
)

ax.legend(handles=legend_elements, ncol=9, loc='center', frameon=False, columnspacing=1.4, handletextpad=0.5)
ax.axis('off')

plt.savefig(
    "../figures/experiments/bicriteria_legend.pdf",
    bbox_inches='tight',
    dpi=300
)

plt.show()

# %% [markdown]
# ### Bicriteria Plots (2D, Collapsed)
#
# Same data as the 3D scatter, but cost and rule length are collapsed into PEC's actual weighted-cost term, `obj4 = obj2 + alpha * obj3` -- this is what makes distorted-greedy and scaled-greedy clearly separable on a single axis where the 3D view often has them nearly overlapping.

# %%
# Plot results: 2D scatter over (obj1, obj4), same visual encoding as the 3D
# scatter (distorted-greedy = bold diamond + solid arrow; scaled-greedy =
# faint circle + dashed arrow; comparison models = plain points, no arrows).
# Uncertainty: Decision-Tree/IDS get errorbar whiskers; safe to call
# unconditionally since other modules' err arrays are 0.

function_name_dict = {0: 'K', 1: 'M', 2: 'P'}

# Sized per-panel so aspect ratio (and the quiver arrows) stays reasonable
# regardless of how many datasets are loaded.
fig, axs = plt.subplots(
    len(objective_names), len(scatter_dict_2d),
    figsize=(6.5 * len(scatter_dict_2d), 4.5 * len(objective_names)), squeeze=False,
)

for i, (dataset, objective_result_dict) in enumerate(scatter_dict_2d.items()):
    for j, (objective, module_result_dict) in enumerate(objective_result_dict.items()):
        ax = axs[j, i]
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.grid(True, which='major', linestyle=':', linewidth=0.8, alpha=0.9)
        ax.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.5)

        for module, pts in module_result_dict.items():
            x, y4 = pts['x'], pts['y4']
            x_err, y4_err = pts['x_err'], pts['y4_err']
            lam = pts['lam']
            lambda_star_idx = pts['lambda_star_idx']
            is_scaled = module.endswith('lazy-greedy')
            color = _muted(color_dict.get(module, 'grey'))

            if is_scaled:
                ax.errorbar(
                    x, y4, xerr=_visible_err(x_err), yerr=_visible_err(y4_err), fmt='none',
                    ecolor=color, alpha=0.5, capsize=3, zorder=1,
                )
                
                ax.scatter(
                    x, y4,
                    color=color, marker='o', s=250, alpha=0.9,
                    edgecolor='k', zorder=2,
                )
                if len(lam) > 1:
                    order = np.argsort(lam)
                    xs, ys = x[order], y4[order]
                    dx, dy = np.diff(xs), np.diff(ys)
                    arrows = ax.quiver(
                        xs[:-1], ys[:-1], dx, dy,
                        angles='xy', scale_units='xy', scale=1,linestyle="dashed",
                        # 'dots' keeps the arrow shaft width an absolute size
                        # (independent of axes aspect ratio), so it doesn't
                        # balloon if a panel ends up short and wide.
                        units='dots', width=4.0,
                        color=color, alpha=0.5,
                        headwidth=5, headlength=6, zorder=2,
                    )
                    arrows.set_linestyle('dashed')

                if lambda_star_idx is not None:
                    ax.scatter(
                        x[lambda_star_idx], y4[lambda_star_idx],
                        marker='*', s=700, color=color, edgecolor='black',
                        alpha=0.9, linewidth=1.0, zorder=5,
                    )
            else:
                ax.errorbar(
                    x, y4, xerr=_visible_err(x_err), yerr=_visible_err(y4_err), fmt='none',
                    ecolor=color, alpha=0.5, capsize=3, zorder=1,
                )
                ax.scatter(
                    x, y4,
                    label=module,
                    color=color,
                    marker='o',
                    s=500,
                    edgecolor='k',
                    alpha=0.9,
                    zorder=3,
                )

                if lam is not None and len(lam) > 1:
                    order = np.argsort(lam)
                    xs, ys = x[order], y4[order]
                    dx, dy = np.diff(xs), np.diff(ys)
                    ax.quiver(
                        xs[:-1], ys[:-1], dx, dy,
                        angles='xy', scale_units='xy', scale=1, linestyle="dashed",
                        units='dots', width=4.8,
                        color=color, alpha=0.5,
                        headwidth=5, headlength=6, zorder=4,
                    )

                if lambda_star_idx is not None:
                    ax.scatter(
                        x[lambda_star_idx], y4[lambda_star_idx],
                        marker='*', s=700, color=color, edgecolor='black',
                        linewidth=1.2, zorder=6, alpha=0.9
                    )

        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 0.5)
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.25, 0.5])
        ax.tick_params(axis='both', which='major', labelsize=28)

        if j == 0:
            if dataset == "kddcup":
                ax.set_title(rf"$KDDCup$", pad=10)
            else:
                ax.set_title(rf"${dataset.capitalize()}$", pad=10)

        if j == len(objective_names) - 1:
            ax.set_xlabel(r"$g$", fontsize=36)
        if i == 0:
            ax.set_ylabel(rf"$h_{{{function_name_dict[j]}}}$", fontsize=36, rotation=0, labelpad=40)

plt.tight_layout()

plt.savefig(
    "../figures/experiments/bicriteria_2d.pdf",
    bbox_inches='tight',
    dpi=300
)
plt.show()

# %% [markdown]
# ### Weighted Average Rule Length

# %%
# Collect the weighted average lengths for each dataset and objective, as
# (mean, SE) tuples (SE=NaN for deterministic modules; see `_se`).
weighted_avg_dict = {
    (objective, dataset): {}
    for dataset in dataset_experiment_dict.keys()
    for objective in objective_names
}

for dataset, experiment_dict in dataset_experiment_dict.items():
    for objective in objective_names:

        for cmod in comparison_modules:
            if cmod not in experiment_dict['modules']:
                continue

            cmod_weighted_avg_lengths = experiment_dict['modules'][cmod]['weighted-avg-length']
            smallest_r = str(min([int(l) for l in cmod_weighted_avg_lengths.keys()]))
            cmod_weighted_avg_length = _scalar(cmod_weighted_avg_lengths[smallest_r])
            cmod_weighted_avg_length_se = _se(cmod_weighted_avg_lengths[smallest_r])
            weighted_avg_dict[(objective, dataset)][cmod] = (cmod_weighted_avg_length, cmod_weighted_avg_length_se)

        selected_dscluster_module = [
            m for m in dscluster_modules if objective == m.split(';')[1].strip()
        ][0]
        dmod_name = selected_dscluster_module.split(';')[0] + ';' + selected_dscluster_module.split(';')[2]
        dmod_weighted_avg_lengths = experiment_dict['modules'][selected_dscluster_module]['weighted-avg-length']
        smallest_r = str(min([int(l) for l in dmod_weighted_avg_lengths.keys()]))
        dmod_weighted_avg_length = _scalar(dmod_weighted_avg_lengths[smallest_r])
        dmod_weighted_avg_length_se = _se(dmod_weighted_avg_lengths[smallest_r])
        weighted_avg_dict[(objective, dataset)][dmod_name] = (dmod_weighted_avg_length, dmod_weighted_avg_length_se)

        # PEC scaled-greedy (selection-algorithm ablation), optional per the
        # "missing_scaled" note in the loading cell above.
        scaled_matches = [
            m for m in scaled_dscluster_modules
            if objective == m.split(';')[1].strip() and m in experiment_dict['modules']
        ]
        if scaled_matches:
            selected_scaled_module = scaled_matches[0]
            smod_name = selected_scaled_module.split(';')[0] + ';' + selected_scaled_module.split(';')[2] + ';' + selected_scaled_module.split(';')[3]
            smod_weighted_avg_lengths = experiment_dict['modules'][selected_scaled_module]['weighted-avg-length']
            smallest_r = str(min([int(l) for l in smod_weighted_avg_lengths.keys()]))
            smod_weighted_avg_length = _scalar(smod_weighted_avg_lengths[smallest_r])
            smod_weighted_avg_length_se = _se(smod_weighted_avg_lengths[smallest_r])
            weighted_avg_dict[(objective, dataset)][smod_name] = (smod_weighted_avg_length, smod_weighted_avg_length_se)


# %%
# Weighted rule length table: "mean ± SE" for Decision-Tree/IDS, plain number otherwise.
def _fmt_weighted_avg(cell):
    mean, se = cell
    if np.isnan(se):
        return f"{mean:.3f}"
    return f"{mean:.3f} ± {se:.3f}"

pd.DataFrame({
    col: {row: _fmt_weighted_avg(val) for row, val in rows.items()}
    for col, rows in weighted_avg_dict.items()
})

# %% [markdown]
# ### Weighted Average Rule Length (LaTeX Table)

# %%
# LaTeX table built from `weighted_avg_dict` above: one row per algorithm,
# one column per dataset. ScaledGreedy is shown only for the k-Means
# (coverage-cost) objective -- it's PEC's selection-algorithm ablation, not a
# full second baseline, so it doesn't get a row per objective the way PEC
# does. The 3 smallest (best) means in each dataset column are bolded;
# comparing means only (not SE), ties broken by whichever sorts first.
_pec_key = 'dscluster; ensemble'
_scaled_greedy_key = 'dscluster; ensemble; lazy-greedy'

_table_rows = [
    (r'\texttt{DecisionTree}', 'coverage-cost', 'Decision-Tree'),
    (r'\texttt{ExKMC}', 'coverage-cost', 'ExKMC'),
    (r'\texttt{CBA}', 'coverage-cost', 'CBA'),
    (r'\texttt{CN2}', 'coverage-cost', 'CN2'),
    (r'\texttt{IDS}', 'coverage-cost', 'IDS'),
    (r'\texttt{ScaledGreedy} (\emph{k-Means}, $h_K$)', 'coverage-cost', _scaled_greedy_key),
    (r'\texttt{PEC} (\emph{k-Means}, $h_K$)', 'coverage-cost', _pec_key),
    (r'\texttt{PEC} (\emph{Mistakes}, $h_M$)', 'coverage-mistake', _pec_key),
    (r'\texttt{PEC} (\emph{Pairwise Distance}, $h_P$)', 'coverage-pairwise-distance', _pec_key),
]

_datasets = list(dataset_experiment_dict.keys())

def _dataset_header(dataset):
    return "KDDCup" if dataset == "kddcup" else dataset.capitalize()

# cell_values[label][dataset] = (mean, se) or None if that module has no
# entry for that (objective, dataset) -- e.g. a dataset missing scaled-greedy.
_cell_values = {
    label: {
        dataset: weighted_avg_dict.get((objective, dataset), {}).get(module_key)
        for dataset in _datasets
    }
    for label, objective, module_key in _table_rows
}

# Per dataset (column), the 3 rows with the smallest mean get bolded.
_bold_labels = {
    dataset: {
        label for label, _ in sorted(
            (
                (label, _cell_values[label][dataset][0])
                for label, _, _ in _table_rows
                if _cell_values[label][dataset] is not None
            ),
            key=lambda t: t[1],
        )[:3]
    }
    for dataset in _datasets
}

def _fmt_latex_cell(label, dataset):
    entry = _cell_values[label][dataset]
    if entry is None:
        return "--"
    mean, se = entry
    text = f"{mean:.3f}" if np.isnan(se) else rf"{mean:.3f} $\pm$ {se:.3f}"
    return rf"\textbf{{{text}}}" if label in _bold_labels[dataset] else text

_lines = [
    r"\begin{table}[t]",
    r"\centering",
    rf"\begin{{tabular}}{{l{'c' * len(_datasets)}}}",
    r"\toprule",
    "Model & " + " & ".join(rf"\emph{{{_dataset_header(d)}}}" for d in _datasets) + r" \\",
    r"\midrule",
]
for label, _, _ in _table_rows:
    _lines.append(label + " & " + " & ".join(_fmt_latex_cell(label, d) for d in _datasets) + r" \\")
_lines += [
    r"\bottomrule",
    r"\end{tabular}",
    r"\caption{Weighted average rule length (mean $\pm$ SE where applicable).}",
    r"\label{tab:weighted-avg-rule-length}",
    r"\end{table}",
]

print("\n".join(_lines))

# %% [markdown]
# ### Overlap

# %%
# Collect the average cluster overlap for each dataset and objective, as
# (mean, SE) tuples (SE=NaN for deterministic modules; see `_se`).
overlap_dict = {
    (objective, dataset): {}
    for dataset in dataset_experiment_dict.keys()
    for objective in objective_names
}

for dataset, experiment_dict in dataset_experiment_dict.items():
    for objective in objective_names:

        for cmod in comparison_modules:
            if cmod not in experiment_dict['modules']:
                continue

            cmod_overlaps = experiment_dict['modules'][cmod]['overlap']
            smallest_r = str(min([int(l) for l in cmod_overlaps.keys()]))
            cmod_overlap = _scalar(cmod_overlaps[smallest_r])
            cmod_overlap_se = _se(cmod_overlaps[smallest_r])
            overlap_dict[(objective, dataset)][cmod] = (cmod_overlap, cmod_overlap_se)

        selected_dscluster_module = [
            m for m in dscluster_modules if objective == m.split(';')[1].strip()
        ][0]
        dmod_name = selected_dscluster_module.split(';')[0] + ';' + selected_dscluster_module.split(';')[2]
        dmod_overlaps = experiment_dict['modules'][selected_dscluster_module]['overlap']
        smallest_r = str(min([int(l) for l in dmod_overlaps.keys()]))
        dmod_overlap = _scalar(dmod_overlaps[smallest_r])
        dmod_overlap_se = _se(dmod_overlaps[smallest_r])
        overlap_dict[(objective, dataset)][dmod_name] = (dmod_overlap, dmod_overlap_se)

        # PEC scaled-greedy (selection-algorithm ablation), optional per the
        # "missing_scaled" note in the loading cell above.
        scaled_matches = [
            m for m in scaled_dscluster_modules
            if objective == m.split(';')[1].strip() and m in experiment_dict['modules']
        ]
        if scaled_matches:
            selected_scaled_module = scaled_matches[0]
            smod_name = selected_scaled_module.split(';')[0] + ';' + selected_scaled_module.split(';')[2] + ';' + selected_scaled_module.split(';')[3]
            smod_overlaps = experiment_dict['modules'][selected_scaled_module]['overlap']
            smallest_r = str(min([int(l) for l in smod_overlaps.keys()]))
            smod_overlap = _scalar(smod_overlaps[smallest_r])
            smod_overlap_se = _se(smod_overlaps[smallest_r])
            overlap_dict[(objective, dataset)][smod_name] = (smod_overlap, smod_overlap_se)

# %%
# Overlap table: "mean ± SE" for Decision-Tree/IDS, plain number otherwise.
def _fmt_overlap(cell):
    mean, se = cell
    if np.isnan(se):
        return f"{mean:.3f}"
    return f"{mean:.3f} ± {se:.3f}"

pd.DataFrame({
    col: {row: _fmt_overlap(val) for row, val in rows.items()}
    for col, rows in overlap_dict.items()
})

# %% [markdown]
# ### Overlap (LaTeX Table)

# %%
# LaTeX table built from `overlap_dict` above: one row per algorithm,
# one column per dataset. ScaledGreedy is shown only for the k-Means
# (coverage-cost) objective -- it's PEC's selection-algorithm ablation, not a
# full second baseline, so it doesn't get a row per objective the way PEC
# does. The 3 smallest (best, i.e. least-overlapping) means in each dataset
# column are bolded; comparing means only (not SE), ties broken by whichever
# sorts first.
_overlap_cell_values = {
    label: {
        dataset: overlap_dict.get((objective, dataset), {}).get(module_key)
        for dataset in _datasets
    }
    for label, objective, module_key in _table_rows
}

# Per dataset (column), the 3 rows with the smallest mean get bolded.
_overlap_bold_labels = {
    dataset: {
        label for label, _ in sorted(
            (
                (label, _overlap_cell_values[label][dataset][0])
                for label, _, _ in _table_rows
                if _overlap_cell_values[label][dataset] is not None
            ),
            key=lambda t: t[1],
        )[:3]
    }
    for dataset in _datasets
}

def _fmt_overlap_latex_cell(label, dataset):
    entry = _overlap_cell_values[label][dataset]
    if entry is None:
        return "--"
    mean, se = entry
    text = f"{mean:.3f}" if np.isnan(se) else rf"{mean:.3f} $\pm$ {se:.3f}"
    return rf"\textbf{{{text}}}" if label in _overlap_bold_labels[dataset] else text

_overlap_lines = [
    r"\begin{table}[t]",
    r"\centering",
    rf"\begin{{tabular}}{{l{'c' * len(_datasets)}}}",
    r"\toprule",
    "Model & " + " & ".join(rf"\emph{{{_dataset_header(d)}}}" for d in _datasets) + r" \\",
    r"\midrule",
]
for label, _, _ in _table_rows:
    _overlap_lines.append(label + " & " + " & ".join(_fmt_overlap_latex_cell(label, d) for d in _datasets) + r" \\")
_overlap_lines += [
    r"\bottomrule",
    r"\end{tabular}",
    r"\caption{Average cluster overlap (mean $\pm$ SE where applicable).}",
    r"\label{tab:overlap}",
    r"\end{table}",
]

print("\n".join(_overlap_lines))

# %% [markdown]
# ### Uncertainty

# %%
# Collect the distributions of covered weights for each dataset and objective.
# Compares weighted vs. unweighted PEC only, both deterministic single-fit
# models -- no repeated-trial uncertainty applies here.

distribution_dict = {
    dataset: {objective: None for objective in objective_names} for dataset in dataset_experiment_dict.keys()
}
for dataset, experiment_dict in dataset_experiment_dict.items():
    fixed_parameters = experiment_dict['fixed-parameters']

    for objective in objective_names:

        weights = np.array(fixed_parameters['weights'])

        selected_dscluster_module = [
            m for m in dscluster_modules if objective == m.split(';')[1].strip()
        ][0] # there should only be one per objective!!
        mod_covered_sets = experiment_dict['modules'][selected_dscluster_module]['cluster-coverage-set']
        mod_covered_set = mod_covered_sets[
                str(min([int(l) for l in mod_covered_sets.keys()]))
            ]

        selected_dscluster_weighted_module = selected_dscluster_module.split(';')[0] + '; ' + objective + '-weighted;' + selected_dscluster_module.split(';')[2]
        weighted_mod_covered_sets = experiment_dict['modules'][selected_dscluster_weighted_module]['cluster-coverage-set']
        weighted_mod_covered_set = weighted_mod_covered_sets[
                str(min([int(l) for l in weighted_mod_covered_sets.keys()]))
            ]

        samples1 = np.log(weights[mod_covered_set]) / -5
        samples2 = np.log(weights[weighted_mod_covered_set]) / -5
        distribution_dict[dataset][objective] = (samples1, samples2)

# %%
# Plot results:

fig, axs = plt.subplots(len(objective_names), len(dataset_experiment_dict), figsize=(34, 12))

for i, (dataset, objective_result_dict) in enumerate(distribution_dict.items()):
    for j, (objective, (samples1, samples2)) in enumerate(objective_result_dict.items()):
        edges = np.linspace(0.0, 1.0, 31)  # 20 equal-width bins in [0, 1]
        h1, _ = np.histogram(samples1, bins=edges)
        h2, _ = np.histogram(samples2, bins=edges)

        # Convert counts to probabilities (mass per bin)
        p1 = h1 / max(h1.sum(), 1)
        p2 = h2 / max(h2.sum(), 1)
        diff = p2 - p1

        centers = 0.5 * (edges[:-1] + edges[1:])
        width = (edges[1:] - edges[:-1])
        axs[j,i].bar(
            centers,
            diff,
            width=width,
            align="center",
            alpha=0.6,
            edgecolor="black", 
            linewidth=2.0,
        )

        # Y-axis limits and ticks: choose ymin/ymax so 3 evenly spaced ticks are also 'nice'
        obj_min = np.min(diff)
        obj_max = np.max(diff)
        obj_rng = obj_max - obj_min
        pad = 0.01 * obj_rng

        ymin, ymax, yticks = nice_lim_for_3_ticks(obj_min - pad, obj_max + pad, clip=(-1.0, 1.0))
        axs[j,i].set_ylim(ymin, ymax)
        axs[j,i].set_yticks(yticks)
        axs[j,i].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

        # X-axis ticks and labels
        axs[j,i].set_xlim(0.0, 1.0)
        if j != len(objective_names) - 1:
            axs[j,i].set_xticklabels([])

        # Increase tick-label padding away from the axes to reduce overlap at origin
        axs[j,i].tick_params(axis='x', which='major', pad=10)
        axs[j,i].tick_params(axis='y', which='major', pad=10)

        # Title with dataset name
        if j == 0:
            if dataset == "kddcup":
                axs[j,i].set_title(rf"$KDDCup$")
            else:
                axs[j,i].set_title(rf"${dataset.capitalize()}$")

        # Gridlines:
        axs[j,i].grid(which='major', linestyle='-', linewidth=0.8, alpha = 0.5)
        axs[j,i].axhline(0.0, color="black", linewidth=2.0, alpha=0.7)


fig.supylabel(r"$\delta(\mathcal{W}, \mathcal{U})$", x=0.02, rotation = 0)
fig.supxlabel(r"$\textup{distance-ratio}$", y=0.05, x = 0.525)
plt.tight_layout()

plt.savefig(
    "../figures/experiments/uncertainty.pdf",
    bbox_inches='tight',
    dpi=300
)

plt.show()

# %% [markdown]
# # Confidence Experiment
# Plots each model's PEC-objective score as the rule-pool confidence-filter threshold varies (from `confidence.py`). Loading is defensive -- run it per-dataset to generate the data first.

# %%
# Load confidence-experiment data (defensive: most/all datasets may be missing).

dataset_confidence_dict = {}
for dataset in DATASETS:
    fname = f"../data/experiments/{dataset}/confidence/exp_confidence{EXP_REF}.json"
    if not os.path.exists(fname):
        print(f"[confidence] skipping {dataset}: {fname} not found")
        continue
    with open(fname) as f:
        dataset_confidence_dict[dataset] = json.load(f)

if not dataset_confidence_dict:
    print("No confidence experiment data found yet -- run confidence.py for at least one dataset.")


# %%
# Collect experiment data for the confidence-sweep line plots. ±1 SE per
# `_se`; deterministic modules get NaN, rendered as an empty band.

def _is_flat(y, rel_std_threshold=0.02):
    """Empirically detect whether a model's confidence-sweep line is ~constant.
    Can't be a static per-module lookup: even 'pool-independent' models
    (ExKMC, CN2, Decision-Tree, ...) have an objective score that varies with
    confidence here, since it's scored against that confidence level's PEC
    lambda (which changes every level), not because their own decision set
    changes.
    """
    y = np.asarray(y, dtype=float)
    y = y[~np.isnan(y)]
    if y.size < 2:
        return True
    denom = max(abs(np.mean(y)), 1e-8)
    return (np.std(y) / denom) < rel_std_threshold


confidence_line_dict = {
    dataset: {objective: {} for objective in objective_names}
    for dataset in dataset_confidence_dict
}

for dataset, conf_json in dataset_confidence_dict.items():
    conf_keys = sorted((k for k in conf_json if k != 'fixed-parameters'), key=float)
    conf_x = np.array([float(k) for k in conf_keys])
    n = conf_json['fixed-parameters']['n']

    for objective in objective_names:
        dscluster_key = f'dscluster; {objective}; ensemble'
        scaled_dscluster_key = f'{dscluster_key}; lazy-greedy'
        for module in list(comparison_modules) + [dscluster_key, scaled_dscluster_key]:
            y_vals = []
            y_errs = []
            for k in conf_keys:
                level = conf_json[k]
                raw = level.get(module, {}).get('objective', {}).get(objective, np.nan)
                y_vals.append(_scalar(raw))
                y_errs.append(_se(raw))
            # Normalize the objective by n, the size of this dataset, matching every other
            # objective-value plot in this notebook (e.g. the Max Rules bar_dict cell above),
            # which all divide by fixed_parameters['n'] -- this plot's $\bar{f}$ axis label
            # already implied that normalization even though the values weren't actually scaled.
            y_vals = np.array(y_vals, dtype=float) / n
            y_errs = np.array(y_errs, dtype=float) / n

            if module == dscluster_key:
                mod_name = 'dscluster; ensemble'
            elif module == scaled_dscluster_key:
                mod_name = 'dscluster; ensemble; lazy-greedy'
            else:
                mod_name = module
            confidence_line_dict[dataset][objective][mod_name] = {
                'x': conf_x, 'y': y_vals, 'y_err': y_errs, 'is_flat': _is_flat(y_vals),
            }

# %%
# Plot results. Shaded bands = ±1 SE (see `_se`); no band for deterministic modules.

function_name_dict = {0: 'K', 1: 'M', 2: 'P'}
# PEC drawn last so it renders on top of ScaledGreedy where lines/bands overlap.
module_order = list(comparison_modules) + ['dscluster; ensemble; lazy-greedy', 'dscluster; ensemble']

fig, axs = plt.subplots(len(objective_names), len(dataset_confidence_dict), figsize=(34, 14), squeeze=False)

for i, dataset in enumerate(dataset_confidence_dict.keys()):
    for j, objective in enumerate(objective_names):
        module_result_dict = confidence_line_dict[dataset][objective]
        finite_ys = [r['y'][~np.isnan(r['y'])] for r in module_result_dict.values()]
        y_concat = np.concatenate(finite_ys) if any(y.size for y in finite_ys) else np.array([0.0, 1.0])
        y_min, y_max, y_pad_std = y_concat.min(), y_concat.max(), y_concat.std()

        for module in module_order:
            if module not in module_result_dict:
                continue
            r = module_result_dict[module]
            mask = ~np.isnan(r['y'])
            color = _muted(color_dict.get(module, 'grey'))
            axs[j, i].plot(
                r['x'][mask], r['y'][mask],
                color=color,
                alpha=1.0,
                linestyle='dashed' if r['is_flat'] else 'solid',
                linewidth=6, markersize=10, markeredgecolor='black',
                label=title_dict.get(module, module),
            )
            err_mask = mask & ~np.isnan(r['y_err'])
            if np.any(err_mask):
                axs[j, i].fill_between(
                    r['x'][err_mask],
                    r['y'][err_mask] - r['y_err'][err_mask],
                    r['y'][err_mask] + r['y_err'][err_mask],
                    color=color, alpha=0.15, linewidth=0,
                )

        axs[j, i].grid(which='major', linestyle='-', linewidth=0.8, alpha=0.5)
        axs[j, i].set_xlim(0.0, 0.95)
        y_pad = 0.01 * y_pad_std
        y_lo, y_hi, yticks = nice_lim_for_3_ticks(y_min - y_pad, y_max + y_pad, clip=(0, np.inf))
        axs[j, i].set_ylim(y_lo, y_hi)
        axs[j, i].set_yticks(yticks)
        axs[j, i].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        if i == 0:
            axs[j, i].set_ylabel(rf"$\bar{{f}}_{{{function_name_dict[j]}}}$", rotation=0, labelpad=40, fontsize=36)
        if j == 0:
            axs[j, i].set_title(rf"${dataset.capitalize()}$")
        if j == len(objective_names) - 1:
            axs[j, i].set_xlabel(r"\textup{confidence threshold}")

fig.tight_layout()

plt.savefig(
    "../figures/experiments/confidence_sweep.pdf",
    bbox_inches='tight',
    dpi=300
)

# %%
# Create a separate legend for the confidence-sweep plot
fig, ax = plt.subplots(figsize=(12, 2))

legend_labels = dict(title_dict) | {
    'dscluster; ensemble; lazy-greedy': r'\texttt{ScaledGreedy}',
}

legend_elements = [
    mlines.Line2D(
        [], [],
        color=_muted(color_dict.get(m, 'grey')),
        alpha=1.0,
        marker=None,
        markersize=20,
        markeredgecolor='k',
        markeredgewidth=1.5,
        linewidth=10,
        linestyle='solid',
        label=legend_labels.get(m, m),
    )
    for m in module_order
]

ax.legend(handles=legend_elements, ncol=7, loc='center', frameon=False)
ax.axis('off')

plt.savefig(
    "../figures/experiments/confidence_legend.pdf",
    bbox_inches='tight',
    dpi=300
)

plt.show()

# %% [markdown]
# ### $\lambda^*$ Sweep
#
# How PEC's fitted $\lambda^*$ shifts as the confidence threshold filters the rule pool -- one value per confidence level (shared across models), so each panel is a single curve.

# %%
# Collect the PEC lambda* values as they vary with the confidence threshold
# (one scalar per confidence level/objective, shared across comparison models;
# no uncertainty band since it's a single deterministic fit, not per-trial).

confidence_lambda_dict = {
    dataset: {objective: {} for objective in objective_names}
    for dataset in dataset_confidence_dict
}

for dataset, conf_json in dataset_confidence_dict.items():
    conf_keys = sorted((k for k in conf_json if k != 'fixed-parameters'), key=float)
    conf_x = np.array([float(k) for k in conf_keys])

    for objective in objective_names:
        y_vals = []
        for k in conf_keys:
            level = conf_json[k]
            raw = level.get('lambda', {}).get(objective, np.nan)
            y_vals.append(_scalar(raw))
        y_vals = np.array(y_vals, dtype=float)

        confidence_lambda_dict[dataset][objective] = {'x': conf_x, 'y': y_vals}

# %%
# Plot results:

function_name_dict = {0: 'K', 1: 'M', 2: 'P'}
lambda_color = color_dict.get('dscluster; ensemble', 'black')

fig, axs = plt.subplots(len(objective_names), len(dataset_confidence_dict), figsize=(34, 14), squeeze=False)

for i, dataset in enumerate(dataset_confidence_dict.keys()):
    for j, objective in enumerate(objective_names):
        r = confidence_lambda_dict[dataset][objective]
        mask = ~np.isnan(r['y'])
        finite_y = r['y'][mask]
        y_min, y_max = (finite_y.min(), finite_y.max()) if finite_y.size else (0.0, 1.0)
        y_pad_std = finite_y.std() if finite_y.size else 0.0

        axs[j, i].plot(
            r['x'][mask], r['y'][mask],
            color=lambda_color,
            #marker='o',
            linewidth=8, markersize=10, markeredgecolor='black',
        )

        axs[j, i].grid(which='major', linestyle='-', linewidth=0.8, alpha=0.5)
        axs[j, i].set_xlim(0.0, 0.95)
        y_pad = 0.01 * y_pad_std
        y_lo, y_hi, yticks = nice_lim_for_3_ticks(y_min - y_pad, y_max + y_pad, clip=(0, np.inf))
        axs[j, i].set_ylim(y_lo, y_hi)
        axs[j, i].set_yticks(yticks)
        axs[j, i].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        if i == 0:
            axs[j, i].set_ylabel(rf"$\lambda^*_{{{function_name_dict[j]}}}$", rotation=0, labelpad=40, fontsize=36)
        if j == 0:
            axs[j, i].set_title(rf"${dataset.capitalize()}$")
        if j == len(objective_names) - 1:
            axs[j, i].set_xlabel(r"\textup{confidence-threshold}")

fig.tight_layout()

plt.savefig(
    "../figures/experiments/confidence_lambda.pdf",
    bbox_inches='tight',
    dpi=300
)

# %%
