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
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import matplotlib.lines as mlines
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
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
# Single source of truth for which experiment output files this notebook loads, used
# throughout instead of the previous per-section duplicated ref dicts (each of which
# just mapped every dataset to the same "_conf_50" suffix anyway).
EXP_REF = "_conf_50"
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

# Comparison-model set: Decision-Tree, ExKMC, WRA, IDS, CBA, CN2, vs. PEC
# (unweighted and weighted). Colors are assigned with a step-3 stride over the
# 8 husl hues (gcd(3,8)=1, so consecutive assignments are 135 degrees apart in
# hue rather than 45), and every model additionally gets a distinct marker
# shape / hatch pattern as a second visual channel.
color_dict = {
    'Decision-Tree': cmap(5),
    'ExKMC': cmap(1),
    #'WRA': cmap(6),
    'IDS': cmap(0),
    'CBA': cmap(3),
    'CN2': cmap(7),
    'dscluster; ensemble': cmap(6),
    # cmap(4) (teal) is otherwise unused in this palette, keeping PEC's
    # scaled-greedy ablation visually distinct from every comparison model
    # and from PEC itself -- previously plain 'darkgrey', which read as
    # ugly/washed-out next to the husl hues used everywhere else. The
    # bicriteria-plot cells apply `_muted` on top of this base color to keep
    # scaled-greedy deliberately understated there without falling back to grey.
    'dscluster; ensemble; lazy-greedy': cmap(4),
    #'dscluster; ensemble; weighted': cmap(2),
    'Reference': 'black',
}

color_dict = {
    'Decision-Tree': cmap(0),
    'ExKMC': cmap(1),
    #'WRA': cmap(6),
    'IDS': cmap(2),
    'CBA': cmap(3),
    'CN2': cmap(4),
    'dscluster; ensemble': cmap(9),
    # cmap(4) (teal) is otherwise unused in this palette, keeping PEC's
    # scaled-greedy ablation visually distinct from every comparison model
    # and from PEC itself -- previously plain 'darkgrey', which read as
    # ugly/washed-out next to the husl hues used everywhere else. The
    # bicriteria-plot cells apply `_muted` on top of this base color to keep
    # scaled-greedy deliberately understated there without falling back to grey.
    'dscluster; ensemble; lazy-greedy': cmap(5),
    #'dscluster; ensemble; weighted': cmap(2),
    'Reference': 'black',
}


# NOTE: dashed-vs-solid line style for the confidence-sweep plots (see the
# "Confidence Experiment" section) is decided per (dataset, objective, model)
# line via an empirical flatness check, not from this dict.
linestyle_dict = {
    'dscluster; ensemble' : 'solid',
    'dscluster; ensemble; weighted' : 'solid',
    'Decision-Tree': 'solid',
    'ExKMC': 'solid',
    'WRA': 'solid',
    'CBA': 'solid',
    'CN2': 'solid',
    'IDS': 'solid',
    'Reference' : 'dashed',
}

hatch_dict = {
    'Decision-Tree': '//',
    'ExKMC': '\\\\',
    'WRA': '|',
    'CBA': 'O',
    'CN2': '|',
    'IDS': '+',
    'dscluster; ensemble': '..',
    'dscluster; ensemble; weighted': 'o',
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

marker_style_dict = {
    'dscluster; ensemble' : 'D',
    #'dscluster; ensemble; weighted' : 'd',
    'Decision-Tree': 'o',
    'ExKMC': 'o',
    'WRA': 'o',
    'CBA': 'o',
    'CN2': 'o',
    'IDS': 'o',
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
# The following notebook is used to gather computed information and produce plots/tables for our paper. 
# Note that these are reliant upon having run experimenets from `experiments/`.

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
# This is a helper function for Picking nice axis limits and ticks for our plots.
# We want to ensure that the limits and ticks are "nice" numbers (e.g., multiples of 1, 2, or 5)
# and that we have exactly 3 ticks (the endpoints and the midpoint).
# This makes our plots look cleaner and more interpretable.
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
    """Stochastic modules (Decision-Tree, Exp-Tree, Shallow-Tree, IDS) store per-r
    values as {'mean', 'std', 'values'} (via `aggregate_trials`); deterministic
    modules (PEC/dscluster, ExKMC, WRA, CBA, CN2) store a bare float/int. Normalize
    to a scalar (the mean); treat None (JSON-encoded NaN) as missing too.
    """
    if isinstance(v, dict):
        return v.get('mean', np.nan)
    return v if v is not None else np.nan


def _values(v):
    """Raw per-trial values for stochastic modules ({'mean', 'std', 'values'} dicts,
    produced by `aggregate_trials`); None for deterministic modules with no repeats.
    """
    if isinstance(v, dict):
        return v.get('values', None)
    return None


def _std(v):
    """Std across trial repeats for stochastic modules ({'mean', 'std', 'values'}
    dicts); NaN for deterministic modules with no repeats (renders as no band /
    no error bar downstream).

    This notebook distinguishes two different uncertainty questions and uses a
    different statistic for each:
      - Std (this function): "how much might a single fresh fit differ?" -- the
        per-run SPREAD itself. Used as-is only in the Input Sensitivity section
        below, where that spread across randomly-drawn input pools IS the
        quantity the experiment measures -- dividing it down would understate
        exactly what that section is trying to show.
      - SE (`_se` below): "how precisely do we know this model's average
        performance?" -- used everywhere else (Max Rules, Bicriteria,
        Confidence), where the point of a plotted band is to support a
        comparison between models' average performance, and std alone
        (inflated ~sqrt(n_trials)x by within-model noise) can make a real,
        consistent gap between two lines look like noise when it isn't.
    In every section this only applies to Decision-Tree and IDS -- the only
    models whose own fitting procedure is stochastic when the rule pool is
    fixed -- except Input Sensitivity, where the rule pool itself is randomly
    resampled per repeat, so it applies to every pool-dependent algorithm.
    """
    if isinstance(v, dict):
        return v.get('std', np.nan)
    return np.nan


def _se(v):
    """Standard error of the mean (std / sqrt(n_trials)) for stochastic modules;
    NaN for deterministic modules. See `_std`'s docstring above for when to use
    which -- this is the one used for comparative claims between models' average
    performance (Max Rules, Bicriteria, Confidence sections).
    """
    if isinstance(v, dict):
        values = v.get('values', None)
        std = v.get('std', np.nan)
        if not values:
            return np.nan
        return std / np.sqrt(len(values))
    return np.nan



MIN_VISIBLE_ERR = 0.025  # floor for on-screen visibility only, in the same
                          # normalized [0,1]-ish units as x/y/z -- tuned to
                          # exceed the marker's own on-screen radius (~0.02 at
                          # s=240 in this figure's layout) so a nonzero-but-tiny
                          # SE isn't fully swallowed by its own mean marker.
                          # Keep in sync with the other bicriteria plot cell.

def _visible_err(err):
    """Floor a nonzero error up to MIN_VISIBLE_ERR for legibility; leave exact
    zeros (deterministic modules) at 0. Does not shrink real errors already
    above the floor."""
    err = np.asarray(err, dtype=float)
    return np.where(err > 0, np.maximum(err, MIN_VISIBLE_ERR), 0.0)


def _muted(color, amount=0.55):
    """Blend `color` toward white by `amount` (0 = color, 1 = white).

    Used to render PEC's scaled-greedy ablation in a deliberately understated
    tone in the bicriteria scatter plots, so it stays visually distinct from
    (rather than literal grey like before) but never competes with distorted-
    greedy PEC, the paper's main model.
    """
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

# Explicit comparison set -- previously auto-derived from every non-dscluster
# key present in the JSON, which silently pulled in Exp-Tree/Shallow-Tree/
# WRA-weighted even though only some of those had style-dict entries.
comparison_modules = {'Decision-Tree', 'ExKMC', 'IDS', 'CBA', 'CN2'}

# PEC's distorted-greedy (main model) and scaled-greedy (selection-algorithm
# ablation; internally still keyed by the JSON's "; lazy-greedy" suffix, since
# that's what max_rules.py actually writes) modules are tracked separately
# from comparison_modules, NOT merged into it: unlike the comparison models
# above, they're objective-specific (each objective gets its own
# distorted-greedy/scaled-greedy module name), so the bar-plot cells below
# must pick out the one matching the current objective rather than treating
# them as a fixed, objective-agnostic set. Merging scaled-greedy into
# comparison_modules previously caused a given dataset's coverage-cost-tuned
# scaled-greedy module to also get plotted in the coverage-mistake /
# coverage-pairwise-distance panels, since its module dict still carries every
# measurement regardless of which objective it was fit for.
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

# PEC scaled-greedy results come from a separate script (max_rules_pec_lazy.py)
# merged in via max_rules_combine.py, so a dataset may legitimately have no
# scaled-greedy module yet for one or more objectives -- flag it rather than
# silently dropping the comparison bar later.
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

baseline_module = 'KMeans'

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
# Lambda values for eacc and dataset and objective:
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
# Collect experiment data for bar plots.
#
# Uncertainty: for stochastic modules (Decision-Tree, IDS), the plotted quantity is
# derived (obj1 - lambda*(obj2 + alpha*obj3)), so its uncertainty can't just be read
# off one key -- `_module_bar_series` combines it from the raw per-trial 'values'
# lists (aligned across the reward/cost/rule-length keys, which share the same
# trial order), then reports the STANDARD ERROR of that combined quantity
# (std/sqrt(n_trials)) -- these bars support a comparison between models' average
# objective value, so SE (not raw std) is the right statistic; see `_se`'s
# docstring in the helpers cell above. Deterministic modules (PEC, ExKMC, CBA, CN2)
# get 0 (no visible error bar).

def _module_bar_series(module_dict, reward, cost, lambd, alpha):
    """Returns (obj_values, obj_err) arrays over r (in dict-iteration order).
    obj_err is the standard error of the combined objective across trials for
    stochastic modules, else 0."""
    reward_vals = list(module_dict[reward].values())
    cost_vals = list(module_dict[cost].values())
    length_vals = list(module_dict['sum-rule-length'].values())
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
            obj_values, obj_err = _module_bar_series(experiment_dict['modules'][cmod], reward, cost, lambd, alpha)
            bar_dict[dataset][objective][cmod] = (
                x[idxs], obj_values[idxs] / fixed_parameters['n'], obj_err[idxs] / fixed_parameters['n']
            )

        # DSCluster Module (distorted-greedy, the paper's main model):
        obj_values, obj_err = _module_bar_series(
            experiment_dict['modules'][selected_dscluster_module], reward, cost, lambd, alpha
        )
        bar_dict[dataset][objective][selected_dscluster_module] = (
            x[idxs], obj_values[idxs] / fixed_parameters['n'], obj_err[idxs] / fixed_parameters['n']
        )

        # DSCluster Module (scaled-greedy counterpart -- selection-algorithm
        # ablation). Optional: only present once max_rules_pec_lazy.py +
        # max_rules_combine.py have been run for this dataset/objective (see
        # the loading cell's defensive "missing_scaled" check above). Uses the
        # SAME alpha/lambda as the distorted-greedy module above, since
        # scaled-greedy was fit at that identical lambda_star (see
        # max_rules_pec_lazy.py) -- this keeps the comparison an
        # apples-to-apples test of the two PEC selection algorithms rather
        # than two independently-tuned models.
        scaled_matches = [
            m for m in scaled_dscluster_modules
            if objective == m.split(';')[1].strip() and m in experiment_dict['modules']
        ]
        if scaled_matches:
            selected_scaled_module = scaled_matches[0] # there should only be one per objective!!
            obj_values, obj_err = _module_bar_series(
                experiment_dict['modules'][selected_scaled_module], reward, cost, lambd, alpha
            )
            bar_dict[dataset][objective][selected_scaled_module] = (
                x[idxs], obj_values[idxs] / fixed_parameters['n'], obj_err[idxs] / fixed_parameters['n']
            )

# %%
# Plot results
# Error bars: +/-1 standard error (std/sqrt(n_trials=10)) across random-seed
# repeats for stochastic modules (Decision-Tree, IDS); no visible whisker for
# deterministic modules (PEC, scaled-greedy-PEC, ExKMC, CBA, CN2). SE (not raw
# std) is used because these bars support a comparison between models' average
# objective value -- see `_se`'s docstring in the helpers cell.

# Configurable lower bound for the objective-value y-axis, so subplots don't
# waste space below the region where bars actually fall.
BAR_Y_MIN = 0.0

fig,ax = plt.subplots(len(objective_names), len(dataset_experiment_dict), figsize=(34, 12))

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
                # Keep everything after the objective segment (rule-miner
                # name, plus '; lazy-greedy' when present) rather than just
                # the last ';'-separated part, so 4-part scaled-greedy module
                # names ('dscluster; <objective>; ensemble; lazy-greedy')
                # still resolve to the same 'dscluster; ensemble; lazy-greedy'
                # style-dict key used elsewhere in the notebook instead of
                # collapsing to 'dscluster; lazy-greedy'.
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
                color=color_dict.get(mod_name, 'grey'),
                alpha = 0.75,
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


#fig.supylabel(r"$\bar{f}(\mathcal{D})$", x=0.02, rotation = 0)
fig.supxlabel(r"Number of Rules, $\ell$", y=0.05, x = 0.525)
plt.tight_layout()

plt.savefig(
    "../figures/experiments/objectives.pdf",
    bbox_inches='tight',
    dpi=300
)

# %%
# Create a separate legend for the bar plot with hatch patterns
import matplotlib.patches as mpatches

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
    'dscluster; ensemble; lazy-greedy': title_dict.get('dscluster; ensemble', 'PEC') + r' (Scaled Greedy)',
}

legend_elements = []
#for mod in bar_dict[dataset][objective].keys():
for mod in module_order:
    # module_order entries are already short-form keys (no objective
    # segment), so they're valid color_dict/hatch_dict/legend_labels keys
    # as-is -- no reconstruction needed here (unlike the plotting cell above,
    # which has to strip the objective out of real JSON module names).
    mod_name = mod

    legend_elements.append(
        mpatches.Patch(
            facecolor=color_dict[mod_name],
            alpha = 0.75,
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
# Same bar-plot layout as the Max Rules bar plots above, but sourced from `max_rules_ids_alt.py` / `max_rules_combine_ids_alt.py`'s output -- IDS tuned by coordinate ascent to directly maximize the PEC objective (`ids_lambda_search_alt.py`), rather than the held-out-AUC-tuned IDS used everywhere else in this notebook. Only the k-means (`coverage-cost`) objective has been run for this experiment so far, so this shows a single row rather than one row per objective.

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
            obj_values, obj_err = _module_bar_series(experiment_dict['modules'][cmod], reward, cost, lambd, alpha)
            bar_dict_ids_alt[dataset][objective][cmod] = (
                x[idxs], obj_values[idxs] / fixed_parameters['n'], obj_err[idxs] / fixed_parameters['n']
            )

        obj_values, obj_err = _module_bar_series(
            experiment_dict['modules'][selected_dscluster_module], reward, cost, lambd, alpha
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
                experiment_dict['modules'][selected_scaled_module], reward, cost, lambd, alpha
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
    figsize=(34, 12 * len(ids_alt_objective_names) / len(objective_names)), squeeze=False,
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
                color=color_dict.get(mod_name, 'grey'),
                alpha = 0.75,
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
# ### Rule Provenance
#
# Breakdown of the (distorted-greedy) PEC module's selected rules by mining source -- Decision-Tree (DT), Random-Forest (RF), or Class-Association-Rule (CAR) -- at the smallest `max_rules.py` rule budget (`r = k`, the number of clusters, i.e. `n_rules_list[0]`). One row per objective, one column per dataset.

# %%
cmap

# %%
# Rule provenance: rule-source-counts breakdown for the (distorted-greedy) PEC module at the
# smallest rule budget (r = k, i.e. max_rules.py's n_rules_list[0]), one row per objective and
# one column per dataset.

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
                color=[source_color_dict[src] for src in source_order],
                edgecolor='black',
                alpha = 0.75
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
# 3D scatter over all three objectives (`x` = obj1/coverage, `y` = obj2/cost, `z` = obj3/rule length), each scaled to `[0, 1]`. Points come from the `lambda.py` experiment: each point is one swept value of PEC's lambda hyperparameter at a fixed rule budget, rather than a swept rule budget as in `max_rules.py`.

# %%
# Load experiment data for the bicriteria/3D scatter plots below, which read from
# `lambda.py`'s output instead of `max_rules.py`'s -- each point in these plots
# corresponds to a different swept value of PEC's lambda hyperparameter (fixed rule
# budget n_select), not a different rule budget.
# NOTE: Loading is defensive -- datasets whose lambda.py output hasn't been generated
# yet (data/experiments/<dataset>/lambda/exp<ref>.json missing) are skipped with a
# warning rather than raising.

dataset_lambda_experiment_dict = {}
for dataset in DATASETS:
    fname = "../data/experiments/" + dataset + "/lambda/exp" + EXP_REF + ".json"
    if not os.path.exists(fname):
        print(f"[lambda] skipping {dataset}: {fname} not found")
        continue
    with open(fname, 'r') as f:
        dataset_lambda_experiment_dict[dataset] = json.load(f)


# %%
# Collect experiment data for the bicriteria / 3D scatter plots:
#
# Rather than folding the cost and rule-length objectives into a single
# lambda-weighted axis, we keep all three objectives (obj1 = coverage/reward,
# obj2 = cost, obj3 = summed rule length) separate and min-max scale each to
# [0, 1]. Normalization constants are computed jointly across every module for
# a given (dataset, objective) pair, so the scaled values stay comparable
# across modules within a subplot.
#
# Each point plotted here corresponds to one swept lambda value from
# lambda.py's grid (fixed rule budget n_select, varying lambda), not a rule
# budget as in max_rules.py. lambda.py fits two PEC selection-algorithm
# variants per objective -- 'lazy-greedy' (valid across the full swept grid,
# and the secondary/reference curve in the plot below, shown under the
# "Scaled Greedy" label) and 'distorted-greedy' (the paper's main model,
# valid only for lambda >= lambda*) -- we keep both, each under its own key
# ('dscluster; ensemble' for distorted-greedy, 'dscluster; ensemble;
# lazy-greedy' for scaled-greedy) so the plotting cell can render them with
# distinct emphasis. Comparison models don't depend on lambda, so they're
# recorded once per (dataset, objective) using scaled-greedy's grid of lambda
# values (broadcast to an identical value at every point).
#
# We also carry along each dscluster variant's own lambda values (sorted
# ascending) so the plotting cell can draw direction-of-change arrows between
# consecutive-lambda points, plus the index of lambda* (the smallest lambda
# for which distorted-greedy is valid) within EACH variant's own lambda array
# -- lambda* is a grid point for both (scaled-greedy is valid across the whole
# grid, which includes it), so both get a star at that point, not just
# distorted-greedy.
#
# Alongside scatter_dict (the 3-axis version), we also build scatter_dict_2d:
# a 2-axis collapse where obj4 = obj2 + alpha * obj3 recombines cost and rule
# length back into the single weighted-cost term PEC's objective actually
# optimizes (mirroring the y2 term in the alpha-selection cell above), using
# each (dataset, objective)'s already-selected alpha (fixed-parameters['alpha'],
# keyed the same way as lambda_star). obj4 is computed from RAW obj2/obj3 (not
# the separately-normalized x/y/z above) and then min-max scaled on its own,
# since it's a different derived quantity with its own range; obj1 (x = g) is
# reused as-is since it's already comparable between the two plots.
#
# Uncertainty: only Decision-Tree/IDS are stochastic here (the rule pool is
# fixed at each lambda value; only these two models refit with fresh
# randomness per trial seed). Their STANDARD ERROR (std/sqrt(n_trials), not raw
# std -- these bars support a comparison between models' average position, see
# `_se`'s docstring in the helpers cell) is combined from the raw per-trial
# 'values' lists (aligned by lambda-grid position) using the SAME linear
# combination as the mean (obj4 = obj2 + alpha*obj3), then scaled by the same
# min-max range used for the mean -- SE scales linearly under an affine
# transform, same as std. Deterministic modules (both PEC variants, ExKMC, CBA,
# CN2) get 0.

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

        for mod, (obj1, obj2, obj3) in raw_values.items():
            x = _minmax(obj1, obj1_lo, obj1_hi)
            y = _minmax(obj2, obj2_lo, obj2_hi)
            z = _minmax(obj3, obj3_lo, obj3_hi)
            obj1_err, obj2_err, obj3_err, _ = raw_errs[mod]
            x_err = _minmax_err(obj1_err, obj1_lo, obj1_hi)
            y_err = _minmax_err(obj2_err, obj2_lo, obj2_hi)
            z_err = _minmax_err(obj3_err, obj3_lo, obj3_hi)
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
# All three axes (`x` = obj1, `y` = obj2, `z` = obj3) are shown directly, each scaled to `[0, 1]`. Kept simple -- one fixed viewing angle, matching color/marker encoding used elsewhere -- to stay legible in print.

# %%
# Plot results (Option 2): 3D scatter over (obj1, obj2, obj3).
#
# Reading exact axis values off a 3D scatter is hard -- once a point is lifted
# off the floor, the eye has no fixed reference for its position. Each point
# gets a thin drop-line straight down to the z=0 floor plus a faint marker at
# that projection: the marker's (x, y) position is then readable directly
# against the floor grid, and the stem length conveys z.
#
# Two PEC variants are shown per (dataset, objective) panel, each with its
# own color (color_dict) so they're distinguishable even where their
# objective values nearly overlap:
#   - 'distorted-greedy' is the paper's main model: full-strength diamond
#     markers, opaque, with a floor drop-line like the comparison models.
#   - 'scaled-greedy' is the secondary reference: small, faint, edge-less
#     circle markers so it never visually competes with distorted-greedy --
#     understating it (smaller, fainter, a different marker shape, AND its
#     own color) is what keeps the two legible as distinct series where they
#     nearly overlap.
# Both change with lambda (comparison models don't), so both get thin arrows
# chained between consecutive lambda values, smallest to largest -- solid for
# distorted-greedy, dashed for scaled-greedy. lambda* (the smallest lambda for
# which distorted-greedy is valid) is a grid point for both variants, so both
# get a star there, each in its own color.
#
# Uncertainty: Decision-Tree/IDS get thin per-axis error segments (x/y/z_err
# from the collection cell above) through each point, drawn under the marker;
# all other modules' err arrays are 0 (deterministic), so no segment is drawn
# for them.

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers the '3d' projection)

# mplot3d's default zorder is recomputed every draw from each artist's camera
# distance, which ignores the static zorder= kwargs passed below (that's why
# scaled-greedy's points/lines could render on top of distorted-greedy's even
# though they're given a lower zorder). ax.computed_zorder = False switches
# an Axes3D to respect the zorder values we actually pass, like a normal 2D
# Axes -- set per-axis below, right after each ax is created.

# Beta's subscript names the objective's cost term, not just "beta".
beta_subscript_dict = {
    'coverage-cost': 'K',
    'coverage-mistake': 'M',
    'coverage-pairwise-distance': 'P',
}

# Fixed length (in the shared [0,1]-normalized x/y/z units) for the solid
# arrowhead on the direction-of-change arrows below. Axes3D.quiver's own
# arrow_length_ratio scales the head with that arrow's shaft length, so the
# long first segment (lambda=0 to the next grid point) got a huge head;
# drawing the head as its own short, fixed-length quiver call instead keeps
# it a constant size regardless of segment length.
ARROW_HEAD_LEN = 0.2

def _draw_lambda_arrows(ax, x, y, z, lam, color, *, dashed, alpha, linewidth, zorder):
    """Direction-of-change arrows between consecutive (sorted-by-lambda)
    points: a plain line for the shaft (dashed or solid per `dashed`) and a
    fixed-size, always-solid arrowhead at its tip.
    """
    if lam is None or len(lam) <= 1:
        return
    order = np.argsort(lam)
    xs, ys, zs = x[order], y[order], z[order]
    for k in range(len(xs) - 1):
        x0, y0, z0 = xs[k], ys[k], zs[k]
        dxk, dyk, dzk = xs[k + 1] - x0, ys[k + 1] - y0, zs[k + 1] - z0
        seg_len = np.sqrt(dxk ** 2 + dyk ** 2 + dzk ** 2)
        if seg_len == 0:
            continue
        head_len = min(ARROW_HEAD_LEN, 0.5 * seg_len)
        ux, uy, uz = dxk / seg_len, dyk / seg_len, dzk / seg_len
        xh, yh, zh = x0 + dxk - ux * head_len, y0 + dyk - uy * head_len, z0 + dzk - uz * head_len

        ax.plot(
            [x0, xh], [y0, yh], [z0, zh],
            color=color, alpha=alpha, linewidth=linewidth,
            linestyle='dashed' if dashed else 'solid', zorder=zorder,
        )
        ax.quiver(
            xh, yh, zh, ux * head_len, uy * head_len, uz * head_len,
            color=color, alpha=alpha, linewidth=linewidth,
            arrow_length_ratio=0.6, normalize=False, zorder=zorder,
        )

# Pastel (pre-lightened) tab10 colors, used only in this 3D scatter -- relying on alpha for a
# light/diluted look (as the 2D bar/line/scatter plots elsewhere in this notebook do) doesn't
# render consistently in mplot3d, since translucent markers there don't composite against a
# fixed background the way ordinary 2D alpha does (compositing order instead depends on each
# artist's per-draw camera distance). Pre-lightening the colors themselves sidesteps that.
pastel_color_dict = {k: _muted(v, amount=0.35) for k, v in color_dict.items()}

# One figure per objective (rather than one giant figure with an
# objective-per-row grid), each laid out as 3 columns x 2 rows over datasets
# -- gives every dataset panel enough size to stay legible, and lets each
# objective's figure be saved to (and read from) its own output file.
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

        for module, pts in module_result_dict.items():
            x, y, z = pts['x'], pts['y'], pts['z']
            x_err, y_err, z_err = pts['x_err'], pts['y_err'], pts['z_err']
            lam = pts['lam']
            lambda_star_idx = pts['lambda_star_idx']
            is_scaled = module.endswith('lazy-greedy')
            # Further muted (blended toward white) on top of the already-pastel base for
            # scaled-greedy -- keeps it visually distinct from every other series via its own
            # hue, while staying deliberately understated so it never competes with
            # distorted-greedy PEC, the paper's main model.
            color = _muted(pastel_color_dict.get(module, 'grey')) if is_scaled else pastel_color_dict.get(module, 'grey')

            # Floor drop-line + projection marker (drawn first, low
            # zorder, so the actual data points render on top of them).
            for xi, yi, zi in zip(x, y, z):
                ax.plot([xi, xi], [yi, yi], [0, zi], color=color, alpha=0.5, linewidth=1.0, zorder=1)
                ax.scatter(xi, yi, 0, color=color, s=15, alpha=0.5, edgecolor='none', zorder=1)

            # Per-axis error segments (only nonzero for Decision-Tree/IDS).
            x_err_disp, y_err_disp, z_err_disp = _visible_err(x_err), _visible_err(y_err), _visible_err(z_err)
            for xi, yi, zi, xe, ye, ze in zip(x, y, z, x_err_disp, y_err_disp, z_err_disp):
            #for xi, yi, zi, xe, ye, ze in zip(x, y, z, x_err, y_err, z_err):
                if xe > 0:
                    ax.plot([xi - xe, xi + xe], [yi, yi], [zi, zi], color=color, alpha=0.5, linewidth=1.5, zorder=2)
                if ye > 0:
                    ax.plot([xi, xi], [yi - ye, yi + ye], [zi, zi], color=color, alpha=0.5, linewidth=1.5, zorder=2)
                if ze > 0:
                    ax.plot([xi, xi], [yi, yi], [zi - ze, zi + ze], color=color, alpha=0.5, linewidth=1.5, zorder=2)

            if is_scaled:
                # Faint, edge-less markers -- deliberately understated so this
                # secondary curve doesn't compete with distorted-greedy. Now that the
                # color itself is pre-lightened (pastel_color_dict above), alpha here
                # only needs to add a little extra fade relative to the main model
                # below, not do the heavy lifting of lightening the color.
                ax.scatter(
                    x, y, z,
                    color=color, marker='o', s=80, alpha=0.75,
                    edgecolor='black', depthshade=False, zorder=2,
                )
                _draw_lambda_arrows(
                    ax, x, y, z, lam, color,
                    dashed=True, alpha=0.7, linewidth=2.0, zorder=1,
                )

                if lambda_star_idx is not None:
                    ax.scatter(
                        x[lambda_star_idx], y[lambda_star_idx], z[lambda_star_idx],
                        marker='*', s=600, color=color, edgecolor='black',
                        alpha=0.9, linewidth=1.0, depthshade=False, zorder=5,
                    )
            else:
                # Floor drop-line + projection marker (drawn first, low
                # zorder, so the actual data points render on top of them).
                #for xi, yi, zi in zip(x, y, z):
                #    ax.plot([xi, xi], [yi, yi], [0, zi], color=color, alpha=0.3, linewidth=1.0, zorder=1)
                #    ax.scatter(xi, yi, 0, color=color, s=15, alpha=0.3, edgecolor='none', zorder=1)

                # Opaque now that the color itself is pre-lightened (pastel_color_dict
                # above) -- alpha < 1 here previously was doing double duty trying (and,
                # per the 3D-specific compositing quirk noted above, failing) to also
                # dilute the color.
                ax.scatter(
                    x, y, z,
                    label=module,
                    color=color,
                    marker='o',#marker_style_dict.get(module, 'o'),
                    s=240,
                    edgecolor='black',
                    alpha=1.0,
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
                        linewidth=1.2, depthshade=False, zorder=6, alpha=1.0
                    )

        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_zlim(-0.05, 1.05)
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        ax.set_zticks([0, 0.5, 1])
        # The y=0 and z=0 tick labels each land right on top of a neighboring
        # axis's tick label at this view angle (y=0 against x=1's corner,
        # z=0 against y=1's corner) -- drop those two labels rather than let
        # them collide; the floor/wall gridlines still make the origin
        # readable.
        ax.set_yticklabels(['', '0.5', '1.0'])
        ax.set_zticklabels(['', '0.5', '1.0'])
        ax.tick_params(axis='both', which='major', labelsize=16, pad=0)
        # Zoom the cube slightly within its own bounding box to claw back
        # some of the whitespace 3D subplots otherwise reserve for rotation.
        ax.set_box_aspect(None, zoom=1.12)

        ax.view_init(elev=22, azim=-60)

        # Title with dataset name -- every subplot gets one now (each panel
        # is a distinct dataset within a single-objective figure, rather than
        # sharing a title with the rest of its column as in the old
        # objective-per-row layout).
        if dataset == "kddcup":
            ax.set_title(rf"$KDDCup$", pad=6)
        else:
            ax.set_title(rf"${dataset.capitalize()}$", pad=6, fontsize=28)

        # Axis titles on every panel, same as x and z below -- consistent
        # rather than shown only on an edge, since each panel shares the same
        # view angle and axis meaning throughout the grid.
        ax.set_ylabel(rf"$\bar{{\beta}}_{{{beta_subscript_dict.get(objective, '')}}}$", labelpad=6, fontsize=22)
        ax.set_xlabel(r"$\bar{g}$", labelpad=2, fontsize=22)
        # set_rotate_label(False) + rotation=0 keeps the z-label horizontal/upright instead of
        # mplot3d's default of rotating it to run vertically alongside the z-axis -- with
        # text.usetex on, that default rotation also made the label's true render extent read
        # wrong at savefig time, which was clipping it off the right edge of the rightmost
        # column's panels.
        ax.zaxis.set_rotate_label(False)
        ax.set_zlabel(r"$\bar{s}$", labelpad=14, fontsize=22, rotation=0)

        # Light, uncluttered panes/gridlines so points stay legible in print
        ax.xaxis.pane.set_alpha(0.05)
        ax.yaxis.pane.set_alpha(0.05)
        ax.zaxis.pane.set_alpha(0.05)
        ax.grid(True, linestyle=':', linewidth=0.6, alpha=0.5)

    # Hide any trailing unused panels (only if n_datasets isn't a multiple of
    # n_cols -- with the expected 6 datasets this loop body never runs).
    for idx in range(n_datasets, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axs[row, col].set_axis_off()

    # 3D subplots report an inflated layout bbox (extra margin reserved for
    # rotation), which makes plt.tight_layout() leave excess whitespace and
    # doesn't resolve corner tick-label crowding -- pack the grid manually
    # instead, slightly overlapping the reserved (not visible-content) margins.
    # right is pulled in further (0.98 -> 0.90) and wspace opened back up
    # (-0.05 -> 0.15) to leave the rightmost column's now-horizontal z-label
    # room to sit fully inside the figure instead of running off its edge.
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
    'dscluster; ensemble; lazy-greedy': title_dict.get('dscluster; ensemble', 'PEC') + r' (Scaled Greedy)',
}

legend_elements = []
for cmod in module_order:
    is_scaled = cmod.endswith('lazy-greedy')
    # Scaled-greedy has its own color_dict entry (distinct from distorted-
    # greedy's) and is drawn as a faint, edge-less circle with a dashed
    # connector -- matching the scatter plot cells' muted treatment.
    color = _muted(color_dict[cmod]) if is_scaled else color_dict[cmod]
    legend_elements.append(
        mlines.Line2D(
            [], [],
            color=color,
            marker='o',# if is_scaled else marker_style_dict.get(cmod, 'o'),
            markersize=30,
            markeredgecolor='k',
            markeredgewidth=1.5,
            alpha=0.55 if is_scaled else 1.0,
            linestyle='--' if is_scaled else 'None',
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
# Same data as the 3D scatter above, but with the cost and rule-length axes collapsed into a single term: `obj4 = obj2 + alpha * obj3`, i.e. PEC's actual weighted-cost objective (obj2 = cost, obj3 = summed rule length, `alpha` = the value selected for that dataset/objective in the alpha-selection step). `obj1` (`x = g`) is unchanged. Distorted-greedy vs. scaled-greedy are distinguished the same way as the 3D version (see the legend above), and this collapse is what makes the two clearly separable on a single axis where the 3D view often has them nearly overlapping.

# %%
# Plot results: 2D scatter over (obj1, obj4) -- the collapsed-cost version of
# the bicriteria plot above.
#
# Same visual encoding as the 3D scatter: 'distorted-greedy' (PEC's main
# model) is a bold, opaque diamond with a solid direction-of-change arrow
# (smallest to largest lambda) and a star at lambda*; 'scaled-greedy' is a
# faint, edge-less circle with a dashed arrow and its own color_dict color
# (distinct from distorted-greedy's) so the two stay legible even where they
# nearly overlap. lambda* is a grid point for both, so both get a star there.
# Comparison models don't depend on lambda, so they're drawn as plain points
# with no arrows.
#
# Uncertainty: Decision-Tree/IDS get errorbar whiskers (xerr/yerr from the
# collection cell above); all other modules' err arrays are 0 (deterministic),
# so `ax.errorbar` renders no visible whisker for them -- safe to call
# unconditionally.

function_name_dict = {0: 'K', 1: 'M', 2: 'P'}

# Sized per-panel (rather than a fixed figsize) so panel aspect ratio -- and
# with it, the quiver arrows below -- stays reasonable regardless of how many
# datasets are actually loaded.
fig, axs = plt.subplots(
    len(objective_names), len(scatter_dict_2d),
    figsize=(6.5 * len(scatter_dict_2d), 4.5 * len(objective_names)), squeeze=False,
)

for i, (dataset, objective_result_dict) in enumerate(scatter_dict_2d.items()):
    for j, (objective, module_result_dict) in enumerate(objective_result_dict.items()):
        ax = axs[j, i]
        ax.grid(True, linestyle=':', linewidth=0.8, alpha=0.9)

        for module, pts in module_result_dict.items():
            x, y4 = pts['x'], pts['y4']
            x_err, y4_err = pts['x_err'], pts['y4_err']
            lam = pts['lam']
            lambda_star_idx = pts['lambda_star_idx']
            is_scaled = module.endswith('lazy-greedy')
            # Muted, same as the 3D scatter cell and the shared bicriteria
            # legend, so scaled-greedy reads as understated here too instead
            # of showing up in full-strength color.
            color = _muted(color_dict.get(module, 'grey')) if is_scaled else color_dict.get(module, 'grey')

            if is_scaled:
                #ax.errorbar(
                #    x, y4, xerr=x_err, yerr=y4_err, fmt='none',
                #    ecolor=color, alpha=0.3, capsize=3, zorder=1,
                #)
                ax.errorbar(
                    x, y4, xerr=_visible_err(x_err), yerr=_visible_err(y4_err), fmt='none',
                    ecolor=color, alpha=0.3, capsize=3, zorder=1,
                )
                
                ax.scatter(
                    x, y4,
                    color=color, marker='o', s=100, alpha=0.35,
                    edgecolor='black', zorder=2,
                )
                if len(lam) > 1:
                    order = np.argsort(lam)
                    xs, ys = x[order], y4[order]
                    dx, dy = np.diff(xs), np.diff(ys)
                    arrows = ax.quiver(
                        xs[:-1], ys[:-1], dx, dy,
                        angles='xy', scale_units='xy', scale=1,
                        # 'dots' keeps the arrow shaft width an absolute size
                        # (independent of axes aspect ratio), so it doesn't
                        # balloon if a panel ends up short and wide.
                        units='dots', width=4.0,
                        color=color, alpha=0.45,
                        headwidth=3, headlength=4, zorder=2,
                    )
                    arrows.set_linestyle('dashed')

                if lambda_star_idx is not None:
                    ax.scatter(
                        x[lambda_star_idx], y4[lambda_star_idx],
                        marker='*', s=600, color=color, edgecolor='black',
                        alpha=0.7, linewidth=1.0, zorder=5,
                    )
            else:
                #ax.errorbar(
                #    x, y4, xerr=x_err, yerr=y4_err, fmt='none',
                #    ecolor=color, alpha=0.4, capsize=3, zorder=2,
                #)
                ax.errorbar(
                    x, y4, xerr=_visible_err(x_err), yerr=_visible_err(y4_err), fmt='none',
                    ecolor=color, alpha=0.3, capsize=3, zorder=1,
                )
                ax.scatter(
                    x, y4,
                    label=module,
                    color=color,
                    marker='o',#marker_style_dict.get(module, 'o'),
                    s=240,
                    edgecolor='black',
                    alpha=0.9,
                    zorder=3,
                )

                if lam is not None and len(lam) > 1:
                    order = np.argsort(lam)
                    xs, ys = x[order], y4[order]
                    dx, dy = np.diff(xs), np.diff(ys)
                    ax.quiver(
                        xs[:-1], ys[:-1], dx, dy,
                        angles='xy', scale_units='xy', scale=1,
                        units='dots', width=4.8,
                        color=color, alpha=0.8,
                        headwidth=4, headlength=5, zorder=4,
                    )

                if lambda_star_idx is not None:
                    ax.scatter(
                        x[lambda_star_idx], y4[lambda_star_idx],
                        marker='*', s=600, color=color, edgecolor='black',
                        linewidth=1.2, zorder=6,
                    )

        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        ax.tick_params(axis='both', which='major', labelsize=16)

        if j == 0:
            if dataset == "kddcup":
                ax.set_title(rf"$KDDCup$", pad=10)
            else:
                ax.set_title(rf"${dataset.capitalize()}$", pad=10)

        if j == len(objective_names) - 1:
            ax.set_xlabel(r"$g$", fontsize=22)
        ax.set_ylabel(rf"$h_{{{function_name_dict[j]}}}$", fontsize=22)

plt.tight_layout()

plt.savefig(
    "../figures/experiments/bicriteria_2d.pdf",
    bbox_inches='tight',
    dpi=300
)

# %% [markdown]
# ### Weighted Average Rule Length

# %%
# Collect the weighted average lengths for each dataset and objective.
# Uncertainty: for Decision-Tree/IDS, 'weighted-avg-length' at the smallest
# rule budget is an aggregate_trials dict; stored here as (mean, SE) tuples so
# the table cell below can render "mean +/- SE" (standard error, not raw std --
# this table supports a comparison between models' average rule length, see
# `_se`'s docstring in the helpers cell). Deterministic modules (PEC, ExKMC,
# CBA, CN2) get SE=NaN, rendered as a plain number.
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


# %%
# Weighted rule length table. "mean +/- SE" (standard error across n_trials=10
# random-seed repeats) for Decision-Tree/IDS; a plain number for deterministic
# modules (PEC, ExKMC, CBA, CN2), which have no repeats.
def _fmt_weighted_avg(cell):
    mean, se = cell
    if np.isnan(se):
        return f"{mean:.3f}"
    return f"{mean:.3f} ± {se:.3f}"

pd.DataFrame({
    col: {row: _fmt_weighted_avg(val) for row, val in rows.items()}
    for col, rows in weighted_avg_dict.items()
})

# %%

# %% [markdown]
# ### Uncertainty

# %%
# Finally, we collect the distributions of covered weights for each dataset and objective.
# Note: this section only compares weighted vs. unweighted PEC, both deterministic
# single-fit models (no Decision-Tree/IDS here) -- no repeated-trial uncertainty
# applies, so this plot is unaffected by the notebook's uncertainty changes elsewhere.

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
        mod_covered_weights = weights[mod_covered_set]

        selected_dscluster_weighted_module = selected_dscluster_module.split(';')[0] + '; ' + objective + '-weighted;' + selected_dscluster_module.split(';')[2]
        weighted_mod_covered_sets = experiment_dict['modules'][selected_dscluster_weighted_module]['cluster-coverage-set']
        weighted_mod_covered_set = weighted_mod_covered_sets[
                str(min([int(l) for l in weighted_mod_covered_sets.keys()]))
            ]
        weighted_mod_covered_weights = weights[weighted_mod_covered_set]

        samples1 = np.log(weights[mod_covered_set]) / -5
        samples2 = np.log(weights[weighted_mod_covered_set]) / -5
        distribution_dict[dataset][objective] = (samples1, samples2)

# %%
# Plot results:

fig, axs = plt.subplots(len(objective_names), len(dataset_experiment_dict), figsize=(34, 12))

for i, (dataset, objective_result_dict) in enumerate(distribution_dict.items()):
    for j, (objective, (samples1, samples2)) in enumerate(objective_result_dict.items()):
        # Plot histograms
        #all_vals = np.concatenate([samples1, samples2]) if (samples1.size and samples2.size) else (samples1 if samples1.size else samples2)
        #edges = np.histogram_bin_edges(all_vals, bins=20)
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

        # Y-label with objective
        #if i == 0:
        #    axs[j,i].set_ylabel(
        #        rf"{objective_name_dict[objective]}"
        #    )

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
# Plots results from the `confidence.py` experiment: each comparison model's PEC-objective score as the rule-pool confidence-filter threshold varies. Loading is defensive -- run `experiments/<dataset>/confidence.py` to generate `data/experiments/<dataset>/confidence/exp_confidence.json` for a dataset before it will appear here.
#

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
# Collect experiment data for the confidence-sweep line plots.
#
# Uncertainty: for Decision-Tree/IDS, 'objective' at each confidence level is
# already an aggregate_trials dict ({'mean','std','values'}) -- no derived
# combination needed here (unlike the max_rules bar plot), just read '_se' off
# the same raw value. SE (not raw std) is used because this plot's bands
# support a comparison between models' average objective score -- see `_se`'s
# docstring in the helpers cell. Deterministic modules (PEC, scaled-greedy-PEC,
# ExKMC, CBA, CN2) get NaN, which fill_between (in the plot cell below) renders
# as an empty band.

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
# Plot results:
# Shaded bands = +/-1 standard error (std/sqrt(n_trials=10)) across random-seed
# repeats (Decision-Tree, IDS); no band for deterministic modules (PEC,
# scaled-greedy-PEC, ExKMC, CBA, CN2). SE (not raw std) is used because these
# bands support a comparison between models' average objective score -- see
# `_se`'s docstring in the helpers cell.

function_name_dict = {0: 'K', 1: 'M', 2: 'P'}
# Drawn in this order so PEC (plotted last) renders on top of PEC Scaled
# Greedy where their lines/bands sit right on top of each other -- PEC is
# the paper's main model and should win visibility ties.
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
            color = color_dict.get(module, 'grey')
            axs[j, i].plot(
                r['x'][mask], r['y'][mask],
                color=color,
                alpha = 0.75,
                #marker=marker_style_dict.get(module, 'o'),
                linestyle='dashed' if r['is_flat'] else 'solid',
                linewidth=5, markersize=10, markeredgecolor='black',
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
    'dscluster; ensemble; lazy-greedy': title_dict.get('dscluster; ensemble', 'PEC') + r' (Scaled Greedy)',
}

legend_elements = [
    mlines.Line2D(
        [], [],
        color=color_dict.get(m, 'grey'),
        alpha=0.75,
        marker=None,#marker_style_dict.get(m, 'o'),
        markersize=20,
        markeredgecolor='k',
        markeredgewidth=1.5,
        linewidth=10,
        linestyle='solid',
        label=legend_labels.get(m, m),
    )
    for m in module_order
]
#legend_elements += [
#    mlines.Line2D([], [], color='black', linestyle='solid', linewidth=3, label='Varies with confidence'),
#    mlines.Line2D([], [], color='black', linestyle='dashed', linewidth=3, label='Constant'),
#]

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
# Shows how the PEC objective's fitted $\lambda^*$ (the coverage/cost trade-off multiplier chosen by distorted greedy) shifts as the confidence threshold filters the rule pool. Unlike the objective-score plots above, $\lambda^*$ is fit once per confidence level and shared across every comparison model scored at that level, so each panel below shows a single curve rather than one line per model.

# %%
# Collect the PEC lambda* values as they vary with the confidence threshold.
# lambda* is saved per confidence level in conf_json[k]['lambda'][objective]
# (see the "PEC" block of run_confidence_level in confidence.py) -- one value
# per (confidence level, objective), not per comparison model.
#
# No uncertainty band here: in confidence.py, lambda* is a plain scalar per
# confidence level (a single PEC fit against a fixed, deterministically-
# filtered rule pool) -- not a per-trial quantity, unlike its counterpart in
# the Input Sensitivity section below, where the rule pool itself is randomly
# resampled per repeat.

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
