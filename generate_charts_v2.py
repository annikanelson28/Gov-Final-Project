"""
Generate 7 publication-quality poster charts — upgraded spec.
Palette: Claude=#2f6b4f (deep green), ChatGPT=#7da894 (sage), Grok=#9a9a9a (grey)
Font: DejaVu Serif.  Grid: subtle horizontal lines.  Titles baked in.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy import stats
from pathlib import Path

OUT  = Path('/Users/carolinekrantz/Desktop/project_analysis/poster_charts')
DATA = Path('/Users/carolinekrantz/Desktop/project_analysis/RESULTS')

# ── Palette & constants ───────────────────────────────────────────────────────
C = {'claude': '#2f6b4f', 'chat': '#7da894', 'groc': '#9a9a9a'}
LABEL = {'claude': 'Claude', 'chat': 'ChatGPT', 'groc': 'Grok'}
ORDER = ['claude', 'chat', 'groc']
TOPICS = ['Healthcare', 'Immigration', 'Middle East', 'Tax Policy']

SPINE_COL = '#aaaaaa'
GRID_COL  = '#cccccc'

# ── Global rcParams ───────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':        'DejaVu Serif',
    'font.size':          13,
    'axes.facecolor':     'white',
    'figure.facecolor':   'white',
    'axes.edgecolor':     SPINE_COL,
    'axes.linewidth':     0.8,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.spines.left':   True,
    'axes.spines.bottom': True,
    'axes.grid':          True,
    'axes.grid.axis':     'y',
    'grid.color':         GRID_COL,
    'grid.linewidth':     0.6,
    'grid.alpha':         0.3,
    'xtick.labelsize':    12,
    'ytick.labelsize':    12,
    'axes.labelsize':     13,
    'legend.frameon':     True,
    'legend.framealpha':  1.0,
    'legend.edgecolor':   SPINE_COL,
    'legend.fontsize':    11,
})

def styled_ax(ax, title, ylabel=None, xlabel=None):
    ax.set_title(title, fontsize=16, fontweight='bold', pad=10,
                 fontfamily='DejaVu Serif')
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=13)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=13)
    ax.spines['left'].set_color(SPINE_COL)
    ax.spines['bottom'].set_color(SPINE_COL)

def save(fig, name):
    fig.savefig(OUT / f'{name}.png', dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(OUT / f'{name}.svg', format='svg', bbox_inches='tight', facecolor='white')
    print(f'  ✓  {name}.png  +  {name}.svg')
    plt.close(fig)

# ── Load data ─────────────────────────────────────────────────────────────────
merged       = pd.read_csv(DATA / 'merged_results.csv')
summary_mod  = pd.read_csv(DATA / 'summary_by_model.csv')
summary_top  = pd.read_csv(DATA / 'summary_by_topic_model.csv')
stat_tests   = pd.read_csv(DATA / 'statistical_tests.csv')
balanced_res = pd.read_csv(DATA / 'balanced_subset_results.csv')

# ══════════════════════════════════════════════════════════════════════════════
# CHART 1 — "AI Summaries Rarely Preserve Balance"
#   Bar chart: % of summaries where |gap_sideA| > 0.05
# ══════════════════════════════════════════════════════════════════════════════
print('\n=== Chart 1 ===')
c1 = {}
for m in ORDER:
    df    = merged[merged.model == m]
    absg  = df.gap_sideA.abs()
    c1[m] = {'n': len(df), 'mean_abs': absg.mean(),
              'pct': (absg > 0.05).sum() / len(df) * 100}
    print(f'  {LABEL[m]}: n={c1[m]["n"]}, mean_abs_gap={c1[m]["mean_abs"]:.4f}, '
          f'pct_distorted={c1[m]["pct"]:.1f}%')

fig, ax = plt.subplots(figsize=(6.5, 5.2))
xs   = np.arange(len(ORDER))
bars = ax.bar(xs, [c1[m]['pct'] for m in ORDER],
              color=[C[m] for m in ORDER], width=0.5, zorder=3)
for bar, m in zip(bars, ORDER):
    v = c1[m]['pct']
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
            f'{v:.1f}%', ha='center', va='bottom', fontsize=13,
            fontweight='bold', fontfamily='DejaVu Serif')

ax.set_xticks(xs)
ax.set_xticklabels([LABEL[m] for m in ORDER], fontsize=12)
ax.set_ylim(0, 112)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0f}%'))
styled_ax(ax,
          title='AI Summaries Rarely Preserve Balance',
          ylabel='% of summaries with |gap| > 0.05')
fig.tight_layout()
save(fig, 'chart1_distorted_balance')

# ══════════════════════════════════════════════════════════════════════════════
# CHART 2 — "Distribution of Compression Across Articles"
#   Box plot + jitter; one column per model; mean marker; line at 0
# ══════════════════════════════════════════════════════════════════════════════
print('\n=== Chart 2 ===')
c2_data = {}
for m in ORDER:
    g = merged[merged.model == m].gap_sideA
    c2_data[m] = g.values
    print(f'  {LABEL[m]}: min={g.min():.4f}, Q1={g.quantile(.25):.4f}, '
          f'median={g.median():.4f}, mean={g.mean():.4f}, '
          f'Q3={g.quantile(.75):.4f}, max={g.max():.4f}')

rng = np.random.default_rng(42)
fig, ax = plt.subplots(figsize=(7, 5.5))

bp_pos = [1, 2, 3]
bp = ax.boxplot(
    [c2_data[m] for m in ORDER],
    positions=bp_pos,
    widths=0.38,
    patch_artist=True,
    medianprops=dict(color='white', linewidth=2.0),
    whiskerprops=dict(color=SPINE_COL, linewidth=1.0),
    capprops=dict(color=SPINE_COL, linewidth=1.0),
    flierprops=dict(marker='', markersize=0),  # hide default outlier markers
    zorder=3,
)
for patch, m in zip(bp['boxes'], ORDER):
    patch.set_facecolor(C[m])
    patch.set_alpha(0.55)
    patch.set_edgecolor(C[m])

# jitter strip
for pos, m in zip(bp_pos, ORDER):
    vals = c2_data[m]
    jitter = rng.uniform(-0.12, 0.12, size=len(vals))
    ax.scatter(pos + jitter, vals, color=C[m], s=18, alpha=0.65, zorder=4,
               edgecolors='none')

# mean marker (white diamond with colored edge)
for pos, m in zip(bp_pos, ORDER):
    ax.scatter(pos, np.mean(c2_data[m]), marker='D', s=55,
               color='white', edgecolors=C[m], linewidths=1.8, zorder=5)

ax.axhline(0, color='#555555', linewidth=1.1, linestyle='--', zorder=2)
ax.text(3.45, 0.01, 'perfect\nbalance', va='bottom', ha='right',
        fontsize=10, color='#555555', style='italic')

ax.set_xticks(bp_pos)
ax.set_xticklabels([LABEL[m] for m in ORDER], fontsize=12)
ax.set_xlim(0.5, 3.5)

# legend
mean_marker = Line2D([0], [0], marker='D', color='w', markeredgecolor='#555555',
                     markersize=7, label='Mean', linewidth=0)
med_patch   = mpatches.Patch(facecolor='white', edgecolor='#555555',
                              linewidth=1.5, label='Median (white line)')
ax.legend(handles=[mean_marker, med_patch], loc='lower right', fontsize=10)

styled_ax(ax,
          title='Distribution of Compression Across Articles',
          ylabel='Side A compression gap')
fig.tight_layout()
save(fig, 'chart2_distribution')

# ══════════════════════════════════════════════════════════════════════════════
# CHART 3 — "How Often Each Model Compressed the Conservative Side"
#   Bar: count with gap < 0; reference line at n/2; sig markers
# ══════════════════════════════════════════════════════════════════════════════
print('\n=== Chart 3 ===')
c3 = {}
for m in ORDER:
    df  = merged[merged.model == m]
    neg = int((df.gap_sideA < 0).sum())
    res = stats.binomtest(neg, len(df), 0.5, alternative='two-sided')
    c3[m] = {'neg': neg, 'n': len(df), 'p': res.pvalue, 'sig': res.pvalue < 0.05}
    print(f'  {LABEL[m]}: neg={neg}/60, p_binomial={res.pvalue:.4f} [COMPUTED], '
          f'sig={"YES" if res.pvalue<0.05 else "no"}')
print('  Reference line at 30 (n/2)')

fig, ax = plt.subplots(figsize=(6.5, 5.2))
xs   = np.arange(len(ORDER))
bars = ax.bar(xs, [c3[m]['neg'] for m in ORDER],
              color=[C[m] for m in ORDER], width=0.5, zorder=3)
ax.axhline(30, color='#555555', linewidth=1.2, linestyle='--', zorder=2,
           label='Chance (n = 30)')
ax.legend(loc='upper right', fontsize=11)

for bar, m in zip(bars, ORDER):
    neg = c3[m]['neg']
    sig = c3[m]['sig']
    label = f'{neg}{"*" if sig else ""}'
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
            label, ha='center', va='bottom', fontsize=14,
            fontweight='bold', fontfamily='DejaVu Serif')

ax.set_xticks(xs)
ax.set_xticklabels([LABEL[m] for m in ORDER], fontsize=12)
ax.set_ylim(0, 52)
ax.set_ylabel('Articles compressing conservative side', fontsize=13)
styled_ax(ax,
          title='How Often Each Model Compressed\nthe Conservative Side',
          ylabel='Articles with gap < 0 (count out of 60)')
fig.tight_layout()
save(fig, 'chart3_sign_test')

# ══════════════════════════════════════════════════════════════════════════════
# CHART 4 — "Mean Compression by Model, with 95% Confidence Intervals"
#   Bar + CI error bars; line at 0; label each bar with its mean
# ══════════════════════════════════════════════════════════════════════════════
print('\n=== Chart 4 ===')
c4 = {}
for m in ORDER:
    row    = summary_mod[summary_mod.model == m].iloc[0]
    mu, sd = row.gap_sideA_mean, row.gap_sideA_sd
    n      = int(row.gap_sideA_n)
    t_crit = stats.t.ppf(0.975, df=n - 1)
    se     = sd / np.sqrt(n)
    ci_h   = t_crit * se
    c4[m]  = {'mean': mu, 'sd': sd, 'n': n, 'se': se,
               'ci_h': ci_h, 'ci_lo': mu - ci_h, 'ci_hi': mu + ci_h}
    print(f'  {LABEL[m]}: mean={mu:.4f}, SD={sd:.4f}, n={n}, '
          f't_crit={t_crit:.4f}, SE={se:.4f}, '
          f'95%CI=[{mu-ci_h:.4f}, {mu+ci_h:.4f}]')

fig, ax = plt.subplots(figsize=(6.5, 5.5))
xs   = np.arange(len(ORDER))
means = [c4[m]['mean'] for m in ORDER]
errs  = [c4[m]['ci_h']  for m in ORDER]

bars = ax.bar(xs, means, color=[C[m] for m in ORDER],
              width=0.5, zorder=3, alpha=0.85)

ax.errorbar(xs, means, yerr=errs,
            fmt='none', ecolor='#333333', elinewidth=1.5,
            capsize=6, capthick=1.5, zorder=5)

ax.axhline(0, color='#555555', linewidth=1.1, linestyle='--', zorder=2)

for bar, m in zip(bars, ORDER):
    mu   = c4[m]['mean']
    ci_h = c4[m]['ci_h']
    # label sits above the error bar cap (or below for positive bars)
    if mu >= 0:
        y_lbl = mu + ci_h + 0.006
        va = 'bottom'
    else:
        y_lbl = mu - ci_h - 0.006
        va = 'top'
    ax.text(bar.get_x() + bar.get_width()/2, y_lbl,
            f'{mu:+.4f}', ha='center', va=va,
            fontsize=12, fontweight='bold', fontfamily='DejaVu Serif')

ax.set_xticks(xs)
ax.set_xticklabels([LABEL[m] for m in ORDER], fontsize=12)
ax.set_ylabel('Mean Side A compression gap', fontsize=13)

# CI excludes / crosses 0 annotation
for i, m in enumerate(ORDER):
    lo, hi = c4[m]['ci_lo'], c4[m]['ci_hi']
    note = 'CI excl. 0' if not (lo <= 0 <= hi) else 'CI incl. 0'
    col  = '#2f6b4f' if note == 'CI excl. 0' else '#888888'
    ax.text(i, ax.get_ylim()[0] * 0.82, note,
            ha='center', va='bottom', fontsize=9.5, color=col, style='italic')

styled_ax(ax,
          title='Mean Compression by Model\nwith 95% Confidence Intervals',
          ylabel='Mean Side A compression gap')
fig.tight_layout()
save(fig, 'chart4_mean_ci')

# ══════════════════════════════════════════════════════════════════════════════
# CHART 5 — "Compression Pattern Across Topics"
#   Grouped bar; x=4 topics; 3 bars each; line at 0; boxed legend
# ══════════════════════════════════════════════════════════════════════════════
print('\n=== Chart 5 ===')
c5 = {m: {} for m in ORDER}
for m in ORDER:
    for t in TOPICS:
        v = summary_top[(summary_top.model==m)&(summary_top.topic==t)].gap_sideA_mean.values[0]
        c5[m][t] = v
        print(f'  {t}/{LABEL[m]}: {v:.4f}')

fig, ax = plt.subplots(figsize=(10, 5.5))
width   = 0.24
x_base  = np.arange(len(TOPICS))
offsets = np.array([-1, 0, 1]) * width

for i, m in enumerate(ORDER):
    vals = [c5[m][t] for t in TOPICS]
    xs   = x_base + offsets[i]
    bars = ax.bar(xs, vals, width=width * 0.92, color=C[m],
                  label=LABEL[m], zorder=3, alpha=0.9)
    for bar, v in zip(bars, vals):
        if v >= 0:
            ax.text(bar.get_x() + bar.get_width()/2,
                    v + 0.005, f'{v:+.2f}',
                    ha='center', va='bottom', fontsize=9.5,
                    fontfamily='DejaVu Serif')
        else:
            ax.text(bar.get_x() + bar.get_width()/2,
                    v - 0.005, f'{v:+.2f}',
                    ha='center', va='top', fontsize=9.5,
                    fontfamily='DejaVu Serif')

ax.axhline(0, color='#555555', linewidth=1.0, zorder=2)
ax.set_xticks(x_base)
ax.set_xticklabels(TOPICS, fontsize=12)

leg = ax.legend(fontsize=11, loc='upper right',
                frameon=True, edgecolor=SPINE_COL, fancybox=False)
leg.get_frame().set_linewidth(0.8)

yabs = max(abs(v) for m in ORDER for v in c5[m].values())
ax.set_ylim(-yabs * 1.55, yabs * 1.55)
styled_ax(ax,
          title='Compression Pattern Across Topics',
          ylabel='Mean Side A compression gap')
fig.tight_layout()
save(fig, 'chart5_by_topic')

# ══════════════════════════════════════════════════════════════════════════════
# CHART 6 — "Effect Concentrates in Already-Tilted Articles"
#   Two bars for Claude: full corpus vs balanced subset
# ══════════════════════════════════════════════════════════════════════════════
print('\n=== Chart 6 ===')
full_mean = summary_mod[summary_mod.model=='claude'].gap_sideA_mean.values[0]
p_full    = stat_tests[
    (stat_tests.model=='claude') &
    (stat_tests.metric=='gap_sideA') &
    (stat_tests.version=='all')
].p.values[0]
sig_full  = stat_tests[
    (stat_tests.model=='claude') &
    (stat_tests.metric=='gap_sideA') &
    (stat_tests.version=='all')
].sig.values[0]

df_bal   = merged[(merged.model=='claude') & (merged.imbalance_direction=='balanced')]
bal_mean = df_bal.gap_sideA.mean()
bal_n    = len(df_bal)
p_bal    = balanced_res[
    (balanced_res.model=='claude') & (balanced_res.metric=='gap_sideA')
].p.values[0]
sig_bal  = balanced_res[
    (balanced_res.model=='claude') & (balanced_res.metric=='gap_sideA')
].sig.values[0]

print(f'  Full: mean={full_mean:.4f}, p={p_full}, sig={sig_full} [stat_tests.csv]')
print(f'  Balanced: mean={bal_mean:.4f} [COMPUTED merged_results, n={bal_n}], '
      f'p={p_bal}, sig={sig_bal} [balanced_subset_results.csv]')

labels = [f'Full corpus\n(n = 60)', f'Balanced subset\n(n = {bal_n})']
vals   = [full_mean, bal_mean]
p_vals = [p_full, p_bal]
sigs   = [sig_full, sig_bal]

fig, ax = plt.subplots(figsize=(6, 5.5))
xs = np.arange(2)
# Use lighter shade for balanced subset bar
bar_colors = [C['claude'], '#7da894']  # deep green vs sage for contrast
bars = ax.bar(xs, vals, color=bar_colors, width=0.42, zorder=3, alpha=0.9)

ax.axhline(0, color='#555555', linewidth=1.1, linestyle='--', zorder=2)

for bar, v, p, s in zip(bars, vals, p_vals, sigs):
    # mean label inside / near bar
    y_mean = v - 0.004 if v < 0 else v + 0.004
    va_mean = 'top' if v < 0 else 'bottom'
    ax.text(bar.get_x() + bar.get_width()/2, y_mean,
            f'{v:.4f}', ha='center', va=va_mean,
            fontsize=13, fontweight='bold', fontfamily='DejaVu Serif')

    # p-value annotation on opposite side of zero
    sig_str = f'p = {p:.4f}{"  *" if s == "YES" else "  n.s."}'
    y_ann = 0.004 if v < 0 else -0.004
    va_ann = 'bottom' if v < 0 else 'top'
    ax.text(bar.get_x() + bar.get_width()/2, y_ann,
            sig_str, ha='center', va=va_ann,
            fontsize=11, color='#333333', style='italic',
            fontfamily='DejaVu Serif')

ax.set_xticks(xs)
ax.set_xticklabels(labels, fontsize=12)
ymin = min(vals) * 2.2
ymax = abs(ymin) * 0.35
ax.set_ylim(ymin, ymax)
styled_ax(ax,
          title='Effect Concentrates in Already-Tilted Articles\n(Claude only)',
          ylabel='Mean Side A compression gap')
fig.tight_layout()
save(fig, 'chart6_full_vs_balanced')

# ══════════════════════════════════════════════════════════════════════════════
# CHART 7 — "Inter-Annotator Reliability (Fleiss' κ)"
#   Horizontal bars; reference at 0.61; average line; deep green bars
# ══════════════════════════════════════════════════════════════════════════════
print('\n=== Chart 7 ===')
kappa_labels = [
    "Preliminary 3-way κ",
    "Annika cross-check",
    "Anna cross-check",
    "Caroline cross-check",
    "Anna–Caroline cross-check",
]
kappa_vals = [0.73, 0.81, 0.83, 0.88, 0.80]
kappa_avg  = np.mean(kappa_vals)
for lbl, v in zip(kappa_labels, kappa_vals):
    print(f'  {lbl}: {v}')
print(f'  Average: {kappa_avg:.2f}  (given values — not from CSVs)')

fig, ax = plt.subplots(figsize=(8, 4.8))
# flip to get "preliminary" at top
ys       = np.arange(len(kappa_labels))
y_order  = ys[::-1]  # reverse so first entry is at top

ax.barh(y_order, kappa_vals, color=C['claude'], height=0.45, zorder=3, alpha=0.9)

for y, v in zip(y_order, kappa_vals):
    ax.text(v + 0.012, y, f'{v:.2f}',
            va='center', ha='left', fontsize=13, fontweight='bold',
            fontfamily='DejaVu Serif')

# substantial agreement line
ax.axvline(0.61, color='#777777', linewidth=1.5, linestyle='--', zorder=4)
ax.text(0.61, len(kappa_labels) - 0.55,
        'Substantial\nagreement\n(0.61)',
        va='top', ha='center', fontsize=10,
        color='#555555', style='italic',
        fontfamily='DejaVu Serif')

# average line
ax.axvline(kappa_avg, color='#2f2f2f', linewidth=1.0, linestyle='-', zorder=4)
ax.text(kappa_avg, -0.65, f'Avg = {kappa_avg:.2f}',
        va='top', ha='center', fontsize=10.5, color='#2f2f2f',
        fontfamily='DejaVu Serif')

ax.set_yticks(y_order)
ax.set_yticklabels(kappa_labels, fontsize=12)
ax.set_xlim(0, 1.05)
ax.set_ylim(-1.0, len(kappa_labels))
ax.set_xlabel("κ value", fontsize=13)

# disable horizontal grid for this chart (only vertical makes sense here)
ax.yaxis.grid(False)
ax.xaxis.grid(True)
ax.grid(True, axis='x', color=GRID_COL, linewidth=0.6, alpha=0.3)

styled_ax(ax, title='Inter-Annotator Reliability (Fleiss\' κ)')
fig.tight_layout()
save(fig, 'chart7_kappa')

# ══════════════════════════════════════════════════════════════════════════════
# CONSOLIDATED NUMBERS
# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '='*68)
print('CONSOLIDATED NUMBERS — ALL SEVEN CHARTS')
print('='*68)

print('\nCHART 1 — % Distorted Balance (|gap_sideA| > 0.05):')
for m in ORDER:
    d = c1[m]
    print(f'  {LABEL[m]}: n={d["n"]}, mean_abs_gap={d["mean_abs"]:.4f}, '
          f'pct_distorted={d["pct"]:.1f}%')

print('\nCHART 2 — Distribution stats (from merged_results.csv):')
for m in ORDER:
    g = merged[merged.model==m].gap_sideA
    print(f'  {LABEL[m]}: min={g.min():.4f}, Q1={g.quantile(.25):.4f}, '
          f'median={g.median():.4f}, mean={g.mean():.4f}, '
          f'Q3={g.quantile(.75):.4f}, max={g.max():.4f}')

print('\nCHART 3 — Sign test (gap_sideA < 0), binomial tests COMPUTED:')
for m in ORDER:
    d = c3[m]
    print(f'  {LABEL[m]}: {d["neg"]}/60, p={d["p"]:.4f}, '
          f'sig={"YES" if d["sig"] else "no"}')
print('  Reference line: 30 (n/2)')

print('\nCHART 4 — Mean ± 95% CI (from summary_by_model.csv):')
for m in ORDER:
    d = c4[m]
    print(f'  {LABEL[m]}: mean={d["mean"]:.4f}, SD={d["sd"]:.4f}, n={d["n"]}, '
          f'SE={d["se"]:.4f}, CI=[{d["ci_lo"]:.4f}, {d["ci_hi"]:.4f}]')

print('\nCHART 5 — Mean gap by topic (summary_by_topic_model.csv):')
for t in TOPICS:
    for m in ORDER:
        print(f'  {t}/{LABEL[m]}: {c5[m][t]:.4f}')

print('\nCHART 6 — Claude full vs balanced:')
print(f'  Full corpus:      mean={full_mean:.4f}, p={p_full}, sig={sig_full}')
print(f'  Balanced subset:  mean={bal_mean:.4f} [COMPUTED, n={bal_n}], '
      f'p={p_bal}, sig={sig_bal}')

print('\nCHART 7 — Kappa (values given, not from CSVs):')
for lbl, v in zip(kappa_labels, kappa_vals):
    print(f'  {lbl}: {v}')
print(f'  Average: {kappa_avg:.2f}')
print(f'  Reference line: 0.61 (substantial agreement threshold)')
