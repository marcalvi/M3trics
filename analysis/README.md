# Analysis Workflows

This folder contains the notebooks and helper code used to summarize M3TRICS outputs after nested cross-validation training. The analysis is split into two complementary views:

- `MM_decay_analysis.ipynb`: ablation-style missing-modality decay analysis.
- `fixed_dataset_analysis.ipynb`: fixed complete-test-dataset comparison.
- `results_analysis.py`: shared loading, aggregation, ranking, significance testing, and plotting utilities.

The notebooks expect M3TRICS result folders under `../results/<dataset_tag>/training_runs/`. Generated summaries are written under analysis-specific output folders and can be regenerated from the notebooks.

## MM Decay / Ablation Study

`MM_decay_analysis.ipynb` studies how each method behaves as missingness increases. For each method, seed, outer fold, and inner model, the notebook expands the stored predictions, computes replicate AUCs, aggregates them by missingness condition, and then builds rankings and statistical comparisons.

Distillation methods such as `Di-PAM` and `Di-MMLP` can be marked through `DISTILLATION_MODEL_NAMES`. They are still included in performance summaries and statistical comparisons, but are excluded from the intuition ranking because they require complete training data and are not directly comparable for that criterion.

### Level 0: Mean AUC Curves

![MM decay mean AUC curves](assets/readme_figures/mmcrc_ablation_level0_mean_auc_curves.png)

This figure shows the mean AUC trajectory for each method across missingness levels. It is the quickest view of performance degradation: flatter curves indicate stronger robustness as missing modalities increase, while steep drops indicate sensitivity to missingness.

### Level 1: Method Ranking Overview

![MM decay level 1 rankings](assets/readme_figures/mmcrc_ablation_level1_rankings.png)

Level 1 summarizes method-level behavior into interpretable ranking axes. The tables behind this plot include mean AUC, robustness/resilience summaries, post-adaptation envelope metrics, and intuition-oriented metrics. The ranking table is useful for identifying methods that are globally strong, not only methods that win a single missingness cell.

Main outputs:

- `replicate_auc_table.csv`: replicate-level AUCs used as the statistical unit.
- `method_condition_mean_auc_summary.csv`: mean AUC and confidence intervals per method and missingness condition.
- `method_level_metrics.csv`: method-level metrics used for global interpretation.
- `method_metric_orderings.csv`: ranking/order table for each summary metric.
- `resilience_post_adaptation_envelope.csv`: best post-adaptation behavior over train-missingness settings.
- `level1_global_friedman.csv`: global Friedman test over paired replicate AUCs.

### Level 2: Pairwise Condition Matrices

![MM decay level 2 pairwise matrices](assets/readme_figures/mmcrc_ablation_level2_pairwise_matrices.png)

Level 2 compares methods pairwise within each train/test missingness condition. Each matrix cell represents a significant winner-loser relationship after paired Wilcoxon testing and FDR correction. The lower triangle is intentionally left blank; the upper triangle contains the readable comparison area, with non-significant cells shown in gray.

Main outputs:

- `wilcoxon_significant.csv`: significant pairwise method comparisons after correction.
- Pairwise matrix figures: visual ranking of which methods significantly outperform others under each condition.

### Level 3: Significant Pair Summary

![MM decay level 3 significant pairs](assets/readme_figures/mmcrc_ablation_level3_significant_pairs.png)

Level 3 collapses the significant pairwise tests into an overview of robust winners and losers. This is useful for checking whether a method wins consistently across missingness conditions or only under a narrow scenario.

## Fixed Dataset Analysis

`fixed_dataset_analysis.ipynb` compares methods when evaluation is restricted to the complete fixed test setting: `train_missing_prop = 0.0` and `test_missing_prop = 0.0`. In the current mmCRC example, fixed outputs are formatted exactly like normal M3TRICS outputs under each method's `FIXED/seed_*` folder, so the same loader and aggregation logic can be reused.

### AUC Distribution

![Fixed dataset AUC distribution](assets/readme_figures/mmcrc_fixed_auc_violin_points.png)

The violin shows the distribution of replicate AUCs per method, and the black points are the individual replicate results. This plot is meant to show both the full spread and the exact replicate-level evidence behind the ranking.

### Fixed Pairwise Ranking

![Fixed dataset pairwise matrix](assets/readme_figures/mmcrc_fixed_pairwise_matrix.png)

The fixed pairwise matrix compares methods using paired Wilcoxon tests over shared replicate IDs. The row method is the winner and the column method is the loser. Colored cells indicate significant positive mean AUC differences after FDR correction; blank lower-triangle cells are hidden by design.

Main outputs:

- `replicate_auc_table.csv`: replicate-level AUCs for the fixed setting.
- `fixed_dataset_method_condition_summary.csv`: condition-level fixed summary.
- `fixed_dataset_method_summary.csv`: method-level mean AUC, standard deviation, replicate count, and bootstrap confidence interval.
- `fixed_dataset_global_friedman.csv`: global paired Friedman test across methods.
- `fixed_dataset_pairwise_wilcoxon.csv`: all fixed pairwise Wilcoxon comparisons.
- `fixed_dataset_pairwise_significant.csv`: significant fixed pairwise comparisons after correction.
- `fixed_dataset_method_table.csv`: final fixed ranking table combining performance and significance context.

## Interpreting Rankings

Rankings should be read together with the replicate-level statistics. A method can rank first by mean AUC while not significantly beating all alternatives, especially when replicate variance is high. The pairwise Wilcoxon tables and matrices therefore provide the inferential layer behind the ranking tables.

For decay analysis, rankings answer "which method is robust across missingness scenarios?" For fixed analysis, rankings answer "which method performs best when evaluated on the complete fixed dataset?" These are related but not identical questions.
