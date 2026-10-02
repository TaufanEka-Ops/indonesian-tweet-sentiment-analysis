# Results

Everything in this directory is **generated** — nothing is hand-written. Rebuild
with `make all`.

## `tables/`

### Data audit
| file | contents |
|---|---|
| `audit_modi.csv` | row counts, class balance, mojibake, PII, duplicates for the Modi set |
| `audit_novel_baswedan.csv` | same for the Indonesian corpus |

### Case 1 — Novel Baswedan
| file | contents |
|---|---|
| `novel_preprocessed.csv` | cleaned, de-duplicated corpus (462 rows) |
| `novel_defect_impact.csv` | mentions left behind by the original vs fixed logic |
| `novel_examples_bug.csv` | concrete before/after rows for the mention bug |
| `novel_hashtags.csv` | hashtag frequency |
| `novel_top_terms.csv` | top cleaned terms |
| `novel_entities.csv` | tweets per name/entity cluster |
| `novel_length_stats.csv` | raw vs cleaned length summary |
| `novel_dedup_effect.csv` | rows before/after de-duplication |

### Case 2 — Modi sentiment
| file | contents |
|---|---|
| `modi_data_audit.json` | every row removed and why (incl. the 5 PII rows) |
| `modi_label_distribution.csv` | class counts and shares |
| `modi_model_comparison.csv` | repeated stratified CV, all four models |
| `modi_holdout_metrics.csv` | held-out accuracy / macro-F1 / weighted-F1 / κ + bootstrap CI |
| `modi_classification_report.csv` | per-class precision / recall / F1 / support |
| `modi_confusion_matrix.csv` | held-out confusion matrix |
| `modi_top_features.csv` | top TF-IDF terms per class (Logistic Regression) |
| `modi_test_predictions.csv` | held-out predictions used by the error analysis |
| `modi_error_directions.csv` | error breakdown by (true → predicted) |
| `modi_error_examples.csv` | real misclassified tweets |
| `modi_error_terms.csv` | most frequent terms among errors |

### Shared
| file | contents |
|---|---|
| `vectorization_benchmark.csv` | `np.vectorize` vs comprehension vs pandas `.str` |

## `figures/`

| file | contents |
|---|---|
| `novel_hashtags.png`, `novel_top_terms.png`, `novel_entities.png`, `novel_token_hist.png` | case 1 descriptive plots |
| `modi_class_distribution.png` | label imbalance |
| `modi_model_comparison.png` | macro-F1 with fold standard deviation |
| `modi_confusion_matrix.png`, `modi_confusion_matrix_normalized.png` | held-out confusion matrices |
| `modi_per_class_f1.png` | per-class F1 |
| `modi_error_directions.png` | where the classifier goes wrong |

## `logs/`

Raw stdout of the numbered scripts, kept so the README's numbers can be traced
back to a run.
