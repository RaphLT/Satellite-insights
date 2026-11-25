# Retraining and other-dates results (Wazemmes, Lille)

Seed 0 for every run (deterministic). Test: the 6 hand-labelled Wazemmes squares (432 cars), 3-fold cross-validation: folds {1,2}, {3,4}, {5,6}. Each square is scored by a model that never saw it; the confidence threshold for a fold is chosen on the other folds only. A detection counts as right if within 1.5 m of a labelled car (one-to-one matching). No true negatives exist when counting objects, so the confusion matrix has three cells.

## Cross-validation on the current photo (street and car-park cars only)

| Run | Threshold | Cars found (TP) | False marks (FP) | Missed cars (FN) | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| Original model (trained on COWC aerial cars only), no retraining | 0.1/0.1/0.1 | 331 | 39 | 73 | 89.5% | 81.9% | 0.855 |
| Retrained on Englos + 4 Wazemmes squares (Wazemmes repeated 3x), 20 epochs | 0.25/0.25/0.2 | 289 | 40 | 115 | 87.8% | 71.5% | 0.789 |
| Gentle retraining: Wazemmes x3, first 10 layers frozen, learning rate 0.002, 12 epochs | 0.15/0.15/0.15 | 313 | 85 | 91 | 78.6% | 77.5% | 0.781 |
| Retrained on Englos only (4 squares, 1,181 cars) | 0.3/0.3/0.3 | 292 | 121 | 112 | 70.7% | 72.3% | 0.715 |
| Retrained on 4 Wazemmes squares only, 20 epochs | 0.4/0.4/0.35 | 279 | 63 | 125 | 81.6% | 69.1% | 0.748 |

Winner: the original model with threshold 0.10. Retraining on ~300 local cars made it forget more than it learned.

## Same model on other dates (never used for training or tuning), threshold 0.10

| Photo | TP | FP | FN | Precision | Recall |
|---|---|---|---|---|---|
| Current (cross-validated) | 331 | 39 | 73 | 89.5% | 81.9% |
| 2018 (3 squares, 76 cars) | 57 | 2 | 15 | 96.6% | 79.2% |
| 2021, market day (3 squares, 177 cars) | 116 | 35 | 44 | 76.8% | 72.5% |

2021 square 1 (market stalls on the car park): 30 false marks for 6 real cars. Caveat: 2018/2021 labels were corrected from the model's own pre-marks, so recall may be flattering.

## Occupancy by date (corrections: precision 0.895, recall 0.819, tree share of each date)

| Photo | Street spaces | Cars estimated | Street occupancy | Public car parks occupancy |
|---|---|---|---|---|
| Current | 4154 | 1381 | 33% | 48% |
| 2021 (market on the square) | 4154 | 1477 | 36% | 19% |
| 2018 (square empty after market) | 4154 | 666 | 16% | 7% |
