# Results Tables

## Table 1 - Data

| Item | Train | Validation | Test |
|---|---:|---:|---:|
| Windows | 5891 | 1461 | 2947 |
| Volunteers | 17 | 4 | 9 |

| Channel | Mean (training) | Standard deviation (training) |
|---|---:|---:|
| body_acc_x | -0.00070240 | 0.19895644 |
| body_acc_y | -0.00026787 | 0.12264342 |
| body_acc_z | -0.00032154 | 0.10905767 |
| body_gyro_x | -0.00075924 | 0.40160665 |
| body_gyro_y | -0.00076530 | 0.38940650 |
| body_gyro_z | -0.00029240 | 0.26522923 |
| total_acc_x | 0.80000895 | 0.41951731 |
| total_acc_y | 0.02070463 | 0.38315091 |
| total_acc_z | 0.07646599 | 0.37676328 |

## Table 2 - Network and Training

| Item | Value |
|---|---:|
| Trainable parameters | 46,982 |
| Initial loss | 1.766285 |
| Best epoch | 17 |
| Training loss at best epoch | 0.107875 |
| Validation loss at best epoch | 0.084635 |
| Validation accuracy at best epoch | 0.9795 |
| Training time | 1320.22 seconds (CPU) |

Parameter calculation:
`2 × (4×64×9 + 4×64×64 + 4×64) + (64×128 + 64 + 64) + (128×6 + 6) = 46,982`

## Table 3 - Test Results

| Activity | Per-class accuracy | Mean max alpha |
|---|---:|---:|
| WALKING | 0.9536 | 0.326211 |
| WALKING UPSTAIRS | 0.9682 | 0.289627 |
| WALKING DOWNSTAIRS | 0.9643 | 0.322751 |
| SITTING | 0.8452 | 0.020222 |
| STANDING | 0.8064 | 0.023390 |
| LAYING | 0.9497 | 0.013696 |

Test accuracy: **0.9121**

| True \ Predicted | WALKING | UPSTAIRS | DOWNSTAIRS | SITTING | STANDING | LAYING |
|---|---:|---:|---:|---:|---:|---:|
| WALKING | 473 | 12 | 11 | 0 | 0 | 0 |
| UPSTAIRS | 12 | 456 | 3 | 0 | 0 | 0 |
| DOWNSTAIRS | 0 | 15 | 405 | 0 | 0 | 0 |
| SITTING | 0 | 1 | 0 | 415 | 75 | 0 |
| STANDING | 0 | 0 | 0 | 103 | 429 | 0 |
| LAYING | 0 | 0 | 0 | 0 | 27 | 510 |