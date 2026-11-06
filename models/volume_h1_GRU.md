# Model card — GRU (volume, h=1)

- **training_window**: 2025-01-01 (+7 days warm-up) to 2025-04-10
- **validation_window**: day after 2025-04-10 to 2025-05-05
- **test_window**: day after 2025-05-05 to 2025-05-31
- **seed**: 42
- **n_features**: 27
- **created_utc**: 2026-11-06T15:12:47+00:00

## Hyperparameters

- seq_len: 24
- hidden_size: 64
- num_layers: 2
- dropout: 0.2
- learning_rate: 0.002
- weight_decay: 1e-05
- batch_size: 512
- max_epochs: 30
- early_stopping_patience: 4
- lr_plateau_patience: 2
- lr_plateau_factor: 0.5
- cell: gru
- epochs_trained: 30

## Metrics

| metric | validation | test |
| --- | ---: | ---: |
| rmse | 54.4723 | 55.7296 |
| mae | 37.6972 | 38.4594 |
| mape | 9.7669 | 9.7035 |
| r2 | 0.9629 | 0.9629 |
| n | 28482.0000 | 29606.0000 |

## Features

volume_z, speed_z, occupancy_mm, vc, tod_sin, tod_cos, dow_sin, dow_cos, is_weekend, rain_flag, visibility_mm, temperature_z, public_holiday, event_flag, roadwork_flag, tod_sin, tod_cos, dow_sin, dow_cos, is_weekend, public_holiday, event_flag, roadwork_flag, is_peak, vol_same_window_yday_z, vol_same_window_lastweek_z, segment_embedding(4)

Notes: trained from scratch; best epoch restored
