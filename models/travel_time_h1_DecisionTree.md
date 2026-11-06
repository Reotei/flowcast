# Model card — DecisionTree (travel_time, h=1)

- **training_window**: 2025-01-01 (+7 days warm-up) to 2025-04-10
- **validation_window**: day after 2025-04-10 to 2025-05-05
- **test_window**: day after 2025-05-05 to 2025-05-31
- **seed**: 42
- **n_features**: 75
- **created_utc**: 2026-11-06T15:06:40+00:00

## Hyperparameters

- max_depth: 8
- min_samples_leaf: 20

## Metrics

| metric | validation | test |
| --- | ---: | ---: |
| rmse | 1.2412 | 1.2448 |
| mae | 0.5221 | 0.5360 |
| mape | 11.5785 | 11.6509 |
| r2 | 0.7644 | 0.7719 |
| n | 29246.0000 | 30412.0000 |

## Features

vol_t0, vol_t1, vol_t2, vol_same_window_yday, vol_same_window_lastweek, vol_diff_t0_t1, vol_roll_mean_4, vol_roll_std_4, speed_roll_mean_4, vol_roll_mean_8, vol_roll_std_8, speed_roll_mean_8, speed_t0, speed_t1, occupancy_t0, travel_time_t0, travel_time_t1, signal_timing_t0, share_2w_t0, share_car_t0, share_lcv_t0, share_hcv_t0, road_capacity, vc_t0, headroom_t0, congestion_t0, accidents_past_week, accidents_past_30d, tod_sin, tod_cos, dow_sin, dow_cos, is_weekend, is_peak, public_holiday, event_flag, roadwork_flag, holiday_x_peak, event_x_peak, event_within_1day, temperature_t0, rainfall_t0, visibility_t0, rain_flag_t0, low_visibility_t0, weather_cloudy_t0, weather_rain_t0, weather_fog_t0, latitude, longitude, seg_NL-001, seg_NL-002, seg_NL-003, seg_NL-004, seg_NL-005, seg_NL-006, seg_NL-007, seg_NL-008, seg_NL-009, seg_NL-010, seg_NL-011, seg_NL-012, seg_NL-013, seg_NL-014, seg_NL-015, seg_NL-016, seg_NL-017, seg_NL-018, seg_NL-019, seg_NL-020, seg_NL-021, seg_NL-022, seg_NL-023, seg_NL-024, seg_NL-025

Notes: 
