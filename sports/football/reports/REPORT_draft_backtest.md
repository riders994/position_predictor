# Draft backtest — does the model's board win real past drafts?

_Every past season with a preseason FantasyFootballCalculator board and a leak-safe projection is re-drafted from every slot. The room drafts off that season's real board with noise calibrated to its observed draft-slot spread; my team drafts by one policy; **every team is scored on actual weekly points** under the row's lineup rule (`managed`: each week start the active players with the best season points per game; `bestball`: the best lineup after the fact). 10 room draws per season, shared by every policy, so comparisons are paired._

_League: 1QB / 2RB / 2WR / 1TE / 1FLEX (RB/WR/TE), 14 drafted players. FFC publishes 12-team boards only, so the 10-team room drafts in 12-team market order — an assumption. Standard errors are across seasons (draws inside a season share one outcome)._

_Lookahead ran on slots [1, 5, 10] with 4 draws and 8 planning draws per candidate (it forks every candidate at each of its first 8 picks, ~100x a board draft)._

## Policies

- `adp` — ADP drafter (market order, fills open slots)
- `vorp_board` — VORP board, whole projection pool
- `vorp_board_ranked` — VORP board, market-ranked players only
- `vorp_board_ranked_static` — VORP board, market-ranked only, bench tail re-ranked as shipped
- `vorp_board_ranked_depth` — VORP board, market-ranked only, insurance bench
- `market_window` — Market picks position + round; model picks the player within one round
- `lookahead_ranked` — Lookahead (opportunity cost), market-ranked only

## Paired gains — actual points per week

| lineup | scoring | teams | comparison | gain / week | ± se | seasons won |
|---|---|---|---|---|---|---|
| managed | half_ppr | 10 | vorp_board − adp | -9.61 | 3.90 | 0/5 |
| managed | half_ppr | 12 | vorp_board − adp | -10.52 | 3.48 | 0/5 |
| managed | ppr | 10 | vorp_board − adp | -12.34 | 3.96 | 2/11 |
| managed | ppr | 12 | vorp_board − adp | -15.53 | 4.06 | 2/11 |
| managed | half_ppr | 10 | vorp_board_ranked − vorp_board | 0.38 | 1.08 | 3/5 |
| managed | half_ppr | 12 | vorp_board_ranked − vorp_board | 2.58 | 0.89 | 5/5 |
| managed | ppr | 10 | vorp_board_ranked − vorp_board | 6.21 | 2.56 | 10/11 |
| managed | ppr | 12 | vorp_board_ranked − vorp_board | 7.81 | 3.03 | 11/11 |
| managed | half_ppr | 10 | vorp_board_ranked − adp | -9.23 | 3.99 | 0/5 |
| managed | half_ppr | 12 | vorp_board_ranked − adp | -7.94 | 3.63 | 0/5 |
| managed | ppr | 10 | vorp_board_ranked − adp | -6.13 | 2.56 | 3/11 |
| managed | ppr | 12 | vorp_board_ranked − adp | -7.73 | 2.93 | 2/11 |
| managed | half_ppr | 10 | vorp_board_ranked_static − vorp_board_ranked | -1.38 | 0.42 | 1/5 |
| managed | half_ppr | 12 | vorp_board_ranked_static − vorp_board_ranked | 0.08 | 0.81 | 2/5 |
| managed | ppr | 10 | vorp_board_ranked_static − vorp_board_ranked | 1.15 | 0.56 | 10/11 |
| managed | ppr | 12 | vorp_board_ranked_static − vorp_board_ranked | 0.85 | 0.56 | 9/11 |
| managed | half_ppr | 10 | vorp_board_ranked_depth − vorp_board_ranked_static | 1.71 | 0.44 | 5/5 |
| managed | half_ppr | 12 | vorp_board_ranked_depth − vorp_board_ranked_static | 0.38 | 0.98 | 4/5 |
| managed | ppr | 10 | vorp_board_ranked_depth − vorp_board_ranked_static | 0.30 | 0.68 | 5/11 |
| managed | ppr | 12 | vorp_board_ranked_depth − vorp_board_ranked_static | 0.56 | 0.36 | 7/11 |
| managed | half_ppr | 10 | vorp_board_ranked_depth − vorp_board_ranked | 0.33 | 0.45 | 4/5 |
| managed | half_ppr | 12 | vorp_board_ranked_depth − vorp_board_ranked | 0.46 | 0.50 | 3/5 |
| managed | ppr | 10 | vorp_board_ranked_depth − vorp_board_ranked | 1.46 | 0.69 | 9/11 |
| managed | ppr | 12 | vorp_board_ranked_depth − vorp_board_ranked | 1.41 | 0.53 | 9/11 |
| managed | half_ppr | 10 | vorp_board_ranked_depth − adp | -8.90 | 4.24 | 0/5 |
| managed | half_ppr | 12 | vorp_board_ranked_depth − adp | -7.48 | 3.96 | 0/5 |
| managed | ppr | 10 | vorp_board_ranked_depth − adp | -4.67 | 2.60 | 4/11 |
| managed | ppr | 12 | vorp_board_ranked_depth − adp | -6.32 | 2.80 | 3/11 |
| managed | half_ppr | 10 | market_window − adp | -0.85 | 1.05 | 2/5 |
| managed | half_ppr | 12 | market_window − adp | -1.17 | 1.41 | 2/5 |
| managed | ppr | 10 | market_window − adp | 0.33 | 1.08 | 6/11 |
| managed | ppr | 12 | market_window − adp | -0.01 | 1.19 | 6/11 |
| managed | half_ppr | 10 | market_window − vorp_board_ranked | 8.38 | 3.83 | 5/5 |
| managed | half_ppr | 12 | market_window − vorp_board_ranked | 6.77 | 3.16 | 4/5 |
| managed | ppr | 10 | market_window − vorp_board_ranked | 6.46 | 2.14 | 8/11 |
| managed | ppr | 12 | market_window − vorp_board_ranked | 7.72 | 2.65 | 9/11 |
| managed | half_ppr | 10 | lookahead_ranked − vorp_board_ranked | 1.08 | 2.32 | 3/5 |
| managed | half_ppr | 12 | lookahead_ranked − vorp_board_ranked | 0.48 | 1.49 | 4/5 |
| managed | ppr | 10 | lookahead_ranked − vorp_board_ranked | -0.21 | 1.26 | 5/11 |
| managed | ppr | 12 | lookahead_ranked − vorp_board_ranked | 0.57 | 1.04 | 6/11 |
| managed | half_ppr | 10 | lookahead_ranked − adp | -8.66 | 3.35 | 0/5 |
| managed | half_ppr | 12 | lookahead_ranked − adp | -9.44 | 3.88 | 0/5 |
| managed | ppr | 10 | lookahead_ranked − adp | -4.97 | 2.30 | 2/11 |
| managed | ppr | 12 | lookahead_ranked − adp | -5.41 | 2.22 | 2/11 |

## Policy outcomes

_`vs league` is my points per week minus the league's mean team. `win %` = finished first in actual points; `proj lineup` is the model's own projected starting lineup — the gap between it and actual points is the model's optimism._

| lineup | scoring | teams | policy | pts / week | vs league | win % | top-3 % | mean finish | proj lineup |
|---|---|---|---|---|---|---|---|---|---|
| managed | half_ppr | 10 | adp | 93.11 | 4.49 | 14.80 | 47.00 | 4.20 | 87.95 |
| managed | half_ppr | 10 | lookahead_ranked | 83.99 | -4.55 | 1.70 | 6.70 | 6.98 | 99.63 |
| managed | half_ppr | 10 | market_window | 92.26 | 3.71 | 18.60 | 43.00 | 4.38 | 93.65 |
| managed | half_ppr | 10 | vorp_board | 83.50 | -4.56 | 4.60 | 15.80 | 6.59 | 97.36 |
| managed | half_ppr | 10 | vorp_board_ranked | 83.88 | -4.40 | 6.00 | 19.20 | 6.50 | 97.28 |
| managed | half_ppr | 10 | vorp_board_ranked_depth | 84.21 | -4.06 | 6.60 | 18.60 | 6.35 | 97.28 |
| managed | half_ppr | 10 | vorp_board_ranked_static | 82.50 | -5.59 | 4.20 | 13.80 | 6.82 | 97.28 |
| managed | half_ppr | 12 | adp | 89.48 | 4.08 | 13.30 | 39.00 | 4.94 | 84.97 |
| managed | half_ppr | 12 | lookahead_ranked | 80.51 | -4.66 | 1.70 | 10.00 | 7.72 | 97.72 |
| managed | half_ppr | 12 | market_window | 88.31 | 2.90 | 11.30 | 33.80 | 5.35 | 91.77 |
| managed | half_ppr | 12 | vorp_board | 78.96 | -6.10 | 2.50 | 8.80 | 8.44 | 95.61 |
| managed | half_ppr | 12 | vorp_board_ranked | 81.54 | -3.71 | 4.20 | 14.00 | 7.64 | 95.57 |
| managed | half_ppr | 12 | vorp_board_ranked_depth | 82.00 | -3.28 | 4.70 | 14.50 | 7.41 | 95.57 |
| managed | half_ppr | 12 | vorp_board_ranked_static | 81.62 | -3.63 | 3.70 | 11.80 | 7.76 | 95.57 |
| managed | ppr | 10 | adp | 103.64 | 4.93 | 17.70 | 48.50 | 4.13 | 99.73 |
| managed | ppr | 10 | lookahead_ranked | 97.72 | -0.99 | 6.80 | 25.00 | 5.89 | 112.09 |
| managed | ppr | 10 | market_window | 103.97 | 5.27 | 19.50 | 48.40 | 4.09 | 105.74 |
| managed | ppr | 10 | vorp_board | 91.30 | -6.62 | 5.40 | 17.90 | 6.81 | 112.17 |
| managed | ppr | 10 | vorp_board_ranked | 97.51 | -0.91 | 7.90 | 26.10 | 5.77 | 110.54 |
| managed | ppr | 10 | vorp_board_ranked_depth | 98.96 | 0.39 | 9.40 | 31.70 | 5.31 | 110.54 |
| managed | ppr | 10 | vorp_board_ranked_static | 98.66 | 0.16 | 9.70 | 29.60 | 5.42 | 110.54 |
| managed | ppr | 12 | adp | 99.84 | 4.34 | 15.20 | 40.80 | 5.02 | 96.87 |
| managed | ppr | 12 | lookahead_ranked | 93.21 | -2.23 | 4.50 | 12.90 | 7.21 | 109.98 |
| managed | ppr | 12 | market_window | 99.84 | 4.31 | 15.40 | 39.70 | 5.00 | 103.63 |
| managed | ppr | 12 | vorp_board | 84.31 | -10.46 | 3.00 | 8.30 | 9.04 | 110.47 |
| managed | ppr | 12 | vorp_board_ranked | 92.11 | -3.21 | 5.40 | 16.20 | 7.35 | 108.61 |
| managed | ppr | 12 | vorp_board_ranked_depth | 93.53 | -1.91 | 5.20 | 17.50 | 6.93 | 108.61 |
| managed | ppr | 12 | vorp_board_ranked_static | 92.97 | -2.44 | 5.80 | 19.30 | 7.04 | 108.61 |

## By season

| lineup | scoring | teams | season | vorp_board − adp | vorp_board_ranked − vorp_board | vorp_board_ranked − adp | vorp_board_ranked_static − vorp_board_ranked | vorp_board_ranked_depth − vorp_board_ranked_static | vorp_board_ranked_depth − vorp_board_ranked | vorp_board_ranked_depth − adp | market_window − adp | market_window − vorp_board_ranked | lookahead_ranked − vorp_board_ranked | lookahead_ranked − adp |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| managed | half_ppr | 10 | 2018 | -22.73 | 2.84 | -19.90 | -2.13 | 2.45 | 0.32 | -19.58 | -3.52 | 16.38 | 2.04 | -19.46 |
| managed | half_ppr | 10 | 2019 | -3.17 | 1.91 | -1.27 | -2.16 | 2.18 | 0.02 | -1.25 | -0.11 | 1.16 | -1.51 | -2.06 |
| managed | half_ppr | 10 | 2023 | -3.80 | 0.86 | -2.94 | 0.07 | 0.66 | 0.73 | -2.21 | 1.64 | 4.58 | -5.10 | -10.01 |
| managed | half_ppr | 10 | 2024 | -3.83 | -0.29 | -4.13 | -0.97 | 2.61 | 1.64 | -2.48 | -3.14 | 0.99 | 1.03 | -1.03 |
| managed | half_ppr | 10 | 2025 | -14.51 | -3.40 | -17.91 | -1.71 | 0.63 | -1.08 | -18.99 | 0.89 | 18.80 | 8.93 | -10.76 |
| managed | half_ppr | 12 | 2018 | -23.89 | 2.30 | -21.58 | 2.04 | -3.19 | -1.15 | -22.73 | -3.86 | 17.73 | 2.72 | -23.77 |
| managed | half_ppr | 12 | 2019 | -8.19 | 3.04 | -5.15 | -0.01 | 0.71 | 0.71 | -4.45 | -1.12 | 4.03 | 1.98 | -5.43 |
| managed | half_ppr | 12 | 2023 | -6.08 | 5.57 | -0.50 | -2.32 | 2.22 | -0.10 | -0.60 | 2.67 | 3.18 | -5.32 | -4.27 |
| managed | half_ppr | 12 | 2024 | -4.28 | 0.09 | -4.20 | 1.63 | 0.03 | 1.67 | -2.53 | -4.67 | -0.47 | 0.69 | -2.40 |
| managed | half_ppr | 12 | 2025 | -10.17 | 1.91 | -8.26 | -0.95 | 2.11 | 1.16 | -7.10 | 1.13 | 9.39 | 2.34 | -11.34 |
| managed | ppr | 10 | 2013 | -12.27 | 6.63 | -5.64 | 0.20 | 6.09 | 6.29 | 0.65 | -3.93 | 1.71 | -0.96 | -3.94 |
| managed | ppr | 10 | 2014 | -37.11 | 28.78 | -8.33 | 3.53 | -1.16 | 2.37 | -5.96 | 4.54 | 12.87 | -2.69 | -10.48 |
| managed | ppr | 10 | 2015 | -24.24 | 12.31 | -11.93 | 0.01 | 2.02 | 2.03 | -9.90 | 0.38 | 12.31 | -0.51 | -9.82 |
| managed | ppr | 10 | 2016 | -15.60 | 9.54 | -6.07 | 0.34 | -0.15 | 0.20 | -5.87 | -3.20 | 2.87 | 1.69 | -4.03 |
| managed | ppr | 10 | 2017 | 5.49 | 1.76 | 7.25 | -1.39 | 1.26 | -0.12 | 7.13 | 7.01 | -0.24 | 5.43 | 10.33 |
| managed | ppr | 10 | 2018 | -26.64 | 3.40 | -23.24 | 0.10 | -1.22 | -1.12 | -24.36 | -1.81 | 21.43 | 4.14 | -14.89 |
| managed | ppr | 10 | 2019 | 5.20 | 0.47 | 5.67 | 0.06 | 0.06 | 0.12 | 5.79 | 3.63 | -2.04 | 1.01 | 5.49 |
| managed | ppr | 10 | 2022 | -9.87 | 1.60 | -8.27 | 2.74 | -1.98 | 0.76 | -7.51 | -1.86 | 6.41 | 3.91 | -2.06 |
| managed | ppr | 10 | 2023 | -2.93 | 3.62 | 0.69 | 1.69 | -1.32 | 0.37 | 1.06 | 0.37 | -0.32 | -6.38 | -4.67 |
| managed | ppr | 10 | 2024 | -10.59 | 0.00 | -10.59 | 4.85 | 0.10 | 4.95 | -5.64 | -3.24 | 7.35 | -0.27 | -7.42 |
| managed | ppr | 10 | 2025 | -7.15 | 0.17 | -6.98 | 0.57 | -0.40 | 0.16 | -6.82 | 1.78 | 8.76 | -7.72 | -13.21 |
| managed | ppr | 12 | 2013 | -10.81 | 9.13 | -1.69 | 1.27 | 0.87 | 2.13 | 0.44 | -4.69 | -3.00 | -5.73 | -3.23 |
| managed | ppr | 12 | 2014 | -41.93 | 34.48 | -7.45 | 4.61 | 0.45 | 5.05 | -2.39 | 3.38 | 10.82 | -2.05 | -7.43 |
| managed | ppr | 12 | 2015 | -28.57 | 16.70 | -11.87 | 0.18 | -0.91 | -0.74 | -12.61 | 2.06 | 13.93 | 0.94 | -13.65 |
| managed | ppr | 12 | 2016 | -10.62 | 7.44 | -3.19 | 0.60 | -0.46 | 0.14 | -3.04 | -2.82 | 0.36 | -1.80 | -4.50 |
| managed | ppr | 12 | 2017 | 0.67 | 6.05 | 6.72 | 0.12 | -0.44 | -0.32 | 6.41 | 4.24 | -2.48 | 2.75 | 6.85 |
| managed | ppr | 12 | 2018 | -32.48 | 1.14 | -31.35 | 0.27 | 2.44 | 2.71 | -28.64 | -5.26 | 26.09 | 6.39 | -20.75 |
| managed | ppr | 12 | 2019 | 0.87 | 0.07 | 0.94 | -0.16 | 0.31 | 0.15 | 1.09 | 4.42 | 3.48 | 2.64 | 3.21 |
| managed | ppr | 12 | 2022 | -15.01 | 2.54 | -12.47 | 0.68 | 1.07 | 1.75 | -10.71 | 0.04 | 12.50 | 4.56 | -5.99 |
| managed | ppr | 12 | 2023 | -14.60 | 4.78 | -9.82 | 0.44 | 0.31 | 0.75 | -9.07 | 4.18 | 14.00 | -0.07 | -4.78 |
| managed | ppr | 12 | 2024 | -8.07 | 1.02 | -7.06 | 3.65 | -0.32 | 3.34 | -3.72 | -5.21 | 1.84 | 0.80 | -4.09 |
| managed | ppr | 12 | 2025 | -10.30 | 2.51 | -7.79 | -2.29 | 2.84 | 0.55 | -7.23 | -0.41 | 7.38 | -2.16 | -5.18 |

## By draft slot (mean over seasons and draws)

| lineup | scoring | teams | slot | vorp_board − adp | vorp_board_ranked − vorp_board | vorp_board_ranked − adp | vorp_board_ranked_static − vorp_board_ranked | vorp_board_ranked_depth − vorp_board_ranked_static | vorp_board_ranked_depth − vorp_board_ranked | vorp_board_ranked_depth − adp | market_window − adp | market_window − vorp_board_ranked | lookahead_ranked − vorp_board_ranked | lookahead_ranked − adp |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| managed | half_ppr | 10 | 1 | -8.30 | 0.35 | -7.94 | -1.20 | 1.77 | 0.57 | -7.37 | -3.89 | 4.05 | -0.86 | -10.67 |
| managed | half_ppr | 10 | 2 | -6.80 | 1.06 | -5.74 | -1.77 | 1.89 | 0.12 | -5.62 | 1.28 | 7.02 | — | — |
| managed | half_ppr | 10 | 3 | -9.97 | 0.74 | -9.24 | -0.83 | 1.21 | 0.38 | -8.85 | -0.83 | 8.41 | — | — |
| managed | half_ppr | 10 | 4 | -12.72 | 0.11 | -12.62 | 0.20 | 0.23 | 0.43 | -12.19 | -1.83 | 10.78 | — | — |
| managed | half_ppr | 10 | 5 | -10.73 | 0.18 | -10.55 | -0.79 | 1.47 | 0.68 | -9.87 | 0.19 | 10.74 | 2.03 | -9.54 |
| managed | half_ppr | 10 | 6 | -11.61 | -0.28 | -11.89 | -0.76 | 1.44 | 0.69 | -11.20 | -1.73 | 10.16 | — | — |
| managed | half_ppr | 10 | 7 | -10.09 | 0.40 | -9.69 | -1.57 | 2.08 | 0.50 | -9.19 | -0.14 | 9.55 | — | — |
| managed | half_ppr | 10 | 8 | -9.25 | 0.16 | -9.08 | -2.05 | 2.23 | 0.18 | -8.90 | -0.62 | 8.46 | — | — |
| managed | half_ppr | 10 | 9 | -8.78 | 0.59 | -8.19 | -2.54 | 2.72 | 0.18 | -8.01 | -0.67 | 7.53 | — | — |
| managed | half_ppr | 10 | 10 | -7.85 | 0.51 | -7.34 | -2.49 | 2.01 | -0.48 | -7.82 | -0.24 | 7.10 | 2.06 | -5.79 |
| managed | half_ppr | 12 | 1 | -10.81 | 2.70 | -8.11 | 0.22 | 0.24 | 0.47 | -7.65 | -4.66 | 3.45 | -0.34 | -9.60 |
| managed | half_ppr | 12 | 2 | -8.19 | 2.59 | -5.60 | 0.57 | 0.70 | 1.27 | -4.33 | 0.12 | 5.72 | — | — |
| managed | half_ppr | 12 | 3 | -9.82 | 3.42 | -6.40 | 0.05 | 0.12 | 0.17 | -6.23 | -0.32 | 6.08 | — | — |
| managed | half_ppr | 12 | 4 | -8.31 | 2.39 | -5.92 | 0.07 | 0.38 | 0.44 | -5.48 | 0.65 | 6.57 | — | — |
| managed | half_ppr | 12 | 5 | -12.70 | 2.30 | -10.39 | 0.31 | 1.35 | 1.66 | -8.73 | -1.17 | 9.23 | -0.08 | -12.27 |
| managed | half_ppr | 12 | 6 | -10.80 | 2.63 | -8.17 | 0.08 | 0.64 | 0.73 | -7.44 | -0.62 | 7.55 | — | — |
| managed | half_ppr | 12 | 7 | -11.45 | 2.40 | -9.05 | 0.28 | 0.15 | 0.43 | -8.62 | -1.27 | 7.78 | — | — |
| managed | half_ppr | 12 | 8 | -9.11 | 2.46 | -6.65 | 0.26 | 0.32 | 0.57 | -6.08 | 0.41 | 7.06 | — | — |
| managed | half_ppr | 12 | 9 | -8.59 | 2.18 | -6.41 | -0.33 | 0.20 | -0.13 | -6.54 | -0.72 | 5.69 | — | — |
| managed | half_ppr | 12 | 10 | -11.20 | 2.18 | -9.02 | -0.32 | 0.76 | 0.44 | -8.58 | -2.24 | 6.78 | 1.87 | -6.47 |
| managed | half_ppr | 12 | 11 | -11.98 | 2.56 | -9.42 | -0.07 | -0.42 | -0.48 | -9.90 | -1.70 | 7.73 | — | — |
| managed | half_ppr | 12 | 12 | -13.30 | 3.17 | -10.13 | -0.17 | 0.08 | -0.09 | -10.22 | -2.50 | 7.62 | — | — |
| managed | ppr | 10 | 1 | -11.85 | 8.02 | -3.83 | 0.88 | 0.09 | 0.97 | -2.86 | 0.73 | 4.56 | 0.82 | -2.30 |
| managed | ppr | 10 | 2 | -15.27 | 7.42 | -7.85 | 1.34 | 0.51 | 1.85 | -6.00 | -1.26 | 6.59 | — | — |
| managed | ppr | 10 | 3 | -15.17 | 7.55 | -7.63 | 1.54 | -0.03 | 1.51 | -6.12 | -1.07 | 6.56 | — | — |
| managed | ppr | 10 | 4 | -12.75 | 5.83 | -6.92 | 1.30 | 0.11 | 1.41 | -5.51 | 0.50 | 7.42 | — | — |
| managed | ppr | 10 | 5 | -11.59 | 5.90 | -5.68 | 1.26 | 0.20 | 1.45 | -4.23 | 0.21 | 5.89 | 0.06 | -6.11 |
| managed | ppr | 10 | 6 | -11.73 | 5.42 | -6.31 | 1.44 | -0.17 | 1.27 | -5.04 | 1.56 | 7.86 | — | — |
| managed | ppr | 10 | 7 | -12.71 | 5.80 | -6.92 | 0.90 | 0.19 | 1.09 | -5.83 | 0.21 | 7.12 | — | — |
| managed | ppr | 10 | 8 | -10.62 | 5.29 | -5.33 | 0.87 | 0.75 | 1.62 | -3.71 | 1.14 | 6.47 | — | — |
| managed | ppr | 10 | 9 | -11.10 | 5.03 | -6.06 | 1.00 | 0.94 | 1.95 | -4.12 | 1.84 | 7.90 | — | — |
| managed | ppr | 10 | 10 | -10.59 | 5.82 | -4.76 | 1.03 | 0.42 | 1.45 | -3.32 | -0.50 | 4.26 | -1.52 | -6.51 |
| managed | ppr | 12 | 1 | -13.51 | 7.71 | -5.80 | 1.24 | 0.93 | 2.17 | -3.63 | 1.37 | 7.17 | 1.86 | -3.59 |
| managed | ppr | 12 | 2 | -15.64 | 7.24 | -8.40 | 1.10 | 1.32 | 2.42 | -5.99 | 0.23 | 8.63 | — | — |
| managed | ppr | 12 | 3 | -16.05 | 7.27 | -8.78 | 1.17 | 0.67 | 1.84 | -6.94 | -1.80 | 6.98 | — | — |
| managed | ppr | 12 | 4 | -18.02 | 6.83 | -11.19 | 1.02 | 1.16 | 2.17 | -9.01 | -1.41 | 9.78 | — | — |
| managed | ppr | 12 | 5 | -16.82 | 6.92 | -9.89 | 1.07 | 0.78 | 1.86 | -8.04 | -0.60 | 9.30 | 0.09 | -8.10 |
| managed | ppr | 12 | 6 | -17.07 | 7.92 | -9.15 | 0.70 | 0.90 | 1.60 | -7.55 | -0.54 | 8.62 | — | — |
| managed | ppr | 12 | 7 | -15.31 | 7.03 | -8.28 | 0.78 | 1.32 | 2.11 | -6.18 | -0.53 | 7.75 | — | — |
| managed | ppr | 12 | 8 | -15.08 | 7.82 | -7.26 | 0.77 | 0.41 | 1.18 | -6.08 | 0.81 | 8.07 | — | — |
| managed | ppr | 12 | 9 | -14.69 | 8.17 | -6.51 | 0.76 | -0.05 | 0.71 | -5.80 | -0.50 | 6.01 | — | — |
| managed | ppr | 12 | 10 | -13.75 | 8.74 | -5.01 | 0.75 | -0.39 | 0.35 | -4.65 | 1.58 | 6.58 | -0.25 | -4.55 |
| managed | ppr | 12 | 11 | -15.40 | 8.95 | -6.45 | 0.50 | -0.10 | 0.40 | -6.05 | 0.73 | 7.18 | — | — |
| managed | ppr | 12 | 12 | -15.06 | 9.06 | -6.00 | 0.38 | -0.24 | 0.14 | -5.86 | 0.59 | 6.59 | — | — |

## Seasons simulated

_`ranked` = offensive players on the FFC board; `scored share` = share of the market's first `picks` players the model can project (the rest are mostly rookies); `noise` = calibrated room noise scale._

| scoring | season | teams | ranked | picks | pool | scored share | noise | skipped |
|---|---|---|---|---|---|---|---|---|
| half_ppr | 2018 | 10 | 189 | 140 | 250 | 0.89 | 1.25 |  |
| half_ppr | 2019 | 10 | 169 | 140 | 243 | 0.88 | 1.25 |  |
| half_ppr | 2022 | 10 | 117 | 140 | 228 | 0.90 |  | board ranks 117 players for 140 picks |
| half_ppr | 2023 | 10 | 175 | 140 | 246 | 0.92 | 1.50 |  |
| half_ppr | 2024 | 10 | 159 | 140 | 235 | 0.91 | 1.25 |  |
| half_ppr | 2025 | 10 | 144 | 140 | 231 | 0.87 | 1.50 |  |
| half_ppr | 2018 | 12 | 189 | 168 | 287 | 0.89 | 1.25 |  |
| half_ppr | 2019 | 12 | 169 | 168 | 279 | 0.88 | 1.25 |  |
| half_ppr | 2022 | 12 | 117 | 168 | 267 | 0.90 |  | board ranks 117 players for 168 picks |
| half_ppr | 2023 | 12 | 175 | 168 | 275 | 0.91 | 1.50 |  |
| half_ppr | 2024 | 12 | 159 | 168 | 272 | 0.89 | 1.25 |  |
| half_ppr | 2025 | 12 | 144 | 168 | 270 | 0.88 | 1.50 |  |
| ppr | 2012 | 10 | 92 | 140 | 220 | 0.93 |  | board ranks 92 players for 140 picks |
| ppr | 2013 | 10 | 162 | 140 | 236 | 0.93 | 1.50 |  |
| ppr | 2014 | 10 | 165 | 140 | 247 | 0.91 | 1.50 |  |
| ppr | 2015 | 10 | 172 | 140 | 247 | 0.87 | 1.50 |  |
| ppr | 2016 | 10 | 163 | 140 | 237 | 0.90 | 1.25 |  |
| ppr | 2017 | 10 | 164 | 140 | 239 | 0.90 | 1.25 |  |
| ppr | 2018 | 10 | 169 | 140 | 241 | 0.90 | 1.25 |  |
| ppr | 2019 | 10 | 170 | 140 | 242 | 0.88 | 1.25 |  |
| ppr | 2022 | 10 | 146 | 140 | 236 | 0.87 | 1.25 |  |
| ppr | 2023 | 10 | 177 | 140 | 247 | 0.92 | 1.25 |  |
| ppr | 2024 | 10 | 178 | 140 | 242 | 0.91 | 1.25 |  |
| ppr | 2025 | 10 | 206 | 140 | 257 | 0.89 | 1.50 |  |
| ppr | 2012 | 12 | 92 | 168 | 259 | 0.93 |  | board ranks 92 players for 168 picks |
| ppr | 2013 | 12 | 162 | 168 | 271 | 0.91 | 1.50 |  |
| ppr | 2014 | 12 | 165 | 168 | 279 | 0.89 | 1.25 |  |
| ppr | 2015 | 12 | 172 | 168 | 283 | 0.87 | 1.50 |  |
| ppr | 2016 | 12 | 163 | 168 | 273 | 0.90 | 1.25 |  |
| ppr | 2017 | 12 | 164 | 168 | 275 | 0.88 | 1.25 |  |
| ppr | 2018 | 12 | 169 | 168 | 279 | 0.89 | 1.25 |  |
| ppr | 2019 | 12 | 170 | 168 | 280 | 0.88 | 1.25 |  |
| ppr | 2022 | 12 | 146 | 168 | 273 | 0.87 | 1.25 |  |
| ppr | 2023 | 12 | 177 | 168 | 277 | 0.89 | 1.25 |  |
| ppr | 2024 | 12 | 178 | 168 | 278 | 0.89 | 1.25 |  |
| ppr | 2025 | 12 | 206 | 168 | 293 | 0.86 | 1.50 |  |
