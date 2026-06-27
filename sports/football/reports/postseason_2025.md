# Postseason report — 2025

_Generated 2026-06-23._ Model preseason projection (trained on labels < 2025) vs preseason **ECR** and **ADP**, graded against actual 2025 finish (PPR PPG). Ranking metrics are computed on each source's covered ∩ actual-eligible players.

## QB

| source | n | Spearman | P@6 | P@12 | P@24 | P@36 | mean rank err |
|---|---|---|---|---|---|---|---|
| **model** | 63 | 0.708 | 0.17 | 0.58 | 0.83 | nan | 10.3 |
| ECR | 54 | 0.789 | 0.33 | 0.50 | 0.83 | nan | 7.8 |
| ADP | 40 | 0.705 | 0.33 | 0.58 | 0.79 | nan | 7.1 |

_ADP coverage: matched 40/40 FantasyPros names to QB._

- **Model hits (called top tier, finished top tier):** Josh Allen (act 1, mdl 3)
- **Model busts (ranked top tier, finished worse):** Jayden Daniels (act 17, mdl 1), Lamar Jackson (act 16, mdl 2), Joe Burrow (act 15, mdl 6), Bo Nix (act 10, mdl 5), Jalen Hurts (act 7, mdl 4)
- **Best ADP values (finished better than drafted):** Matthew Stafford (act 3, adp 25), Daniel Jones (act 13, adp 33), Jaxson Dart (act 14, adp 32), Trevor Lawrence (act 5, adp 20), Drake Maye (act 2, adp 16)
- **Biggest ADP reaches (drafted earlier than they finished):** Lamar Jackson (act 16, adp 2), Jayden Daniels (act 17, adp 3), Baker Mayfield (act 19, adp 7), Tua Tagovailoa (act 32, adp 21), Joe Burrow (act 15, adp 4)


### QB — top 24 by actual finish

| actual | player | model | ECR | ADP | actual PPG |
|---|---|---|---|---|---|
| 1 | Josh Allen | 3 | 2 | 1 | 22.79 |
| 2 | Drake Maye | 23 | 14 | 16 | 20.70 |
| 3 | Matthew Stafford | 20 | 23 | 25 | 20.61 |
| 4 | Patrick Mahomes | 17 | 6 | 6 | 20.41 |
| 5 | Trevor Lawrence | 19 | 20 | 20 | 19.89 |
| 6 | Brock Purdy | 10 | 13 | 13 | 19.71 |
| 7 | Jalen Hurts | 4 | 4 | 5 | 18.82 |
| 8 | Caleb Williams | 11 | 18 | 12 | 18.72 |
| 9 | Dak Prescott | 25 | 11 | 11 | 18.46 |
| 10 | Bo Nix | 5 | 10 | 8 | 17.93 |
| 11 | Justin Herbert | 7 | 7 | 14 | 17.93 |
| 12 | Jared Goff | 13 | 17 | 10 | 17.47 |
| 13 | Daniel Jones | 30 | 28 | 33 | 17.42 |
| 14 | Jaxson Dart | — | 30 | 32 | 17.26 |
| 15 | Joe Burrow | 6 | 5 | 4 | 16.81 |
| 16 | Lamar Jackson | 2 | 1 | 2 | 16.53 |
| 17 | Jayden Daniels | 1 | 3 | 3 | 16.33 |
| 18 | Jacoby Brissett | 64 | — | — | 16.25 |
| 19 | Baker Mayfield | 8 | 9 | 7 | 16.00 |
| 20 | Justin Fields | 21 | 8 | 15 | 15.85 |
| 21 | Jordan Love | 15 | 15 | 17 | 15.68 |
| 22 | C.J. Stroud | 18 | 19 | 18 | 14.90 |
| 23 | Tyler Shough | — | 35 | 35 | 14.36 |
| 24 | Aaron Rodgers | 31 | 29 | 28 | 14.19 |

## RB

| source | n | Spearman | P@6 | P@12 | P@24 | P@36 | mean rank err |
|---|---|---|---|---|---|---|---|
| **model** | 112 | 0.829 | nan | 0.67 | 0.71 | 0.86 | 20.5 |
| ECR | 108 | 0.751 | nan | 0.75 | 0.75 | 0.81 | 18.7 |
| ADP | 85 | 0.755 | nan | 0.75 | 0.71 | 0.83 | 17.8 |

_ADP coverage: matched 85/92 FantasyPros names to RB._

- **Model hits (called top tier, finished top tier):** Bijan Robinson (act 2, mdl 3), Jahmyr Gibbs (act 3, mdl 1), Jonathan Taylor (act 4, mdl 10), De'Von Achane (act 5, mdl 6), James Cook (act 6, mdl 11)
- **Model busts (ranked top tier, finished worse):** Alvin Kamara (act 35, mdl 7), Breece Hall (act 21, mdl 5), Saquon Barkley (act 15, mdl 2), Bucky Irving (act 18, mdl 9), Derrick Henry (act 8, mdl 4)
- **Best ADP values (finished better than drafted):** Kenneth Gainwell (act 20, adp 66), Cam Skattebo (act 9, adp 32), Woody Marks (act 37, adp 57), Ty Johnson (act 58, adp 77), Rico Dowdle (act 23, adp 42)
- **Biggest ADP reaches (drafted earlier than they finished):** Kaleb Johnson (act 111, adp 23), Isaac Guerendo (act 129, adp 47), Roschon Johnson (act 123, adp 54), Dameon Pierce (act 114, adp 59), Tahj Brooks (act 116, adp 62)


### RB — top 24 by actual finish

| actual | player | model | ECR | ADP | actual PPG |
|---|---|---|---|---|---|
| 1 | Christian McCaffrey | 14 | 5 | 5 | 24.51 |
| 2 | Bijan Robinson | 3 | 1 | 1 | 21.81 |
| 3 | Jahmyr Gibbs | 1 | 2 | 3 | 21.58 |
| 4 | Jonathan Taylor | 10 | 11 | 9 | 21.31 |
| 5 | De'Von Achane | 6 | 6 | 8 | 20.18 |
| 6 | James Cook | 11 | 13 | 11 | 17.78 |
| 7 | Chase Brown | 20 | 8 | 9 | 16.62 |
| 8 | Derrick Henry | 4 | 7 | 6 | 16.44 |
| 9 | Cam Skattebo | — | 38 | 32 | 15.96 |
| 10 | Josh Jacobs | 8 | 10 | 7 | 15.81 |
| 11 | Kyren Williams | 12 | 12 | 10 | 15.49 |
| 12 | Javonte Williams | 37 | 31 | 29 | 15.18 |
| 13 | Omarion Hampton | — | 15 | 12 | 15.08 |
| 14 | Travis Etienne | 27 | 22 | 25 | 14.94 |
| 15 | Saquon Barkley | 2 | 3 | 2 | 14.52 |
| 16 | Ashton Jeanty | — | 4 | 4 | 14.42 |
| 17 | D'Andre Swift | 18 | 23 | 20 | 14.29 |
| 18 | Bucky Irving | 9 | 9 | 7 | 13.85 |
| 19 | Jaylen Warren | 39 | 32 | 24 | 13.57 |
| 20 | Kenneth Gainwell | 79 | 72 | 66 | 13.02 |
| 21 | Breece Hall | 5 | 16 | 13 | 12.98 |
| 22 | Rhamondre Stevenson | 24 | 41 | 33 | 12.77 |
| 23 | Rico Dowdle | 38 | 54 | 42 | 12.72 |
| 24 | RJ Harvey | — | 20 | 18 | 12.15 |

## WR

| source | n | Spearman | P@6 | P@12 | P@24 | P@36 | mean rank err |
|---|---|---|---|---|---|---|---|
| **model** | 176 | 0.854 | nan | 0.58 | 0.58 | 0.64 | 29.1 |
| ECR | 145 | 0.746 | nan | 0.67 | 0.67 | 0.69 | 23.3 |
| ADP | 106 | 0.720 | nan | 0.58 | 0.62 | 0.72 | 24.1 |

_ADP coverage: matched 106/110 FantasyPros names to WR._

- **Model hits (called top tier, finished top tier):** Puka Nacua (act 1, mdl 7), Jaxon Smith-Njigba (act 2, mdl 9), Ja'Marr Chase (act 3, mdl 1), Amon-Ra St. Brown (act 4, mdl 2), Drake London (act 7, mdl 8)
- **Model busts (ranked top tier, finished worse):** Malik Nabers (act —, mdl 4), Brian Thomas Jr. (act 45, mdl 6), Justin Jefferson (act 30, mdl 5), Ladd McConkey (act 36, mdl 12), CeeDee Lamb (act 10, mdl 3)
- **Best ADP values (finished better than drafted):** Christian Watson (act 17, adp 88), Michael Wilson (act 19, adp 70), Tre Tucker (act 48, adp 95), Quentin Johnston (act 18, adp 62), Troy Franklin (act 41, adp 84)
- **Biggest ADP reaches (drafted earlier than they finished):** Adam Thielen (act 129, adp 57), Dont'e Thornton Jr. (act 139, adp 67), Jack Bech (act 126, adp 58), Kyle Williams (act 122, adp 57), Efton Chism III (act 142, adp 79)


### WR — top 24 by actual finish

| actual | player | model | ECR | ADP | actual PPG |
|---|---|---|---|---|---|
| 1 | Puka Nacua | 7 | 5 | 6 | 23.44 |
| 2 | Jaxon Smith-Njigba | 9 | 10 | 11 | 21.17 |
| 3 | Ja'Marr Chase | 1 | 1 | 1 | 19.60 |
| 4 | Amon-Ra St. Brown | 2 | 7 | 4 | 19.06 |
| 5 | Rashee Rice | 34 | 36 | 24 | 18.76 |
| 6 | George Pickens | 35 | 24 | 25 | 17.17 |
| 7 | Drake London | 8 | 9 | 8 | 16.82 |
| 8 | Chris Olave | 44 | 39 | 30 | 16.75 |
| 9 | Davante Adams | 17 | 17 | 17 | 15.92 |
| 10 | CeeDee Lamb | 3 | 2 | 3 | 15.45 |
| 11 | Nico Collins | 14 | 4 | 5 | 15.08 |
| 12 | A.J. Brown | 10 | 11 | 9 | 14.69 |
| 13 | Zay Flowers | 19 | 33 | 23 | 14.31 |
| 14 | Garrett Wilson | 11 | 14 | 14 | 14.21 |
| 15 | Tee Higgins | 16 | 13 | 12 | 14.11 |
| 16 | Wan'Dale Robinson | 40 | 62 | 58 | 13.62 |
| 17 | Christian Watson | 64 | 107 | 88 | 13.24 |
| 18 | Quentin Johnston | 47 | 66 | 62 | 13.17 |
| 19 | Michael Wilson | 62 | 79 | 70 | 12.98 |
| 20 | Jameson Williams | 42 | 28 | 22 | 12.94 |
| 21 | Courtland Sutton | 26 | 20 | 18 | 12.92 |
| 22 | Tetairoa McMillan | — | 16 | 20 | 12.55 |
| 23 | DK Metcalf | 28 | 29 | 19 | 12.48 |
| 24 | Stefon Diggs | 50 | 35 | 34 | 12.37 |
