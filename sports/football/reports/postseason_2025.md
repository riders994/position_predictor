# Postseason report — 2025

_Generated 2026-08-28._ Model preseason projection (trained on labels < 2025) vs preseason **ECR** and **ADP**, graded against actual 2025 finish (PPR PPG). Ranking metrics are computed on each source's covered ∩ actual-eligible players.

## QB

| source | n | Spearman | P@6 | P@12 | P@24 | P@36 | mean rank err |
|---|---|---|---|---|---|---|---|
| **model** | 63 | 0.746 | 0.17 | 0.58 | 0.79 | nan | 10.1 |
| ECR | 54 | 0.789 | 0.33 | 0.50 | 0.83 | nan | 7.8 |
| ADP | 29 | 0.506 | 0.33 | 0.58 | 0.83 | nan | 7.1 |

_ADP coverage: matched 29/29 FFC names to QB._

- **Model hits (called top tier, finished top tier):** Josh Allen (act 1, mdl 3)
- **Model busts (ranked top tier, finished worse):** Jayden Daniels (act 17, mdl 1), Lamar Jackson (act 16, mdl 2), Joe Burrow (act 15, mdl 5), Bo Nix (act 10, mdl 6), Jalen Hurts (act 7, mdl 4)
- **Best ADP values (finished better than drafted):** Matthew Stafford (act 3, adp 22), Drake Maye (act 2, adp 16), Jaxson Dart (act 14, adp 27), Trevor Lawrence (act 5, adp 17), Caleb Williams (act 8, adp 13)
- **Biggest ADP reaches (drafted earlier than they finished):** Lamar Jackson (act 16, adp 1), J.J. McCarthy (act 29, adp 14), Jayden Daniels (act 17, adp 4), Baker Mayfield (act 19, adp 7), Joe Burrow (act 15, adp 3)


### QB — top 24 by actual finish

| actual | player | model | ECR | ADP | actual PPG |
|---|---|---|---|---|---|
| 1 | Josh Allen | 3 | 2 | 2 | 22.79 |
| 2 | Drake Maye | 21 | 14 | 16 | 20.70 |
| 3 | Matthew Stafford | 20 | 23 | 22 | 20.61 |
| 4 | Patrick Mahomes | 17 | 6 | 6 | 20.41 |
| 5 | Trevor Lawrence | 16 | 20 | 17 | 19.89 |
| 6 | Brock Purdy | 10 | 13 | 9 | 19.71 |
| 7 | Jalen Hurts | 4 | 4 | 5 | 18.82 |
| 8 | Caleb Williams | 12 | 18 | 13 | 18.72 |
| 9 | Dak Prescott | 26 | 11 | 10 | 18.46 |
| 10 | Bo Nix | 6 | 10 | 8 | 17.93 |
| 11 | Justin Herbert | 7 | 7 | 15 | 17.93 |
| 12 | Jared Goff | 9 | 17 | 12 | 17.47 |
| 13 | Daniel Jones | 25 | 28 | — | 17.42 |
| 14 | Jaxson Dart | — | 30 | 27 | 17.26 |
| 15 | Joe Burrow | 5 | 5 | 3 | 16.81 |
| 16 | Lamar Jackson | 2 | 1 | 1 | 16.53 |
| 17 | Jayden Daniels | 1 | 3 | 4 | 16.33 |
| 18 | Jacoby Brissett | 60 | — | — | 16.25 |
| 19 | Baker Mayfield | 8 | 9 | 7 | 16.00 |
| 20 | Justin Fields | 13 | 8 | 16 | 15.85 |
| 21 | Jordan Love | 15 | 15 | 18 | 15.68 |
| 22 | C.J. Stroud | 19 | 19 | 19 | 14.90 |
| 23 | Tyler Shough | — | 35 | — | 14.36 |
| 24 | Aaron Rodgers | 33 | 29 | 26 | 14.19 |

## RB

| source | n | Spearman | P@6 | P@12 | P@24 | P@36 | mean rank err |
|---|---|---|---|---|---|---|---|
| **model** | 112 | 0.860 | nan | 0.75 | 0.75 | 0.86 | 18.6 |
| ECR | 108 | 0.751 | nan | 0.75 | 0.75 | 0.81 | 18.7 |
| ADP | 63 | 0.735 | nan | 0.75 | 0.75 | 0.83 | 14.1 |

_ADP coverage: matched 63/64 FFC names to RB._

- **Model hits (called top tier, finished top tier):** Bijan Robinson (act 2, mdl 1), Jahmyr Gibbs (act 3, mdl 2), Jonathan Taylor (act 4, mdl 8), De'Von Achane (act 5, mdl 7), James Cook (act 6, mdl 11)
- **Model busts (ranked top tier, finished worse):** Chuba Hubbard (act 43, mdl 9), Alvin Kamara (act 35, mdl 4), Saquon Barkley (act 15, mdl 3), Kyren Williams (act 11, mdl 6), Derrick Henry (act 8, mdl 5)
- **Best ADP values (finished better than drafted):** Rico Dowdle (act 23, adp 58), Cam Skattebo (act 9, adp 38), Javonte Williams (act 12, adp 35), Kareem Hunt (act 41, adp 60), Woody Marks (act 37, adp 56)
- **Biggest ADP reaches (drafted earlier than they finished):** Kaleb Johnson (act 111, adp 28), Will Shipley (act 104, adp 49), Jerome Ford (act 83, adp 39), Tank Bigsby (act 80, adp 40), Ollie Gordon II (act 84, adp 51)


### RB — top 24 by actual finish

| actual | player | model | ECR | ADP | actual PPG |
|---|---|---|---|---|---|
| 1 | Christian McCaffrey | 25 | 5 | 4 | 24.51 |
| 2 | Bijan Robinson | 1 | 1 | 1 | 21.81 |
| 3 | Jahmyr Gibbs | 2 | 2 | 3 | 21.58 |
| 4 | Jonathan Taylor | 8 | 11 | 9 | 21.31 |
| 5 | De'Von Achane | 7 | 6 | 7 | 20.18 |
| 6 | James Cook | 11 | 13 | 14 | 17.78 |
| 7 | Chase Brown | 12 | 8 | 11 | 16.62 |
| 8 | Derrick Henry | 5 | 7 | 5 | 16.44 |
| 9 | Cam Skattebo | — | 38 | 38 | 15.96 |
| 10 | Josh Jacobs | 10 | 10 | 8 | 15.81 |
| 11 | Kyren Williams | 6 | 12 | 12 | 15.49 |
| 12 | Javonte Williams | 36 | 31 | 35 | 15.18 |
| 13 | Omarion Hampton | — | 15 | 15 | 15.08 |
| 14 | Travis Etienne | 32 | 22 | 32 | 14.94 |
| 15 | Saquon Barkley | 3 | 3 | 2 | 14.52 |
| 16 | Ashton Jeanty | — | 4 | 6 | 14.42 |
| 17 | D'Andre Swift | 19 | 23 | 21 | 14.29 |
| 18 | Bucky Irving | 13 | 9 | 10 | 13.85 |
| 19 | Jaylen Warren | 35 | 32 | 27 | 13.57 |
| 20 | Kenneth Gainwell | 61 | 72 | — | 13.02 |
| 21 | Breece Hall | 15 | 16 | 19 | 12.98 |
| 22 | Rhamondre Stevenson | 30 | 41 | 36 | 12.77 |
| 23 | Rico Dowdle | 24 | 54 | 58 | 12.72 |
| 24 | RJ Harvey | — | 20 | 23 | 12.15 |

## WR

| source | n | Spearman | P@6 | P@12 | P@24 | P@36 | mean rank err |
|---|---|---|---|---|---|---|---|
| **model** | 176 | 0.861 | nan | 0.58 | 0.58 | 0.64 | 27.2 |
| ECR | 145 | 0.746 | nan | 0.67 | 0.67 | 0.69 | 23.3 |
| ADP | 83 | 0.695 | nan | 0.58 | 0.62 | 0.72 | 19.9 |

_ADP coverage: matched 83/87 FFC names to WR._

- **Model hits (called top tier, finished top tier):** Puka Nacua (act 1, mdl 7), Jaxon Smith-Njigba (act 2, mdl 11), Ja'Marr Chase (act 3, mdl 1), Amon-Ra St. Brown (act 4, mdl 2), Drake London (act 7, mdl 8)
- **Model busts (ranked top tier, finished worse):** Malik Nabers (act —, mdl 5), Brian Thomas Jr. (act 45, mdl 6), Mike Evans (act 40, mdl 12), Justin Jefferson (act 30, mdl 4), CeeDee Lamb (act 10, mdl 3)
- **Best ADP values (finished better than drafted):** Alec Pierce (act 25, adp 82), Quentin Johnston (act 18, adp 72), Michael Wilson (act 19, adp 67), Wan'Dale Robinson (act 16, adp 57), Rashee Rice (act 5, adp 40)
- **Biggest ADP reaches (drafted earlier than they finished):** Dont'e Thornton Jr. (act 139, adp 64), Adam Thielen (act 129, adp 59), Matthew Golden (act 93, adp 38), Kyle Williams (act 122, adp 74), Jack Bech (act 126, adp 81)


### WR — top 24 by actual finish

| actual | player | model | ECR | ADP | actual PPG |
|---|---|---|---|---|---|
| 1 | Puka Nacua | 7 | 5 | 5 | 23.44 |
| 2 | Jaxon Smith-Njigba | 11 | 10 | 16 | 21.17 |
| 3 | Ja'Marr Chase | 1 | 1 | 1 | 19.60 |
| 4 | Amon-Ra St. Brown | 2 | 7 | 7 | 19.06 |
| 5 | Rashee Rice | 35 | 36 | 40 | 18.76 |
| 6 | George Pickens | 38 | 24 | 29 | 17.17 |
| 7 | Drake London | 8 | 9 | 9 | 16.82 |
| 8 | Chris Olave | 44 | 39 | 35 | 16.75 |
| 9 | Davante Adams | 17 | 17 | 14 | 15.92 |
| 10 | CeeDee Lamb | 3 | 2 | 3 | 15.45 |
| 11 | Nico Collins | 13 | 4 | 6 | 15.08 |
| 12 | A.J. Brown | 10 | 11 | 10 | 14.69 |
| 13 | Zay Flowers | 21 | 33 | 26 | 14.31 |
| 14 | Garrett Wilson | 9 | 14 | 20 | 14.21 |
| 15 | Tee Higgins | 16 | 13 | 11 | 14.11 |
| 16 | Wan'Dale Robinson | 36 | 62 | 57 | 13.62 |
| 17 | Christian Watson | 61 | 107 | — | 13.24 |
| 18 | Quentin Johnston | 48 | 66 | 72 | 13.17 |
| 19 | Michael Wilson | 62 | 79 | 67 | 12.98 |
| 20 | Jameson Williams | 34 | 28 | 30 | 12.94 |
| 21 | Courtland Sutton | 26 | 20 | 22 | 12.92 |
| 22 | Tetairoa McMillan | — | 16 | 25 | 12.55 |
| 23 | DK Metcalf | 29 | 29 | 17 | 12.48 |
| 24 | Stefon Diggs | 46 | 35 | 37 | 12.37 |

## TE

| source | n | Spearman | P@6 | P@12 | P@24 | P@36 | mean rank err |
|---|---|---|---|---|---|---|---|
| **model** | 107 | 0.802 | nan | 0.58 | 0.79 | nan | 17.7 |
| ECR | 71 | 0.803 | nan | 0.67 | 0.71 | nan | 11.5 |
| ADP | 26 | 0.561 | nan | 0.67 | 0.92 | nan | 8.6 |

_ADP coverage: matched 26/26 FFC names to TE._

- **Model hits (called top tier, finished top tier):** Trey McBride (act 1, mdl 2), Brock Bowers (act 2, mdl 1), George Kittle (act 3, mdl 3), Tucker Kraft (act 4, mdl 9), Kyle Pitts (act 5, mdl 12)
- **Model busts (ranked top tier, finished worse):** Taysom Hill (act 61, mdl 6), Jonnu Smith (act 42, mdl 10), Mark Andrews (act 27, mdl 7), David Njoku (act 26, mdl 8), Zach Ertz (act 19, mdl 11)
- **Best ADP values (finished better than drafted):** Kyle Pitts (act 5, adp 16), Dallas Goedert (act 6, adp 13), Tucker Kraft (act 4, adp 11), Darren Waller (act 17, adp 24), Hunter Henry (act 13, adp 18)
- **Biggest ADP reaches (drafted earlier than they finished):** Evan Engram (act 35, adp 9), T.J. Hockenson (act 29, adp 6), Isaiah Likely (act 41, adp 18), Jonnu Smith (act 42, adp 20), David Njoku (act 26, adp 7)


### TE — top 24 by actual finish

| actual | player | model | ECR | ADP | actual PPG |
|---|---|---|---|---|---|
| 1 | Trey McBride | 2 | 2 | 2 | 18.58 |
| 2 | Brock Bowers | 1 | 1 | 1 | 14.68 |
| 3 | George Kittle | 3 | 3 | 3 | 14.68 |
| 4 | Tucker Kraft | 9 | 9 | 11 | 14.65 |
| 5 | Kyle Pitts | 12 | 13 | 16 | 12.40 |
| 6 | Dallas Goedert | 14 | 16 | 13 | 12.34 |
| 7 | Sam LaPorta | 4 | 6 | 4 | 11.88 |
| 8 | Harold Fannin Jr. | — | 21 | — | 11.65 |
| 9 | Travis Kelce | 5 | 5 | 5 | 11.36 |
| 10 | Tyler Warren | — | 4 | 10 | 11.09 |
| 11 | Jake Ferguson | 21 | 12 | 12 | 11.06 |
| 12 | Juwan Johnson | 22 | 22 | — | 10.58 |
| 13 | Hunter Henry | 16 | 15 | 18 | 10.52 |
| 14 | Dalton Kincaid | 19 | 14 | 14 | 10.51 |
| 15 | Dalton Schultz | 24 | 30 | — | 10.45 |
| 16 | Colston Loveland | — | 17 | 15 | 10.32 |
| 17 | Darren Waller | — | 33 | 24 | 9.86 |
| 18 | Brenton Strange | 31 | 18 | 22 | 9.83 |
| 19 | Zach Ertz | 11 | 19 | 17 | 9.72 |
| 20 | Jake Tonges | 113 | — | — | 9.33 |
| 21 | Colby Parkinson | 40 | 49 | — | 9.27 |
| 22 | Oronde Gadsden II | — | 42 | — | 8.76 |
| 23 | AJ Barner | 38 | 38 | — | 8.66 |
| 24 | Theo Johnson | 28 | 29 | — | 8.52 |
