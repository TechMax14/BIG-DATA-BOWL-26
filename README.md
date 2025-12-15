# 🏈 AirDuel: Air Battle Index (ABI)

### Quantifying Receiver–Defender Battles While the Ball Is in the Air

_A metric developed for the 2026 NFL Big Data Bowl — Broadcast Visualization Track_

---

## 📘 Overview

**AirDuel** is a framework for evaluating the competitive interaction between the **targeted receiver** and the **defensive coverage** **during the ball’s flight only**. Rather than modeling pre-throw route running or post-catch YAC, AirDuel isolates the pure **air battle** — the decisive moment when skill, leverage, pursuit, and timing collide.

The main output is the **Air Battle Index (ABI)**:  
a **0–100 score** composed of four **0–25 submetrics**:

```
ABI_100 = Separation_25
        + Closing_25
        + Contest_25
        + CatchSurprise_25
```

This produces a clean, interpretable measurement of how a downfield passing duel played out.

---

## ⚡ Motivation

Downfield passes create some of the highest-leverage moments in football — jump balls, deep shots, contested catches, late separation wins, and elite defensive recoveries. Yet many analytics blur these moments together or treat them as binary outcomes.

AirDuel focuses specifically on this **in-air microbattle**, providing:

- A **continuous scale**, not just “caught or not caught”
- Player-centric insights for WRs **and** DBs
- Scheme-level summaries for routes and coverages
- Play-level narratives that translate well to broadcast

It answers questions like:

- _How well did the WR separate while the ball was in the air?_
- _How efficiently did the DB close that space?_
- _How tight and crowded was the catch point?_
- _How surprising was the outcome given the physics of the play?_

---

## 🎯 Scope of Analysis — Why Only Passes ≥ 10 Air Yards?

AirDuel restricts analysis to passes with **`pass_length >= 10` yards**.

Short throws (screens, bubbles, stick routes, quick RPO slants) do not meaningfully stress:

- true DB pursuit,
- sustained downfield leverage,
- receiver late separation skill,
- or contested catch environments.

This threshold aligns with the Big Data Bowl’s goal of analyzing **airborne player interactions**, not backfield or quick-game timing.

---

## 🧮 How ABI Works (Conceptual Diagram)

```
             ┌───────────────────┐
             │  Separation (S)    │  — WR space gained mid-flight
             └──────────┬────────┘
                        ↓
             ┌───────────────────┐
             │ Closing (C)       │  — DB pursuit + recovery efficiency
             └──────────┬────────┘
                        ↓
             ┌───────────────────┐
             │ Contest (X)       │  — Tightness & crowding at arrival
             └──────────┬────────┘
                        ↓
             ┌───────────────────┐
             │ Surprise (E)      │  — Catch probability vs outcome
             └──────────┬────────┘
                        ↓
             ╔═══════════════════╗
             ║ ABI Score (0–100) ║
             ╚═══════════════════╝
```

Each component is scaled to **0–25**, enabling a simple, balanced 4-part composition.

---

## 📂 Repository Structure

```
BIG-DATA-BOWL-26/
│
├── code/
│   ├── main.py                  # End-to-end ABI pipeline
│   ├── metrics/
│   │   ├── metric_pipeline.py
│   │   ├── sep_creation_metric.py
│   │   ├── closing_eff_metric.py
│   │   ├── contested_catch_metric.py
│   │   ├── xCatch_prob_metric.py
│   │   ├── abi_aggregator.py
│   │   └── abi_narratives.py
│   ├── utils/
│   │   ├── data_loader.py
│   │   └── data_preprocessor.py
│   └── viz/
│       ├── visual_pipeline.py
│       ├── abi_hero_visual.py
│       ├── play_insights.py
│       └── summary_visuals.py
│
├── data/
│   ├── supplementary_data.csv
│   ├── train/
│   ├── processed/
│   ├── models/
│   └── abi/
│       ├── metrics/
│       └── results/
│
├── visuals/
│   ├── abi_hero/
│   ├── plays/
│   ├── scheme_insights/
│   ├── summary_teams/
│   └── wr_leaderboard/
│
├── media/
│   ├── video_assets/
│   └── images/
│
└── README.md
```

---

## 🧠 Submetric Summary

| Submetric                       | Description                                                                          |
| ------------------------------- | ------------------------------------------------------------------------------------ |
| **Separation Gain (S)**         | How much separation the WR creates during ball flight                                |
| **Closing Efficiency (C)**      | How effectively defenders reduce that space                                          |
| **Contested Arrival (X)**       | Tightness & local defender density at the arrival frame                              |
| **Expected Catch Surprise (E)** | Surprise of outcome vs model expectations (e.g., improbable catches, shocking drops) |

---

## 📈 What AirDuel Produces

### ✔ Play-Level ABI Dataset

Includes metrics, scores, context, classification labels, and automatic highlight sentences.

### ✔ Player Leaderboards

- WR Air Battle Wins (ABW)
- Separation creators
- Catch-over-expected performers
- Defensive closers
- Tight coverage specialists

### ✔ Scheme Insights

- Route × coverage heatmaps
- Team defensive closing + contest profiles
- Offensive separation tendencies

### ✔ Broadcast-Ready Play Packages

- Play animation
- ABI circular progress meter
- Metric progression timelines
- Catch-space snapshots
- Automatically generated analytic blurbs

---

## 🔧 Running the Pipeline

1. Place tracking data into:

```
data/train/
```

- and supplemenarty data into:

```
data/
```

2. Run:

```bash
python code/main.py
```

This produces:

- Enriched frame-level dataset
- Play index of qualifying deep targets
- All submetric CSVs
- ABI results
- WR/DB/team leaderboards
- Visuals for competition submission

Outputs are written to:

```
data/abi/
visuals/
```

---

## ⭐ Sample Featured Plays

- **Justin Jefferson — Week 16 (Game 2023122405, Play 4357)**  
  **ABI ≈ 92 — Elite Receiver Win**  
  On 3rd-and-27, Jefferson accelerates through the stem of a corner route, creating over 16 yards of separation during ball flight. Two defenders close simultaneously at high speed, compressing the window and producing a severely contested arrival. Despite an expected catch probability below 10%, Jefferson completes the catch under heavy closing pressure, resulting in one of the highest xCatch Surprise scores of the season.

- **Rashid Shaheed — Week 18**  
  **ABI ≈ 88 — Late Separation + Improbable Completion**  
  Shaheed gains minimal early separation but creates late space as the ball travels, forcing defenders into recovery mode. Closing pressure arrives late and off-angle, producing a narrow but survivable window at the catch point. The completion significantly outperforms model expectations, highlighting Shaheed’s ability to convert low-probability air battles into explosive plays.

- **A.J. Brown — Week 7**  
  **ABI ≈ 79 — Physical, Prototypical Air Battle**  
  Brown maintains leverage through contact on a downfield route, holding separation against tight man coverage. Defensive closing is sustained and physical, resulting in a contested arrival with multiple defenders within the catch radius. The play exemplifies a high-intensity but expected outcome, showcasing strength, timing, and body control rather than surprise.


---

## 🏈 Weekly Coaching Use Case

The Air Battle Index (ABI) is designed to fit directly into a weekly coaching and scouting workflow. On a week-to-week basis, teams can use ABI to identify which downfield targets consistently win (or lose) the receiver–defender battle while the ball is in the air, independent of quarterback decision-making or scheme design. Coaches can filter for high-ABI incompletions to study receivers who are creating separation but not being rewarded, or low-ABI completions to identify throws that were routine rather than contested. Defensively, ABI highlights which coverage units and defenders most effectively close windows and disrupt catch points, even when the pass is completed. This allows position coaches to pair film with objective movement-based evidence when preparing game plans, evaluating player performance, and coaching technique throughout the season.

---

## 🙌 About

**Author:** Max Fishman  
**Competition:** NFL Big Data Bowl 2026 — Broadcast Visualization Track  
**Project:** AirDuel — Air Battle Index (ABI)

ABI is a **micro-interaction metric**, not a generic WR grade. Its purpose is to explain **how** and **why** downfield air battles are won or lost.

---
