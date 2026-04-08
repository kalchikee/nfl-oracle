# NFL Oracle v4.0 — GitHub Setup Guide

## How It Works

- **Tuesday 3 PM ET** → Recap workflow runs: scores last week, sends Discord results
- **Tuesday 7 PM ET** → Picks workflow runs: sends Discord picks for the upcoming week
- **August 15** → Model auto-trains on 5 seasons (2021–2025) before the season starts
- **September–February** → Both weekly workflows run automatically
- **After Super Bowl** → Workflows keep running but the NFL off-season check exits early

## Step 1: Create the GitHub Repository

```bash
cd "c:\Users\kalch\OneDrive\Desktop\Kalshi\NFL"
git init
git add .
git commit -m "NFL Oracle v4.0 - initial commit"
```

Then create a new repo on GitHub (github.com → New repository → name it `nfl-oracle`).

```bash
git remote add origin https://github.com/YOUR_USERNAME/nfl-oracle.git
git branch -M main
git push -u origin main
```

## Step 2: Add the Discord Webhook Secret

1. Go to your GitHub repo → **Settings** → **Secrets and variables** → **Actions**
2. Click **New repository secret**
3. Name: `DISCORD_WEBHOOK_URL`
4. Value: `https://discord.com/api/webhooks/1491223085322469407/GB7_8ltkvs7TdSQkinK9o-ZfHDnG7XqQFuITm3dSFxmmr79WNce-NsDldG9aD8fgNPQX`
5. Click **Add secret**

## Step 3: (Optional) Add The Odds API Key

If you want sharper Vegas lines (500 free requests/month):
1. Sign up at https://the-odds-api.com/
2. Add secret: `ODDS_API_KEY` = your key

Without it, the system uses ESPN's embedded odds — still works fine.

## Step 4: Train the Model

Trigger the training workflow manually:
1. Go to **Actions** tab in your GitHub repo
2. Click **Train ML Model**
3. Click **Run workflow** → **Run workflow**

This downloads 5 seasons of NFL data (2021–2025), trains the logistic regression, and saves the model. Takes ~5 minutes.

The model auto-retrains every August 15 before each season.

## Step 5: Test a Manual Run

1. Go to **Actions** → **NFL Weekly Picks** → **Run workflow**
2. Check your Discord channel for the message

## Step 6: That's It

The workflows run automatically from here. You don't need your computer on.

---

## Schedule Summary

| Day/Time | Action |
|---|---|
| Tue 3 PM ET | Scores last week's picks, sends recap to Discord |
| Tue 7 PM ET | Generates picks for the upcoming week, sends to Discord |
| Aug 15 | Retrains ML model on prior 5 seasons |

## Discord Message Format

**Weekly Picks (Tuesday 7 PM ET)**
```
🏈 NFL Oracle — 2026 Week 5 Picks
14 games this week · 3 high-conviction picks

🔥🔥🔥 DEN @ KC · Sunday
Pick: KC (74.2%) | Spread: KC -7.5 | O/U: 47
Proj: DEN 21 – KC 31  Edge: +8.3%

🔥🔥 BUF @ MIA · Sunday
Pick: BUF (68.1%) | Spread: BUF -3.0 | O/U: 45
Proj: MIA 22 – BUF 26
...

⭐ High-Conviction Picks — 3 bets this week
🔥🔥🔥 EXTREME: DEN @ KC → BET KC  (74.2%)
Why: Model win%: 74.2% · +8.3% vs Vegas · Away QB doubtful
```

**Weekly Recap (Tuesday 3 PM ET)**
```
📊 NFL Oracle — 2026 Week 4 Results

📈 Summary
🟢 9/13 correct  (69%)
⭐ High-conviction: 3/3  (100%)
Brier score: 0.1823

🎯 Game-by-Game
✅⭐ KC 31 – DEN 21  (picked KC)
✅ BUF 26 – MIA 22  (picked BUF)
❌ PHI 17 – DAL 20  (picked PHI)
...
```

## Architecture

```
ESPN NFL API (free) ──→ Feature Engineering (26 features)
                              ↓
                    Monte Carlo (10k simulations)
                              ↓
               ML Meta-model (Logistic Regression)
                    trained on 2021–2025 data
                              ↓
              Isotonic Calibration + Edge Detection
                              ↓
                    SQLite (GitHub Artifacts)
                              ↓
                    Discord Webhook Alerts
```

## Troubleshooting

**"No games found"**: The ESPN API sometimes returns empty during off-season. Normal behavior.

**"No predictions in DB"**: On first run of the picks workflow, it runs the pipeline automatically. After that, predictions are stored in the artifact.

**Model not loaded**: Run the Train ML Model workflow first. Until then, the system falls back to Monte Carlo predictions (still accurate).

**Discord not receiving**: Double-check the `DISCORD_WEBHOOK_URL` secret is set correctly (no trailing spaces).
