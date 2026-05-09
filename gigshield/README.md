# 🛡️ GigShield — AI-Powered Parametric Micro-Insurance

> Protecting India's delivery backbone, one week at a time.
> Guidewire DEVTrails 2026 · Phase 2

---

## ⚡ Quick Start (60 seconds)

```bash
cd gigshield
uvicorn main:app --reload
```

The FastAPI backend will start at http://127.0.0.1:8000
For the Dashboard:

Right-click on dashboard.html → Select "Open with Live Server"

It will automatically open in your browser (usually at http://127.0.0.1:5500).
API docs: http://localhost:8000/docs
---

## 📁 Project Structure

```
CODE/
├── main.py           ← FastAPI backend (all logic in one file)
├── dashboard.html    ← Worker dashboard UI
└── README.md
```

---

## 🔧 Requirements

```bash
pip install fastapi uvicorn
```

No database setup needed — all data is in-memory (resets on restart).

---

## 🌐 API Endpoints

### Auth & Users

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/users/register` | Register a new worker |
| GET  | `/api/users/{username}` | Get worker profile |

### Claims & Triggers

| Method | Path | Description |
|--------|------|-------------|
| POST | `/trigger` | Fire a parametric trigger (rain/heat/AQI/flood) |
| GET  | `/api/claims` | Get all claims |
| GET  | `/api/claims/{username}` | Get claims for a worker |

### AI Modules

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/analyze-gps` | GPS fraud detection (returns score 0–100) |
| POST | `/api/predict-earnings` | Moving-average weekly earnings prediction |
| POST | `/api/risk-score` | Dynamic premium risk score |
| POST | `/api/check-eligibility` | Full eligibility + payout calculation |

### Scheduler

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/weekly-reset` | Manually trigger Saturday 00:00 reset |

### Admin

| Method | Path | Description |
|--------|------|-------------|
| GET  | `/admin/stats` | Platform overview (users, claims, payouts) |
| GET  | `/admin/users` | All registered workers |
| GET  | `/admin/shifts` | All shift records |
| GET  | `/admin/claims` | All claims |
| POST | `/admin/claims/{id}/approve` | Manually approve a claim |
| POST | `/admin/claims/{id}/reject` | Reject a claim with reason |
| POST | `/admin/simulate-trigger` | Simulate a disruption for any worker |

### Health

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Service health check |

---

## 🧠 How It Works

### Weekly Cycle

| Time | Event |
|------|-------|
| Saturday 00:00 | Week resets. Earnings predicted. Policy activated. |
| Mon–Fri | Weather checked. Active shifts monitored. Claims auto-generated. |
| Friday 23:59 | All approved claims paid. UPI payout batch runs. |

### Payout Formula

```
Payout = 0.5 × (predicted_weekly_earnings / 6) × overlap_hours × severity_multiplier
```

Weekly cap applied per plan.

### Parametric Triggers

| Trigger | Threshold | Multiplier |
|---------|-----------|------------|
| Heavy Rain | > 80 mm/hr | 1.0× |
| Extreme Heat | > 42°C | 1.0× |
| Severe AQI | > 300 | 1.0× |
| Urban Flood | Alert + > 150 mm/24hr | 1.5× |
| Curfew | Section 144 | 1.5× |
| Short Shift | < 4 hr + weather event | 1.0× |

### Fraud Detection

| Flag | Score Added |
|------|-------------|
| GPS spoofing (accuracy = 0) | +35 |
| Impossible speed > 120 km/h | +20 per event |
| Low activity < 15% movement | +25 |
| Low coverage < 10 GPS points | +30 |

| Score Range | Verdict |
|-------------|---------|
| 0 – 39 | ✅ Auto-Approved |
| 40 – 69 | 🟡 Manual Review |
| 70 – 100 | ❌ Auto-Rejected |

### Insurance Plans

| Plan | Premium | Weekly Cap | Max Disruptions |
|------|---------|------------|-----------------|
| Basic | ₹20/wk | ₹150 | 1 |
| Standard | ₹30/wk | ₹200 | 2 |
| Premium | ₹50/wk | ₹500 | 3 |

---

## 🧪 Quick Demo

**1. Register a worker:**
```bash
curl -X POST http://localhost:8000/api/users/register \
  -H "Content-Type: application/json" \
  -d '{"username":"raju","city":"Mumbai","zone":"Bandra","plan":"standard","vehicle":"motorcycle","platform":"swiggy"}'
```

**2. Simulate a rain trigger:**
```bash
curl -X POST http://localhost:8000/trigger \
  -H "Content-Type: application/json" \
  -d '{"username":"raju","rain_mm":85,"aqi":80,"flood_alert":false,"working_hours":3.5}'
```

**3. Check claims:**
```bash
curl http://localhost:8000/api/claims/raju
```

**4. Admin simulate (from admin panel):**
```bash
curl -X POST http://localhost:8000/admin/simulate-trigger \
  -H "Content-Type: application/json" \
  -d '{"username":"raju","disruption_type":"HEAVY_RAIN","working_hours":2.0}'
```

---

## 👥 Team

| Name | Role |
|------|------|
| Saheli Roy | Backend Development |
| Krishnendu Malick | AI & Backend |
| Rishav Kumar | Frontend & UI/UX |
| Aniket Das | Frontend Development |
| Rishika Singhadeo | UI/UX Design |
