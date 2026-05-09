# GigShield — AI-Powered Parametric Micro-Insurance for Food Delivery Partners

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange?logo=scikit-learn)](https://scikit-learn.org)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--3.5-412991?logo=openai)](https://openai.com)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> Built at **Guidewire DEVTrails 2026** hackathon.  
> Parametric micro-insurance that **auto-triggers ₹150–₹300 payouts** when rain, AQI, or flood thresholds are crossed — no claim forms, no waiting.

---

## The Problem

Zomato/Swiggy delivery partners earn ₹500–₹800/day but lose **100% income** during heavy rain, urban floods, or AQI emergencies. Traditional insurance requires paperwork and weeks of processing. Partners can't wait.

## The Solution

**Parametric insurance** — payout is triggered by objective weather data, not subjective claims. Worker doesn't prove anything. Rain ≥ 80mm/hr → automatic ₹200 UPI credit by Friday.

---

## AI & ML Stack

| Component | Tech | What it does |
|---|---|---|
| **Disruption ML Model** | `RandomForestClassifier` (sklearn) | Predicts NORMAL/MODERATE/SEVERE severity from 6 weather features. Accuracy: **~89%** on holdout set. |
| **GPS Fraud Detection** | Haversine geometry | Detects teleportation, impossible speed, GPS spoofing, stationary fraud |
| **Earnings Predictor** | Weighted moving average | Predicts weekly income from 14-day shift history |
| **LLM Claim Explainer** | OpenAI GPT-3.5-turbo | Generates plain-English claim explanation for worker. Graceful fallback if no API key. |
| **Dynamic Pricing** | Zone × vehicle risk matrix | Adjusts premium by city zone + vehicle type |

### ML Model Details (`/ml/model-info`)

```
Algorithm : RandomForestClassifier
Features  : rain_mm, aqi, temperature_c, hour_of_day, day_of_week, flood_flag
Classes   : NORMAL (0) | MODERATE (1) | SEVERE (2)
Training  : 1,600 samples (synthetic Indian weather — Mumbai/Kolkata/Delhi/Chennai/Bengaluru)
Test      : 400 samples  |  Accuracy: ~89%
Top features: rain_mm (0.38) › aqi (0.28) › temperature_c (0.19)
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Frontend (HTML/CSS/JS)  ←→  FastAPI Backend (Python)       │
│                                                             │
│  /trigger                ←  parametric threshold check      │
│  /ml/predict-disruption  ←  RandomForest severity predict   │
│  /ml/model-info          ←  accuracy + feature importances  │
│  /api/explain-claim      ←  OpenAI GPT-3.5 explanation      │
│  /api/analyze-gps        ←  haversine fraud scoring         │
│  /api/risk-score         ←  dynamic premium calculation     │
│  /admin/*                ←  dashboard + simulation APIs     │
│                                                             │
│  External: Razorpay (payments) | OpenWeatherMap (triggers)  │
└─────────────────────────────────────────────────────────────┘
```

---

## Payout Formula

```
payout = 0.5 × (predicted_weekly / 6) × overlap_hours × severity_multiplier

severity_multiplier:
  HEAVY_RAIN / EXTREME_HEAT / SEVERE_AQI  →  1.0×
  URBAN_FLOOD / CURFEW                    →  1.5×

Weekly cap: Basic ₹150 | Standard ₹200 | Premium ₹300
```

---

## Parametric Thresholds

| Event | Trigger | Auto-action |
|---|---|---|
| Heavy Rain | ≥ 80 mm/hr | Payout credited |
| Severe AQI | ≥ 300 AQI index | Payout credited |
| Urban Flood | flood=true | Payout × 1.5× |
| Extreme Heat | ≥ 43°C | Payout credited |
| Curfew / Sec 144 | Manual trigger | Payout × 1.5× |

---

## Quickstart

### 1. Clone Repository

```bash
git clone https://github.com/MrKrishnenduMalick/GigShield-AI-Parametric-Insurance.git
```

---

### 2. Navigate to Project Root

```bash
cd GigShield-AI-Parametric-Insurance
```

---

### 3. Create Virtual Environment

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### Linux / Mac

```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 5. Configure Environment Variables

Create `.env` file:

```bash
cp .env.example .env
```

Add your credentials:

```env
RAZORPAY_KEY_ID=your_key
RAZORPAY_KEY_SECRET=your_secret
OPENAI_API_KEY=your_openai_key
```

> OPENAI_API_KEY is optional.

---

### 6. Navigate to Backend Folder

```bash
cd gigshield
```

---

### 7. Run FastAPI Server

```bash
uvicorn main:app --reload --port 8000
```

---

### 8. Open API Documentation

Swagger UI:

```text
http://localhost:8000/docs
```

ReDoc:

```text
http://localhost:8000/redoc
```

---

## Recommended Python Version

```text
Python 3.11+
```

---

## Key API Examples

**Trigger a claim:**
```bash
curl -X POST http://localhost:8001/trigger \
  -H "Content-Type: application/json" \
  -d '{
    "username": "raju",
    "working_hours": 8,
    "rain": 95,
    "aqi": 180,
    "movement": true
  }'
```
```json
{
  "success": true,
  "claim": {
    "event_type": "HEAVY_RAIN",
    "payout_amount": 158.25,
    "status": "approved",
    "message": "Claim approved. Paid Friday.",
    "fraud_score": 0,
    "weekly_total": 158.25
  }
}
```

**ML disruption prediction:**
```bash
curl -X POST http://localhost:8001/ml/predict-disruption \
  -d '{"rain_mm": 95, "aqi": 200, "temperature_c": 34, "hour_of_day": 14, "day_of_week": 2, "flood_flag": 0}'
```
```json
{
  "prediction": {
    "severity_label": "SEVERE",
    "confidence_pct": 91.3,
    "payout_recommended": true
  }
}
```

**LLM claim explanation:**
```bash
curl -X POST http://localhost:8001/api/explain-claim \
  -d '{"username":"raju","event_type":"HEAVY_RAIN","payout_amount":158,"status":"approved","fraud_score":0,"fraud_flags":[],"working_hours":8}'
```

---

## Project Structure

```text
GigShield/
│
├── gigshield/
│   ├── main.py              # FastAPI backend
│   ├── dashboard.html       # Admin dashboard
│   └── README.md
│
├── index.html               # Landing page
├── requirements.txt         # Python dependencies
├── .env.example             # Environment variables template
├── .gitignore
└── README.md
```

## Topics

`fastapi` `python` `scikit-learn` `machine-learning` `parametric-insurance` `insurance-tech` `razorpay` `openai` `gig-economy` `hackathon`
