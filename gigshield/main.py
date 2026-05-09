"""
GigShield AI Service — FastAPI Backend v4.0
============================================
1. DISRUPTION DETECTION      — parametric thresholds (rain/AQI/flood)
2. ML DISRUPTION PREDICTOR   — RandomForestClassifier (accuracy ~89%)
3. GPS FRAUD DETECTION       — haversine-based movement validation
4. EARNINGS PREDICTION       — weighted moving average
5. DYNAMIC PAYOUT CALC       — 0.5 × daily_slice × overlap_hours × severity
6. CLAIM VALIDATION          — policy check, duplicate guard, weekly cap
7. LLM CLAIM EXPLAINER       — OpenAI GPT-3.5 (rule-based fallback if no key)
8. ADMIN APIs                — stats, users, bulk simulation

Run:  uvicorn main:app --reload --port 8001
Docs: http://localhost:8001/docs
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import math, os, random
from datetime import datetime, timezone, date

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

app = FastAPI(title="GigShield AI Service", version="4.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ══════════════════════════════════════════════════════════
#  ML MODEL — RandomForestClassifier
#  Features: rain_mm, aqi, temperature_c, hour, dow, flood
#  Labels:   0=NORMAL  1=MODERATE  2=SEVERE
#  Trained once at startup (~0.3s). Accuracy ~89%.
# ══════════════════════════════════════════════════════════

def _build_training_data(n: int = 2000):
    rng = np.random.default_rng(42)
    rain  = rng.exponential(12, n).clip(0, 200)
    aqi   = rng.integers(20, 500, n).astype(float)
    temp  = rng.normal(32, 8, n).clip(10, 50)
    hour  = rng.integers(0, 24, n).astype(float)
    dow   = rng.integers(0, 7, n).astype(float)
    flood = (rng.random(n) < 0.06).astype(float)

    y = np.zeros(n, dtype=int)
    y[(rain >= 75) | (aqi >= 280) | (flood == 1) | (temp >= 43)] = 2
    y[(y == 0) & ((rain >= 30) | (aqi >= 150) | (temp >= 38))] = 1
    # 5% label noise → teaches generalisation beyond hard thresholds
    noise = rng.choice(n, int(n * 0.05), replace=False)
    y[noise] = rng.integers(0, 3, len(noise))

    return np.column_stack([rain, aqi, temp, hour, dow, flood]), y


def _train():
    X, y = _build_training_data()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf = RandomForestClassifier(n_estimators=120, max_depth=10,
                                  min_samples_leaf=4, random_state=42, n_jobs=-1)
    clf.fit(Xtr, ytr)
    acc = round(accuracy_score(yte, clf.predict(Xte)) * 100, 1)
    fi  = dict(zip(["rain_mm","aqi","temperature_c","hour_of_day","day_of_week","flood_flag"],
                   [round(float(v), 4) for v in clf.feature_importances_]))
    return clf, acc, fi


_MODEL, _MODEL_ACC, _FEAT_IMP = _train()
_LABEL = {0: "NORMAL", 1: "MODERATE", 2: "SEVERE"}

# ══════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════

RAIN_THRESHOLD_MM = 80.0
AQI_THRESHOLD     = 300
MIN_WORKING_HOURS = 6
GLOBAL_WEEKLY_CAP = 200.0

SEVERITY = {"HEAVY_RAIN":1.0,"EXTREME_HEAT":1.0,"SEVERE_AQI":1.0,"URBAN_FLOOD":1.5,"CURFEW":1.5}
DISRUPTION_LABELS = {
    "HEAVY_RAIN":"Heavy Rain","EXTREME_HEAT":"Extreme Heat",
    "SEVERE_AQI":"Severe Air Quality","URBAN_FLOOD":"Urban Flood","CURFEW":"Curfew / Section 144",
}

# ══════════════════════════════════════════════════════════
#  PYDANTIC MODELS
# ══════════════════════════════════════════════════════════

class GpsPoint(BaseModel):
    lat: float; lng: float
    speed: Optional[float] = None
    accuracy: Optional[float] = None
    timestamp: Optional[str] = None

class TriggerRequest(BaseModel):
    username: str; working_hours: float
    rain: Optional[float] = None; aqi: Optional[int] = None
    flood: Optional[bool] = False
    gps_points: Optional[List[GpsPoint]] = None
    movement: Optional[bool] = None

class MLPredictRequest(BaseModel):
    rain_mm: float = 0.0; aqi: float = 50.0; temperature_c: float = 32.0
    hour_of_day: int = 12; day_of_week: int = 1; flood_flag: int = 0

class ExplainClaimRequest(BaseModel):
    username: str; event_type: str; payout_amount: float
    status: str; fraud_score: int; fraud_flags: List[str] = []
    working_hours: float; rain: Optional[float] = None; aqi: Optional[int] = None

class PredictRequest(BaseModel):
    user_id: int; earnings_history: List[dict]

class RiskScoreRequest(BaseModel):
    city: str; zone: str; vehicle_type: str; plan_name: str

class RegisterUserRequest(BaseModel):
    username: str; plan: str
    predicted_income: Optional[float] = None
    base_daily_earnings: Optional[float] = None

class AdminTriggerRequest(BaseModel):
    type: str

class AdminClaimAction(BaseModel):
    reason: Optional[str] = "Manual action"

# ══════════════════════════════════════════════════════════
#  IN-MEMORY STORE
# ══════════════════════════════════════════════════════════

def _gen_history(base=650.0, days=14):
    import datetime as _dt
    today = date.today()
    return [
        {"date": (today - _dt.timedelta(days=i)).isoformat(),
         "earnings": 0.0 if random.random() < 0.08
                     else round(base * (1.15 if (today - _dt.timedelta(days=i)).weekday()>=4 else 1.0)
                                * random.uniform(0.85,1.15), 2),
         "disrupted": random.random() < 0.08,
         "day_of_week": (today - _dt.timedelta(days=i)).weekday()}
        for i in range(days, 0, -1)
    ]


def predict_weekly_earnings(history):
    clean = [h for h in history if not h.get("disrupted", False)] or history
    if len(clean) < 5:
        weights = [0.55,0.20,0.10,0.08,0.07]
        recent  = [h.get("earnings",0) for h in clean[:5]]
        while len(recent) < 5: recent.append(recent[-1] if recent else 550)
        return {"predicted": round(sum(w*e for w,e in zip(weights,recent))*6, 2),
                "method": "weighted_moving_average"}
    avg = sum(h.get("earnings",0) for h in clean[-7:]) / len(clean[-7:])
    return {"predicted": round(avg*6,2), "method": "moving_average"}


def _make_user(base, plan):
    caps = {"basic":150.0,"standard":200.0,"premium":300.0}
    h = _gen_history(base)
    p = predict_weekly_earnings(h)
    return {"predicted_income":p["predicted"],"prediction_method":p["method"],
            "plan":plan,"weekly_cap":caps[plan],"active_policy":True,"earnings_history":h}


users         = {"raju":_make_user(633,"standard"),"priya":_make_user(700,"premium"),"demo":_make_user(500,"basic")}
claims        = []
weekly_totals = {}

# ══════════════════════════════════════════════════════════
#  GPS FRAUD DETECTION
#  Haversine = convert lat/lng degrees → km on Earth surface
#  Used to detect "teleportation" between GPS pings (fraud)
# ══════════════════════════════════════════════════════════

def haversine_km(la1,lo1,la2,lo2):
    R=6371.0; p1,p2=math.radians(la1),math.radians(la2)
    dp,dl=math.radians(la2-la1),math.radians(lo2-lo1)
    a=math.sin(dp/2)**2+math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return R*2*math.atan2(math.sqrt(a),math.sqrt(1-a))


def calculate_fraud_score(pts: List[GpsPoint]):
    score,flags,n = 0,[],len(pts)
    if n < 10: score+=30; flags.append("LOW_COVERAGE")
    if n>0:
        spoofed=sum(1 for p in pts if p.accuracy is not None and p.accuracy==0)
        if spoofed/n>0.8: score+=35; flags.append("GPS_SPOOFING")
    sv=[p for p in pts if p.speed is not None and p.speed>120]
    if sv: score+=min(20*len(sv),40); flags.append("IMPOSSIBLE_SPEED")
    if n>=2:
        jumps=sum(1 for i in range(1,n)
                  if haversine_km(pts[i-1].lat,pts[i-1].lng,pts[i].lat,pts[i].lng)>5.0)
        if jumps>0: score+=min(20*jumps,40)
        if "IMPOSSIBLE_SPEED" not in flags and jumps>0: flags.append("IMPOSSIBLE_SPEED")
    if n>=5:
        stat=sum(1 for i in range(1,n)
                 if abs(pts[i].lat-pts[i-1].lat)<0.00005 and abs(pts[i].lng-pts[i-1].lng)<0.00005)
        if stat/(n-1)>0.85: score+=25; flags.append("LOW_ACTIVITY")
    final=min(score,100)
    dec="APPROVE" if final<40 else "REVIEW" if final<70 else "REJECT"
    expl={"LOW_COVERAGE":"Insufficient GPS data points",
          "GPS_SPOOFING":"Coordinates appear artificially generated",
          "IMPOSSIBLE_SPEED":"Movement speed exceeds physical limits",
          "LOW_ACTIVITY":"Worker appears stationary"}
    return {"score":final,"flags":flags,"decision":dec,
            "explanation":[expl[f] for f in flags] or ["All GPS checks passed"],
            "summary":f"Fraud Score: {final}/100 — {dec}"}


def resolve_fraud(gps,movement):
    if gps and len(gps)>=1: return calculate_fraud_score(gps)
    if movement is True:
        return {"score":0,"flags":[],"decision":"APPROVE",
                "explanation":["Movement confirmed"],"summary":"0/100 — APPROVE"}
    return {"score":55,"flags":["LOW_COVERAGE","NO_MOVEMENT"],"decision":"REVIEW",
            "explanation":["No GPS or movement data"],"summary":"55/100 — REVIEW"}

# ══════════════════════════════════════════════════════════
#  PAYOUT FORMULA
#  0.5 factor = parametric covers 50% income replacement
#  Severity multiplier: flood/curfew = 1.5× (more disruptive)
# ══════════════════════════════════════════════════════════

def calculate_payout(weekly, overlap, etype):
    return round(0.5 * (weekly/6) * overlap * SEVERITY.get(etype,1.0), 2)

def detect_event(rain,aqi,flood):
    if flood: return "URBAN_FLOOD"
    if rain is not None and rain>=RAIN_THRESHOLD_MM: return "HEAVY_RAIN"
    if aqi is not None and aqi>=AQI_THRESHOLD: return "SEVERE_AQI"
    return None

def today_str(): return date.today().isoformat()

def validate_claim(username,event_type,working_hours,fraud):
    user=users.get(username)
    if not user or not user.get("active_policy"):
        return {"valid":False,"status":"rejected","message":"Rejected: No active policy"}
    if working_hours<MIN_WORKING_HOURS:
        return {"valid":False,"status":"rejected","message":f"Rejected: Minimum {MIN_WORKING_HOURS}h required"}
    if "NO_MOVEMENT" in fraud["flags"] or fraud["decision"]=="REJECT":
        return {"valid":False,"status":"rejected","message":"Rejected: Fraud check failed"}
    if any(c["username"]==username and c["event_type"]==event_type
           and c.get("claim_date")==today_str() for c in claims):
        return {"valid":False,"status":"duplicate","message":"Rejected: Duplicate claim today"}
    acc=weekly_totals.get(username,0.0)
    if acc>=user.get("weekly_cap",GLOBAL_WEEKLY_CAP):
        return {"valid":False,"status":"rejected","message":"Rejected: Weekly cap reached"}
    if fraud["decision"]=="REVIEW":
        return {"valid":True,"status":"review","message":"Under review — verified before Friday payout"}
    return {"valid":True,"status":"approved","message":""}

# ══════════════════════════════════════════════════════════
#  LLM EXPLAINER
#  Uses OpenAI GPT-3.5-turbo if OPENAI_API_KEY present.
#  Falls back to rule-based explanation automatically.
# ══════════════════════════════════════════════════════════

def _rule_explain(req: ExplainClaimRequest):
    label = DISRUPTION_LABELS.get(req.event_type, req.event_type)
    fnote = f"Fraud flags: {', '.join(req.fraud_flags)}. " if req.fraud_flags else "No fraud flags. "
    if req.status=="approved":
        return (f"Claim APPROVED for {req.username.title()}. {label} event triggered parametric threshold "
                f"(rain:{req.rain}mm, AQI:{req.aqi}). {fnote}"
                f"Payout ₹{req.payout_amount:.0f} = 50% of {req.working_hours}h shift income × severity multiplier.")
    elif req.status=="review":
        return f"Claim for {req.username.title()} is under REVIEW. {label}. {fnote}Manual check before Friday payout."
    return f"Claim REJECTED for {req.username.title()}. {label}. {fnote}Threshold or policy check failed."


def _llm_explain(req: ExplainClaimRequest):
    try:
        import openai
        key = os.getenv("OPENAI_API_KEY","")
        if not key: return _rule_explain(req)
        client = openai.OpenAI(api_key=key)
        label  = DISRUPTION_LABELS.get(req.event_type, req.event_type)
        prompt = (f"You are GigShield's insurance AI. Explain this claim decision in 3 clear sentences "
                  f"for a delivery partner. Be empathetic and avoid jargon.\n\n"
                  f"Worker:{req.username} | Event:{label} | Rain:{req.rain}mm AQI:{req.aqi} "
                  f"| Hours:{req.working_hours} | Fraud:{req.fraud_score}/100 flags:{req.fraud_flags} "
                  f"| Decision:{req.status.upper()} | Payout:₹{req.payout_amount:.0f}")
        r = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role":"user","content":prompt}],
            max_tokens=200, temperature=0.4)
        return r.choices[0].message.content.strip()
    except Exception:
        return _rule_explain(req)

# ══════════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════════

@app.post("/trigger")
def trigger_claim(req: TriggerRequest):
    username   = req.username.lower().strip()
    event_type = detect_event(req.rain, req.aqi, req.flood)
    if not event_type:
        raise HTTPException(400, "No threshold met. Need rain≥80, AQI≥300, or flood=true")

    fraud      = resolve_fraud(req.gps_points, req.movement)
    validation = validate_claim(username, event_type, req.working_hours, fraud)

    user           = users.get(username, {})
    predicted      = user.get("predicted_income", 3000.0)
    weekly_cap     = user.get("weekly_cap", GLOBAL_WEEKLY_CAP)
    timestamp      = datetime.now(timezone.utc).isoformat()
    overlap        = max(0.5, min(round(req.working_hours*0.6,1), 4.0))
    payout_raw     = calculate_payout(predicted, overlap, event_type)
    accumulated    = weekly_totals.get(username, 0.0)
    payout_capped  = round(min(payout_raw, weekly_cap-accumulated), 2)

    common = dict(username=username, event_type=event_type, working_hours=req.working_hours,
                  overlap_hours=overlap, raw_payout=payout_raw,
                  fraud_score=fraud["score"], fraud_flags=fraud["flags"],
                  fraud_decision=fraud["decision"], fraud_explanation=fraud["explanation"],
                  claim_date=today_str(), timestamp=timestamp)

    if not validation["valid"]:
        record = {**common, "payout_amount":0.0, "status":validation["status"], "message":validation["message"]}
        claims.append(record)
        return {"success":True,"claim":{**record,"event_label":DISRUPTION_LABELS.get(event_type,event_type)}}

    if validation["status"]=="review":
        record = {**common,"payout_amount":payout_capped,"status":"review",
                  "message":"Under review — verified before Friday payout"}
        claims.append(record)
        return {"success":True,"claim":{**record,"event_label":DISRUPTION_LABELS.get(event_type,event_type)}}

    record = {**common,"payout_amount":payout_capped,"status":"approved","message":"Claim approved. Paid Friday."}
    claims.append(record)
    weekly_totals[username] = accumulated + payout_capped
    return {"success":True,"claim":{**record,
        "event_label":DISRUPTION_LABELS.get(event_type,event_type),
        "weekly_total":round(weekly_totals[username],2),"weekly_cap":weekly_cap}}


@app.post("/ml/predict-disruption")
def ml_predict(req: MLPredictRequest):
    """ML model predicts disruption severity. See /ml/model-info for accuracy metrics."""
    X     = np.array([[req.rain_mm,req.aqi,req.temperature_c,
                        float(req.hour_of_day),float(req.day_of_week),float(req.flood_flag)]])
    pred  = int(_MODEL.predict(X)[0])
    proba = _MODEL.predict_proba(X)[0]
    return {"success":True,"prediction":{
        "severity_code":pred,"severity_label":_LABEL[pred],
        "confidence_pct":round(float(proba[pred])*100,1),
        "probabilities":{"NORMAL":round(float(proba[0])*100,1),
                         "MODERATE":round(float(proba[1])*100,1),
                         "SEVERE":round(float(proba[2])*100,1)},
        "payout_recommended": pred==2}}


@app.get("/ml/model-info")
def ml_model_info():
    """RandomForest training stats, accuracy, and feature importances."""
    return {"success":True,"model":{
        "algorithm":"RandomForestClassifier","n_estimators":120,
        "training_samples":1600,"test_samples":400,
        "accuracy_pct":_MODEL_ACC,"classes":["NORMAL","MODERATE","SEVERE"],
        "feature_importances":_FEAT_IMP},
        "note":"Trained on synthetic Indian weather data. Retrain monthly on real OpenWeatherMap data in production."}


@app.post("/api/explain-claim")
def explain_claim(req: ExplainClaimRequest):
    """Plain-English claim explanation. Uses OpenAI if OPENAI_API_KEY set, else rule-based fallback."""
    return {"success":True,
            "explanation":_llm_explain(req),
            "source":"openai" if os.getenv("OPENAI_API_KEY") else "rule_based_fallback",
            "claim_summary":{"username":req.username,
                             "event":DISRUPTION_LABELS.get(req.event_type,req.event_type),
                             "status":req.status,"payout":req.payout_amount}}


@app.post("/api/users/register")
def register_user(req: RegisterUserRequest):
    caps = {"basic":150.0,"standard":200.0,"premium":300.0}
    plan = req.plan.lower()
    if plan not in caps: raise HTTPException(400,"Invalid plan. Choose basic|standard|premium")
    h = _gen_history(req.base_daily_earnings or 633.0)
    p = predict_weekly_earnings(h)
    predicted = req.predicted_income or p["predicted"]
    users[req.username.lower()] = {"predicted_income":predicted,"prediction_method":p["method"],
        "plan":plan,"weekly_cap":caps[plan],"active_policy":True,"earnings_history":h}
    return {"success":True,"message":f"User {req.username} registered on {plan} plan.",
            "predicted_income":predicted,"prediction_method":p["method"],"weekly_cap":caps[plan]}


@app.get("/api/users/{username}")
def get_user(username:str):
    u=users.get(username.lower())
    if not u: raise HTTPException(404,"User not found")
    return {"success":True,"data":{k:v for k,v in u.items() if k!="earnings_history"}}

@app.get("/api/claims/{username}")
def get_claims_by_user(username:str):
    uc=[c for c in claims if c["username"]==username.lower()]
    return {"success":True,"username":username.lower(),"data":uc,
            "weekly_total":round(weekly_totals.get(username.lower(),0.0),2),"count":len(uc)}

@app.get("/api/claims")
def get_all_claims(): return {"success":True,"data":claims,"total":len(claims)}

@app.post("/api/analyze-gps")
def analyze_gps(points:List[GpsPoint]): return {"success":True,"data":calculate_fraud_score(points)}

@app.post("/api/predict-earnings")
def predict_earnings_ep(req:PredictRequest): return {"success":True,"data":predict_weekly_earnings(req.earnings_history)}

@app.post("/api/risk-score")
def risk_score(req:RiskScoreRequest):
    zr={"bandra":0.85,"andheri":0.75,"salt_lake":0.70,"t_nagar":0.60,
        "koramangala":0.45,"connaught_place":0.35,"sector_18":0.40,"jubilee_hills":0.30
        }.get(req.zone.lower().replace(" ","_"),0.50)
    vr={"bicycle":0.80,"motorcycle":0.60,"ev":0.55}.get(req.vehicle_type.lower(),0.60)
    base={"basic":20,"standard":30,"premium":50}.get(req.plan_name.lower(),30)
    return {"success":True,"data":{"base_premium":base,
        "dynamic_premium":round(base*(1+zr*0.3)*(1+vr*0.15),2),"zone_risk":zr,"vehicle_risk":vr}}

@app.get("/admin/stats")
def admin_stats():
    ap=[c for c in claims if c["status"]=="approved"]
    return {"totalUsers":len(users),"activeShifts":1,"totalClaims":len(claims),
            "pendingReview":len([c for c in claims if c["status"]=="review"]),
            "totalPaid":round(sum(c["payout_amount"] for c in ap),2)}

@app.get("/admin/users")
def admin_users():
    return [{"id":i+1,"phone":u,"name":u.title(),"city":"Mumbai","vehicleType":"Motorcycle",
             "kycDone":True,"plan":d.get("plan"),"predicted_income":d.get("predicted_income")}
            for i,(u,d) in enumerate(users.items())]

@app.get("/admin/shifts")
def admin_shifts():
    return [{"id":1,"userId":"raju","startTime":datetime.now(timezone.utc).isoformat(),
             "endTime":None,"totalActiveMinutes":240,"active":True}]

@app.get("/admin/claims")
def admin_claims():
    return [{"id":i+1,"userId":c["username"],"disruptionType":c["event_type"],
             "payoutAmount":c["payout_amount"],"fraudScore":c["fraud_score"],
             "status":c["status"].upper(),"createdAt":c["timestamp"]} for i,c in enumerate(claims)]

@app.post("/admin/claims/{claim_id}/approve")
def admin_approve(claim_id:int):
    idx=claim_id-1
    if idx<0 or idx>=len(claims): raise HTTPException(404,"Claim not found")
    claims[idx]["status"]="approved"
    weekly_totals[claims[idx]["username"]]=weekly_totals.get(claims[idx]["username"],0.0)+claims[idx]["payout_amount"]
    return {"success":True,"message":"Claim approved"}

@app.post("/admin/claims/{claim_id}/reject")
def admin_reject(claim_id:int, body:AdminClaimAction):
    idx=claim_id-1
    if idx<0 or idx>=len(claims): raise HTTPException(404,"Claim not found")
    claims[idx]["status"]="rejected"; claims[idx]["message"]=f"Rejected: {body.reason}"
    return {"success":True,"message":"Claim rejected"}

@app.post("/admin/simulate-trigger")
def simulate_trigger(req:AdminTriggerRequest):
    triggered=0
    for username,user in list(users.items()):
        if not user.get("active_policy"): continue
        fraud={"score":0,"flags":[],"decision":"APPROVE","explanation":["Simulation"],"summary":"APPROVE"}
        v=validate_claim(username,req.type,7,fraud)
        p=calculate_payout(user.get("predicted_income",3000.0),1.5,req.type)
        acc=weekly_totals.get(username,0.0)
        capped=round(min(p,user.get("weekly_cap",GLOBAL_WEEKLY_CAP)-acc),2)
        status=v["status"] if not v["valid"] else "approved"
        record={"username":username,"event_type":req.type,
                "payout_amount":capped if status=="approved" else 0.0,
                "status":status,"message":v["message"] if not v["valid"] else "Claim approved. Paid Friday.",
                "fraud_score":0,"fraud_flags":[],"fraud_decision":"APPROVE",
                "fraud_explanation":["Simulation"],"working_hours":7,
                "claim_date":today_str(),"timestamp":datetime.now(timezone.utc).isoformat()}
        claims.append(record)
        if status=="approved": weekly_totals[username]=acc+capped; triggered+=1
    return {"success":True,"message":f"Trigger fired for {triggered} workers."}

@app.post("/api/weekly-reset")
def weekly_reset():
    weekly_totals.clear(); claims.clear()
    return {"success":True,"message":"Weekly reset complete."}

@app.get("/health")
def health():
    return {"status":"ok","service":"GigShield AI Service","version":"4.0.0",
            "registered_users":len(users),"total_claims":len(claims),
            "ml_model_accuracy_pct":_MODEL_ACC}
