"""Remote chronic-care monitoring: critical-value rules, personal-baseline trend alerts, missed-reading reminders.

Lab prototype on SYNTHETIC data only (TRL 4). Thresholds are illustrative; clinical governance must set them.
Alerts route to the physician (critical) or the community health worker / behvarz (trend, reminders).
"""
import numpy as np

N_PATIENTS, DAYS = 60, 120
BASE_DAYS = 14

CRITERIA = [
    ("critical_recall", ">=", 1.0, "critical readings that raise an immediate physician alert"),
    ("deterioration_recall", ">=", 0.80, "deteriorations detected within 14 days of onset (ramp lasts 15 days)"),
    ("trend_false_alerts_per_patient_month", "<=", 0.30, "trend alerts outside deterioration, per patient-month"),
]


def critical(glucose=None, sbp=None, dbp=None):
    r = []
    if glucose is not None:
        if glucose < 54 or glucose > 400:
            r.append("glucose %.0f severe" % glucose)
        elif glucose < 70:
            r.append("glucose %.0f low" % glucose)
        elif glucose > 300:
            r.append("glucose %.0f very high" % glucose)
    if sbp is not None and dbp is not None and (sbp >= 180 or dbp >= 120):
        r.append("BP %d/%d crisis range" % (sbp, dbp))
    return r


def simulate(seed=0):
    rng = np.random.default_rng(seed)
    pts = []
    for i in range(N_PATIENTS):
        g0 = rng.normal(130, 18); s0 = rng.normal(130, 10); sd_g = rng.uniform(8, 16); sd_s = rng.uniform(4, 8)
        g = g0 + rng.normal(0, sd_g, DAYS); s = s0 + rng.normal(0, sd_s, DAYS)
        det = None
        if rng.random() < 0.3:
            start = int(rng.integers(30, DAYS - 25)); det = {"start": start, "kind": "glucose" if rng.random() < 0.5 else "bp"}
            ramp = np.clip(np.arange(DAYS) - start, 0, 15)
            if det["kind"] == "glucose":
                g = g + ramp * 4.0
            else:
                s = s + ramp * 1.8
        crit = []
        for d in rng.choice(np.arange(20, DAYS), 2, replace=False) if rng.random() < 0.25 else []:
            if rng.random() < 0.5:
                g[d] = rng.choice([45.0, 420.0])
            else:
                s[d] = 195.0
            crit.append(int(d))
        d_ = s * 0.62 + rng.normal(0, 3, DAYS)
        miss = rng.random(DAYS) < 0.12
        miss[crit] = False
        pts.append({"id": i, "glucose": g, "sbp": s, "dbp": d_, "missing": miss, "deterioration": det, "critical_days": crit})
    return pts


def trend_alerts(values, missing, lam=0.3, k=3.5, consec=3, base_days=BASE_DAYS):
    """EWMA of daily value vs the patient's own first-weeks baseline; alert after `consec` consecutive exceedances."""
    ok = ~missing[:base_days]
    base = values[:base_days][ok]
    mu = np.median(base); sd = max(1.4826 * np.median(np.abs(base - mu)), 1.0)
    z, run, alerts = mu, 0, []
    ew_sd = sd * np.sqrt(lam / (2 - lam))
    for t in range(base_days, len(values)):
        if missing[t]:
            continue
        z = lam * values[t] + (1 - lam) * z
        run = run + 1 if (z - mu) > k * ew_sd else 0
        if run == consec:
            alerts.append(t)
    return alerts


def missed_reading_reminders(missing, max_gap=3):
    out, run = [], 0
    for t, m in enumerate(missing):
        run = run + 1 if m else 0
        if run == max_gap:
            out.append(t)
    return out


def route(alert_kind):
    return {"critical": "physician (immediate)", "trend": "behvarz / health worker (same day)", "missed": "automated reminder"}[alert_kind]


def evaluate(seed=0):
    pts = simulate(seed)
    crit_total = crit_hit = 0
    det_total = det_hit = 0
    fa = 0.0; patient_months = 0.0
    for p in pts:
        for d in p["critical_days"]:
            crit_total += 1; crit_hit += bool(critical(p["glucose"][d], p["sbp"][d], p["dbp"][d]))
        ta = {"glucose": trend_alerts(p["glucose"], p["missing"]), "bp": trend_alerts(p["sbp"], p["missing"])}
        det = p["deterioration"]
        if det:
            det_total += 1
            det_hit += any(det["start"] <= t <= det["start"] + 14 for t in ta[det["kind"]])
        for kind, al in ta.items():
            for t in al:
                if det is None or kind != det["kind"] or t < det["start"] or t > det["start"] + 25:
                    fa += 1
        patient_months += (DAYS - BASE_DAYS) / 30
    return {"critical_recall": crit_hit / max(crit_total, 1), "deterioration_recall": det_hit / max(det_total, 1),
            "trend_false_alerts_per_patient_month": fa / patient_months, "n_deteriorations": det_total}


def generate(seed=0):
    pts = simulate(seed)
    rows = []
    for p in pts:
        for d in range(DAYS):
            if p["missing"][d]:
                continue
            rows.append(dict(patient="PT%03d" % p["id"], day=d, glucose=round(float(p["glucose"][d]), 1), sbp=round(float(p["sbp"][d]), 1), dbp=round(float(p["dbp"][d]), 1)))
    return {"readings": rows, "deteriorations": [dict(patient="PT%03d" % p["id"], onset_day=p["deterioration"]["start"], kind=p["deterioration"]["kind"]) for p in pts if p["deterioration"]]}


def demo(seed=0):
    p = next(x for x in simulate(seed) if x["deterioration"])
    det = p["deterioration"]
    ta = trend_alerts(p["glucose" if det["kind"] == "glucose" else "sbp"], p["missing"])
    return "Patient %d: onset day %d (%s); first trend alert day %s -> %s" % (p["id"], det["start"], det["kind"], ta[0] if ta else None, route("trend"))
