# Lake Windermere Bloom Prediction Project

## Project Overview

This project develops a probabilistic machine learning system to predict bloom events in **Lake Windermere**.  

The objective is to transition from reactive environmental monitoring to proactive, risk-informed decision-making by providing early-warning bloom risk estimates.

The system is designed as a **decision-support pilot**, not a deterministic forecasting tool.

---

# 1. Business Framing

## 1.1 Business Problem

Lake Windermere experiences periodic phytoplankton and potential harmful algal bloom (HAB) events that:

- Impact ecological status  
- Create operational challenges for water abstraction and monitoring  
- Increase regulatory and reputational risk  
- Require rapid advisory and management decisions  

Current responses are largely **reactive**, triggered after visible bloom formation or sampling confirmation.

Stakeholders lack a **quantitative, forward-looking risk signal** that provides actionable lead time before bloom manifestation.

---

## 1.2 Business Solution

### Bloom Risk Intelligence (Pilot)

A decision-support system providing:

- A rolling **7-day bloom probability score (0–1)**
- Risk tier classification (Low / Medium / High)
- Early-warning alerts when risk exceeds threshold
- Transparent indication of key drivers

The goal is to improve:

- Monitoring targeting
- Operational preparedness
- Advisory confidence
- Environmental governance

---

## 1.3 Business Indicators

### Early Warning Performance
- Median lead time before confirmed bloom (target ≥ 5 days)
- % of bloom events preceded by alert

### Operational Efficiency
- % increase in targeted sampling during high-risk periods
- Reduction in unnecessary high-intensity monitoring during low-risk periods

### Alert Management
- False alerts per bloom season
- Alerts per confirmed bloom event

### Adoption & Governance
- % of seasonal weeks system is consulted
- Evidence of decisions referencing system output

---

# 2. Machine Learning Framing

## 2.1 ML Problem / Task

**Binary Classification Task**

> Predict whether a bloom event will occur within the next 7 days.

### Output:
- Probability of bloom occurrence
- Risk tier derived from threshold selection

---

## 2.2 Bloom Event Definition (To Be Finalised)

The label must be objectively defined using one of:

- Chlorophyll-a threshold exceedance  
- Cyanobacteria cell count exceedance  
- Regulatory alert proxy  
- Scientifically validated composite metric  

Clear label definition is critical to model validity.

---

## 2.3 ML Solution Approach

### Inputs
- Historical water quality data
- Meteorological drivers
- Hydrological indicators
- Seasonal encodings
- Lagged and rolling statistics (EDA-informed features)

### Modelling Strategy
Phase 1: Logistic Regression (interpretable baseline)  
Phase 2: Random Forest / Gradient Boosting  
Time-aware cross-validation mandatory  

### Output
- Calibrated probability
- Threshold-based risk classification

---

## 2.4 ML Evaluation Indicators

### Core Metrics (Primary)
- PR AUC (preferred for rare events)
- Recall at operational threshold
- F2 score (recall-weighted)
- Brier score (probability calibration)

### Secondary Metrics
- ROC AUC
- Precision
- Event detection rate
- Median lead time

---

# Project Positioning

This project is positioned as:

✔ A pilot environmental intelligence demonstrator  
✔ A governance-aware applied ML system  
✔ A portfolio-grade environmental data science project  

It is not positioned as:

✘ A guaranteed bloom prevention tool  
✘ A fully commercialised SaaS product  

---

End of Document
