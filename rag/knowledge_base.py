"""
Clinical Knowledge Base
=======================
Structured clinical guidelines, protocols, and drug information
ingested into FAISS for RAG-powered clinical decision support.

Sources modeled after:
- ACC/AHA Guidelines
- ADA Standards of Care
- KDIGO CKD Guidelines
- JNC 8 Hypertension Guidelines
- BNF Drug Information
"""

CLINICAL_GUIDELINES = [
    {
        "id": "GL001",
        "title": "ADA Standards of Care — Type 2 Diabetes HbA1c Management",
        "category": "diabetes",
        "icd10": ["E11.9", "E11.65"],
        "content": """
American Diabetes Association Standards of Medical Care — HbA1c Targets:

GLYCEMIC TARGETS:
- General HbA1c target: <7.0% for most non-pregnant adults with T2DM
- Less stringent target (<8.0%): elderly patients, multiple comorbidities, limited life expectancy, hypoglycemia unawareness, long-standing diabetes
- More stringent target (<6.5%): if safely achievable without hypoglycemia, young patients, newly diagnosed

FIRST-LINE THERAPY:
- Metformin remains first-line pharmacological therapy for T2DM if tolerated and not contraindicated
- Contraindications: eGFR <30 mL/min (discontinue), hepatic impairment, excessive alcohol use
- Dose: start 500mg daily with meals, titrate to 2000-2550mg/day as tolerated

SECOND-LINE ADDITIONS (when HbA1c remains above target):
1. SGLT2 Inhibitors (empagliflozin, canagliflozin, dapagliflozin):
   - Preferred in T2DM with established CVD, heart failure, or CKD (eGFR 20-45)
   - Cardiorenal protection beyond glucose lowering
   - Contraindicated: eGFR <20 mL/min
2. GLP-1 Receptor Agonists (semaglutide, liraglutide):
   - Preferred with established CVD or high cardiovascular risk
   - Benefits: weight loss, CV risk reduction
3. DPP-4 Inhibitors: dose-safe in CKD, weight-neutral
4. Sulfonylureas: low cost, hypoglycemia risk, weight gain
5. Insulin: when HbA1c >10%, marked hyperglycemia, or other therapies fail

MONITORING:
- HbA1c: every 3 months until stable, then every 6 months
- Fasting glucose: daily self-monitoring if on insulin
- Annual: lipid panel, urine albumin/creatinine ratio, eGFR, eye exam, foot exam
- Blood pressure target: <130/80 mmHg

DIABETES + CKD CONSIDERATIONS:
- Metformin: use with caution eGFR 30-45, discontinue if eGFR <30
- SGLT2i: cardiorenal protection, continue if eGFR ≥20
- Avoid nephrotoxic agents; adjust doses for eGFR
- ACE inhibitor or ARB for proteinuria regardless of BP
""",
    },
    {
        "id": "GL002",
        "title": "ACC/AHA Hypertension Guideline — Blood Pressure Management",
        "category": "hypertension",
        "icd10": ["I10"],
        "content": """
ACC/AHA 2017 Hypertension Clinical Practice Guidelines:

BLOOD PRESSURE CLASSIFICATION:
- Normal: SBP <120 AND DBP <80 mmHg
- Elevated: SBP 120-129 AND DBP <80
- Stage 1 HTN: SBP 130-139 OR DBP 80-89
- Stage 2 HTN: SBP ≥140 OR DBP ≥90
- Hypertensive crisis: SBP >180 and/or DBP >120

TREATMENT THRESHOLDS:
- Stage 1 HTN + low CVD risk: lifestyle modification first
- Stage 1 HTN + 10-yr CVD risk ≥10%: lifestyle + medication
- Stage 2 HTN: lifestyle + medication (2-drug combination often needed)

BP TARGETS:
- General: <130/80 mmHg
- Older adults (≥65): <130 mmHg systolic (if tolerated)
- CKD without proteinuria: <130/80
- CKD with proteinuria: <130/80 with ACEi or ARB preferred
- Diabetes: <130/80 mmHg
- Post-stroke: <130/80 mmHg

FIRST-LINE MEDICATIONS:
1. Thiazide/thiazide-like diuretics (chlorthalidone preferred over HCTZ)
2. ACE inhibitors (lisinopril, enalapril) — contraindicated in pregnancy
3. ARBs (losartan, valsartan) — use when ACEi not tolerated; avoid in pregnancy
4. Calcium channel blockers (amlodipine, nifedipine ER)

SPECIAL POPULATIONS:
- CKD with proteinuria: ACEi or ARB (mandatory)
- Heart failure with reduced EF: ACEi/ARB + beta-blocker + MRA
- Post-MI: ACEi + beta-blocker
- Diabetes: ACEi or ARB preferred
- Black patients: thiazide or CCB preferred (ACEi less effective monotherapy)
- Pregnancy: methyldopa, labetalol, nifedipine (avoid ACEi/ARB)

LIFESTYLE MODIFICATIONS (all patients):
- DASH diet: targets SBP reduction 8-14 mmHg
- Sodium reduction <1500 mg/day: 2-8 mmHg reduction
- Weight loss (1 kg): ~1 mmHg reduction
- Physical activity 90-150 min/week: 5-8 mmHg reduction
- Limit alcohol: <2 drinks/day men, <1 drink/day women
""",
    },
    {
        "id": "GL003",
        "title": "KDIGO Clinical Practice Guideline — Chronic Kidney Disease",
        "category": "nephrology",
        "icd10": ["N18.3", "N18.4", "N18.5"],
        "content": """
KDIGO 2022 CKD Clinical Practice Guideline:

CKD STAGING BY eGFR:
- G1: eGFR ≥90 mL/min — normal or high (if markers of kidney damage present)
- G2: eGFR 60-89 — mildly decreased
- G3a: eGFR 45-59 — mildly to moderately decreased
- G3b: eGFR 30-44 — moderately to severely decreased
- G4: eGFR 15-29 — severely decreased (ESRD planning begins)
- G5: eGFR <15 — kidney failure (dialysis/transplant)

ALBUMINURIA CATEGORIES:
- A1: <30 mg/g — normal to mildly increased
- A2: 30-300 mg/g — moderately increased (microalbuminuria)
- A3: >300 mg/g — severely increased (macroalbuminuria)

MANAGEMENT BY STAGE:
- All stages: BP control <130/80, ACEi or ARB if proteinuria, avoid NSAIDs
- G3b-G4: nephrology referral; SGLT2 inhibitor (if T2DM, eGFR ≥20)
- G4: prepare for renal replacement therapy (dialysis/transplant)
- G5: initiate renal replacement therapy

MEDICATION ADJUSTMENTS IN CKD:
- Metformin: OK eGFR 30-60 (reduce dose); STOP if eGFR <30
- SGLT2i: continue if eGFR ≥20 (cardiorenal protection)
- ACEi/ARB: continue unless K+ >5.5 or SCr rises >30% from baseline
- Avoid: NSAIDs, contrast dye without hydration, nephrotoxins
- Dose adjust: many antibiotics, anticoagulants, digoxin, gabapentin

HYPERKALEMIA MANAGEMENT IN CKD:
- K+ >5.0: reduce dietary potassium intake; review medications (ACEi/ARB)
- K+ >5.5: consider potassium binders (patiromer, sodium zirconium cyclosilicate)
- K+ >6.0: urgent treatment; may need to hold ACEi/ARB; IV calcium if ECG changes
- K+ >6.5 with ECG changes: EMERGENCY — kayexalate, insulin/glucose, calcium gluconate

ANEMIA IN CKD:
- Hemoglobin target: 10-11.5 g/dL with ESA therapy
- Iron: replete first; Hb <10 g/dL + iron deficiency → IV iron in dialysis patients
- ESA: initiate if Hb <10 g/dL, avoid if active malignancy

DIALYSIS INDICATIONS:
- Uremic symptoms: encephalopathy, pericarditis, nausea/vomiting refractory
- Fluid overload refractory to diuretics
- Metabolic acidosis refractory
- Hyperkalemia refractory
- eGFR <6-10 mL/min
""",
    },
    {
        "id": "GL004",
        "title": "AHA/ACC Heart Failure Guideline — HFrEF Management",
        "category": "cardiology",
        "icd10": ["I50.9", "I50.2"],
        "content": """
2022 AHA/ACC/HFSA Heart Failure Guideline:

CLASSIFICATION:
- HFrEF: EF ≤40% — reduced ejection fraction
- HFmrEF: EF 41-49% — mildly reduced
- HFpEF: EF ≥50% — preserved EF

FOUNDATIONAL THERAPY FOR HFrEF (the "Fantastic Four"):
1. ACE inhibitor / ARB / ARNI (sacubitril-valsartan):
   - ARNI preferred over ACEi/ARB (reduces hospitalizations/mortality)
   - Start low, titrate to target dose
   - Monitor: K+, creatinine, BP
2. Beta-blocker (carvedilol, metoprolol succinate, bisoprolol):
   - Start low, titrate slowly
   - Do NOT initiate in acute decompensation
3. Mineralocorticoid receptor antagonist (spironolactone, eplerenone):
   - Avoid if K+ >5.0 or eGFR <30
4. SGLT2 inhibitor (dapagliflozin, empagliflozin):
   - Reduces HF hospitalizations regardless of diabetes status
   - New Class I recommendation regardless of EF

DIURETICS:
- Loop diuretics (furosemide, torsemide) for fluid overload
- Adjust dose to achieve euvolemia (dry weight)
- Monitor: K+, Mg, renal function, daily weights
- Torsemide preferred over furosemide (better bioavailability)

ACUTE DECOMPENSATED HEART FAILURE:
- IV loop diuretics: 2.5x oral dose IV, or continuous infusion
- Monitor urine output, electrolytes, renal function daily
- Assess decongestion: symptoms, weight, BNP/NT-proBNP
- Consider ultrafiltration if diuretic resistance
- Avoid: NSAIDs, non-DHP CCBs, most antiarrhythmics

MONITORING:
- Daily weights at home (alert if >2-3 lb gain in 1-2 days)
- Sodium restriction: <2g/day
- Fluid restriction: 1.5-2L/day if hyponatremia
- Device therapy: ICD if EF ≤35% on optimal medical therapy ≥3 months
""",
    },
    {
        "id": "GL005",
        "title": "Drug Reference — Common Cardiac and Metabolic Medications",
        "category": "pharmacology",
        "icd10": [],
        "content": """
DRUG REFERENCE — COMMON MEDICATIONS:

METFORMIN (Glucophage):
- Class: Biguanide antidiabetic
- Mechanism: Decreases hepatic glucose production, improves insulin sensitivity
- Dose: 500-2550mg/day in divided doses with meals
- Contraindications: eGFR <30, hepatic failure, excessive alcohol, iodinated contrast (hold 48h)
- Side effects: GI (nausea, diarrhea — take with food), lactic acidosis (rare)
- Monitoring: eGFR annually, more frequently if declining

LISINOPRIL (Zestril, Prinivil) — ACE Inhibitor:
- Class: ACE inhibitor
- Mechanism: Blocks conversion of Ang I → Ang II; reduces SVR and preload
- Dose: HTN 5-40mg daily; HF 2.5-40mg daily; start low, titrate
- Contraindications: pregnancy, bilateral RAS, angioedema history, K+ >5.5
- Side effects: dry cough (10-15%), hyperkalemia, AKI (especially with NSAID), angioedema
- Monitoring: K+, creatinine (expect 10-20% rise acceptable), BP

FUROSEMIDE (Lasix) — Loop Diuretic:
- Class: Loop diuretic
- Mechanism: Blocks Na-K-2Cl transporter in ascending loop of Henle
- Dose: 20-600mg/day oral or IV; IV dose = 2-2.5x oral dose
- Onset: oral 30-60min; IV 5min
- Contraindications: anuria, severe electrolyte depletion
- Side effects: hypokalemia, hypomagnesemia, hyponatremia, ototoxicity (high doses), prerenal AKI
- Monitoring: electrolytes, creatinine, daily weight, I&O

ATORVASTATIN (Lipitor) — Statin:
- Class: HMG-CoA reductase inhibitor
- Mechanism: Blocks cholesterol synthesis in liver
- Dose: 10-80mg nightly
- Indications: hyperlipidemia, ASCVD risk reduction
- Contraindications: active liver disease, pregnancy
- Side effects: myopathy (rare), hepatotoxicity (rare), rhabdomyolysis (very rare)
- Monitoring: LFTs at baseline, CK if myalgia; lipid panel 4-12 weeks after start

CARVEDILOL (Coreg) — Beta-Blocker (non-selective):
- Class: Alpha-1/beta-1/beta-2 blocker
- Mechanism: Reduces HR and contractility; reduces afterload via alpha blockade
- Dose: HF: 3.125mg BID → titrate to 25mg BID over weeks; HTN: 6.25-25mg BID
- Contraindications: acute decompensated HF, severe bradycardia, 2nd/3rd degree AV block, severe COPD
- Side effects: bradycardia, hypotension, fatigue, dizziness
- Monitoring: HR, BP, signs of decompensation during titration

KAYEXALATE (Sodium Polystyrene Sulfonate):
- Class: Cation-exchange resin
- Use: Hyperkalemia (K+ >5.5 mEq/L)
- Dose: 15-60g oral or rectal; 30g oral with 50mL sorbitol
- Onset: 2-6 hours
- Cautions: intestinal necrosis risk (especially post-op); avoid with sorbitol in high-risk
- Alternative: Patiromer (Veltassa) — safer, better tolerated; Sodium zirconium cyclosilicate (Lokelma)
""",
    },
    {
        "id": "GL006",
        "title": "Clinical Decision Support — Alert Thresholds and Critical Values",
        "category": "clinical_decision_support",
        "icd10": [],
        "content": """
CRITICAL LAB VALUES REQUIRING IMMEDIATE ACTION:

CHEMISTRY:
- Potassium K+ >6.0 mEq/L: CRITICAL — risk of fatal arrhythmia; immediate treatment
- Potassium K+ <2.5 mEq/L: CRITICAL — arrhythmia risk; IV replacement
- Sodium Na+ <120 or >160 mEq/L: CRITICAL — neurological emergency
- Glucose <40 or >600 mg/dL: CRITICAL
- Creatinine >10 mg/dL (new): CRITICAL — dialysis may be needed
- pH <7.2 or >7.6: CRITICAL — acid-base emergency
- Lactate >4 mmol/L: CRITICAL — sepsis/shock

HEMATOLOGY:
- Hemoglobin <7 g/dL: transfusion threshold (symptomatic or active bleeding)
- Hemoglobin <6 g/dL: CRITICAL — urgent transfusion
- WBC >30 or <2 x10^3/uL: hematology consult
- Platelets <20,000: CRITICAL — bleeding risk
- INR >4.0: hold anticoagulation; consider reversal

VITAL SIGNS ALERTS:
- SBP <90 or >180 mmHg: notify MD immediately
- HR <50 or >130 bpm: assess and notify
- SpO2 <90%: supplemental O2, assess airway
- Temperature >39.5°C (103.1°F): sepsis workup
- RR >30/min: respiratory distress protocol

DRUG INTERACTIONS — HIGH ALERT:
- ACEi + ARB: avoid combination (renal failure risk)
- ACEi/ARB + K-sparing diuretics + CKD: hyperkalemia
- Warfarin + many antibiotics: increased INR
- Metformin + contrast dye: hold 48h pre/post
- NSAIDs + ACEi + diuretics: "triple whammy" — AKI risk
- Digoxin + amiodarone: digoxin toxicity
- Fluoroquinolones + QT-prolonging drugs: Torsades risk
""",
    },
    {
        "id": "GL007",
        "title": "Asthma Management — NAEPP / GINA Guidelines",
        "category": "pulmonology",
        "icd10": ["J45.30", "J45.40", "J45.50"],
        "content": """
NAEPP EPR-3 / GINA 2023 Asthma Management Guidelines:

CLASSIFICATION (severity / control):
- Intermittent: symptoms ≤2 days/week, nighttime ≤2x/month, FEV1 ≥80%
- Mild persistent: >2 days/week but not daily, nighttime 3-4x/month, FEV1 ≥80%
- Moderate persistent: daily symptoms, nighttime >1x/week, FEV1 60-80%
- Severe persistent: throughout the day, frequent nighttime, FEV1 <60%

STEP THERAPY:
- Step 1 (intermittent): SABA (albuterol) PRN only
- Step 2 (mild persistent): low-dose ICS (fluticasone 88mcg BID) + SABA PRN
- Step 3: low-dose ICS + LABA (formoterol/salmeterol) or medium ICS
- Step 4: medium ICS + LABA
- Step 5: high-dose ICS + LABA + consider biologics (dupilumab, omalizumab)
- Step 6: step 5 + oral corticosteroids (last resort)

RESCUE THERAPY:
- Albuterol (SABA): 90mcg/puff, 2 puffs q4-6h PRN
- Levalbuterol alternative
- Using SABA >2x/week indicates inadequate control → step up therapy

MONITORING:
- Asthma Control Test (ACT) score at every visit
- Spirometry annually
- Peak flow monitoring for poorly controlled or severe
- Reassess and consider step down after 3 months of good control

SPECIAL CONSIDERATIONS:
- Aspirin-exacerbated respiratory disease: avoid NSAIDs
- Exercise-induced: pre-treatment with SABA 15min before exercise
- Pregnancy: continue ICS therapy (budesonide preferred data)
- Comorbidities: GERD, allergic rhinitis, obesity — treat to improve asthma control
""",
    },
]


CLINICAL_PROTOCOLS = [
    {
        "id": "PR001",
        "title": "Nursing Protocol — Fall Risk Assessment (Morse Fall Scale)",
        "category": "nursing",
        "content": """
MORSE FALL SCALE PROTOCOL:

RISK FACTORS AND SCORING:
1. History of falling (within 3 months): 25 points
2. Secondary diagnosis: 15 points
3. Ambulatory aid: furniture=30, crutches/cane/walker=15, none/bed rest=0
4. IV/heparin lock: 20 points
5. Gait: impaired=20, weak=10, normal/bedrest=0
6. Mental status: forgets limitations=15, oriented=0

RISK LEVELS:
- Low risk: 0-24 points → standard precautions
- Medium risk: 25-44 → fall precautions protocol
- High risk: ≥45 → intensive fall precautions

HIGH-RISK INTERVENTIONS:
- Yellow armband and door signage
- Bed in lowest position, wheels locked
- Call light within reach
- Non-slip footwear
- Hourly rounding
- Bed alarm activated
- Bathroom accompanied at all times
""",
    },
    {
        "id": "PR002",
        "title": "Protocol — Clinical Summary Documentation Standards",
        "category": "documentation",
        "content": """
CLINICAL SUMMARY DOCUMENTATION STANDARDS (ASCENSION):

SOAP NOTE FORMAT:
S — Subjective: Patient complaints, symptoms, history in patient's own words
O — Objective: Vitals, physical exam findings, lab results, imaging
A — Assessment: Differential diagnoses, clinical impressions, problem list
P — Plan: Treatment plan, medications, referrals, patient education, follow-up

REQUIRED ELEMENTS FOR DISCHARGE SUMMARY:
1. Admission diagnosis and reason
2. Hospital course (chronological)
3. Procedures performed
4. Final diagnosis (primary + secondary)
5. Condition at discharge
6. Discharge medications (reconciled list)
7. Follow-up appointments (who, when, what for)
8. Patient instructions
9. Pending test results

AI-GENERATED DRAFT POLICY:
- AI drafts require physician review and attestation before finalization
- Any AI-generated content must be marked [AI-DRAFT] until signed
- Physician is responsible for accuracy of all content
- HIPAA: AI tools must access only data needed for the specific patient encounter
""",
    },
]
