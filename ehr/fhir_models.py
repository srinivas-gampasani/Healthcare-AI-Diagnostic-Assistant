"""
HL7 FHIR EHR Integration
========================
Provides:
- FHIR R4 resource models (Patient, Observation, Condition, MedicationRequest, etc.)
- Mock EHR database with realistic synthetic patient data
- FHIR-compliant data access layer
- CDS Hooks-style clinical decision support integration

All patient data is SYNTHETIC — no real PHI.
"""
import json
import random
from datetime import datetime, date, timedelta
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field, asdict
from enum import Enum


# ── FHIR Resource Models ───────────────────────────────────────────────────────

class Gender(str, Enum):
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"
    UNKNOWN = "unknown"


@dataclass
class FHIRCoding:
    system: str
    code: str
    display: str


@dataclass
class FHIRPatient:
    """HL7 FHIR R4 Patient resource."""
    id: str
    mrn: str
    first_name: str
    last_name: str
    date_of_birth: str        # YYYY-MM-DD
    gender: str
    phone: str
    email: str
    address: str
    blood_type: str
    allergies: List[str]
    primary_physician: str
    insurance_id: str
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_fhir_dict(self) -> Dict:
        """Return FHIR R4 Patient resource JSON."""
        return {
            "resourceType": "Patient",
            "id": self.id,
            "identifier": [
                {"system": "http://hospital.example.org/mrn", "value": self.mrn}
            ],
            "name": [{"family": self.last_name, "given": [self.first_name]}],
            "gender": self.gender,
            "birthDate": self.date_of_birth,
            "telecom": [
                {"system": "phone", "value": self.phone},
                {"system": "email", "value": self.email},
            ],
            "address": [{"text": self.address}],
        }

    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}"

    @property
    def age(self) -> int:
        dob = date.fromisoformat(self.date_of_birth)
        today = date.today()
        return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))


@dataclass
class FHIRObservation:
    """HL7 FHIR R4 Observation (lab result, vital sign)."""
    id: str
    patient_id: str
    category: str             # vital-signs | laboratory | imaging
    code: str                 # LOINC code
    display: str              # Human-readable name
    value: float
    unit: str
    reference_range_low: Optional[float]
    reference_range_high: Optional[float]
    status: str               # final | preliminary | amended
    effective_date: str
    interpretation: str       # normal | high | low | critical-high | critical-low

    @property
    def is_abnormal(self) -> bool:
        if self.reference_range_low and self.value < self.reference_range_low:
            return True
        if self.reference_range_high and self.value > self.reference_range_high:
            return True
        return False

    def to_fhir_dict(self) -> Dict:
        return {
            "resourceType": "Observation",
            "id": self.id,
            "status": self.status,
            "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category",
                                       "code": self.category}]}],
            "code": {"coding": [{"system": "http://loinc.org", "code": self.code, "display": self.display}]},
            "subject": {"reference": f"Patient/{self.patient_id}"},
            "effectiveDateTime": self.effective_date,
            "valueQuantity": {"value": self.value, "unit": self.unit},
            "interpretation": [{"coding": [{"code": self.interpretation}]}],
        }


@dataclass
class FHIRCondition:
    """HL7 FHIR R4 Condition (diagnosis)."""
    id: str
    patient_id: str
    icd10_code: str
    display: str
    clinical_status: str      # active | resolved | recurrence
    onset_date: str
    recorded_date: str
    severity: str             # mild | moderate | severe
    notes: str = ""

    def to_fhir_dict(self) -> Dict:
        return {
            "resourceType": "Condition",
            "id": self.id,
            "clinicalStatus": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
                                            "code": self.clinical_status}]},
            "code": {"coding": [{"system": "http://hl7.org/fhir/sid/icd-10", "code": self.icd10_code,
                                  "display": self.display}]},
            "subject": {"reference": f"Patient/{self.patient_id}"},
            "onsetDateTime": self.onset_date,
        }


@dataclass
class FHIRMedication:
    """HL7 FHIR R4 MedicationRequest."""
    id: str
    patient_id: str
    medication_name: str
    rxnorm_code: str
    dosage: str
    frequency: str
    route: str                # oral | IV | topical | subcutaneous
    status: str               # active | stopped | completed
    prescribed_date: str
    prescriber: str
    indication: str

    def to_fhir_dict(self) -> Dict:
        return {
            "resourceType": "MedicationRequest",
            "id": self.id,
            "status": self.status,
            "medicationCodeableConcept": {
                "coding": [{"system": "http://www.nlm.nih.gov/research/umls/rxnorm",
                             "code": self.rxnorm_code, "display": self.medication_name}]
            },
            "subject": {"reference": f"Patient/{self.patient_id}"},
            "dosageInstruction": [{"text": f"{self.dosage} {self.frequency} {self.route}"}],
        }


@dataclass
class ClinicalNote:
    """Clinical note (SOAP format)."""
    id: str
    patient_id: str
    note_type: str            # progress | discharge | nursing | radiology | operative
    author: str
    date: str
    chief_complaint: str
    subjective: str
    objective: str
    assessment: str
    plan: str
    signed: bool = True


# ── Synthetic EHR Database ─────────────────────────────────────────────────────

SYNTHETIC_PATIENTS: List[FHIRPatient] = [
    FHIRPatient(
        id="PT001", mrn="MRN-100001",
        first_name="James", last_name="Carter",
        date_of_birth="1958-03-14", gender="male",
        phone="555-000-0001", email="j.carter@example.com",
        address="123 Maple St, St. Louis, MO 63101",
        blood_type="O+", allergies=["Penicillin", "Sulfa drugs"],
        primary_physician="Dr. Emily Roberts", insurance_id="INS-88821",
    ),
    FHIRPatient(
        id="PT002", mrn="MRN-100002",
        first_name="Maria", last_name="Gonzalez",
        date_of_birth="1972-07-22", gender="female",
        phone="555-000-0002", email="m.gonzalez@example.com",
        address="456 Oak Ave, St. Louis, MO 63102",
        blood_type="A-", allergies=["Aspirin", "Latex"],
        primary_physician="Dr. David Kim", insurance_id="INS-55344",
    ),
    FHIRPatient(
        id="PT003", mrn="MRN-100003",
        first_name="Robert", last_name="Thompson",
        date_of_birth="1945-11-05", gender="male",
        phone="555-000-0003", email="r.thompson@example.com",
        address="789 Pine Rd, St. Louis, MO 63103",
        blood_type="B+", allergies=["Codeine"],
        primary_physician="Dr. Emily Roberts", insurance_id="INS-67891",
    ),
    FHIRPatient(
        id="PT004", mrn="MRN-100004",
        first_name="Sarah", last_name="Williams",
        date_of_birth="1985-01-30", gender="female",
        phone="555-000-0004", email="s.williams@example.com",
        address="321 Elm Blvd, St. Louis, MO 63104",
        blood_type="AB+", allergies=[],
        primary_physician="Dr. Michael Chen", insurance_id="INS-44102",
    ),
    FHIRPatient(
        id="PT005", mrn="MRN-100005",
        first_name="David", last_name="Patel",
        date_of_birth="1963-09-17", gender="male",
        phone="555-000-0005", email="d.patel@example.com",
        address="654 Birch Ln, St. Louis, MO 63105",
        blood_type="O-", allergies=["NSAIDs", "Contrast dye"],
        primary_physician="Dr. Lisa Chang", insurance_id="INS-23456",
    ),
]

SYNTHETIC_CONDITIONS: List[FHIRCondition] = [
    FHIRCondition("C001","PT001","E11.9","Type 2 Diabetes Mellitus","active","2015-03-10","2015-03-10","moderate"),
    FHIRCondition("C002","PT001","I10","Essential Hypertension","active","2012-06-15","2012-06-15","mild"),
    FHIRCondition("C003","PT001","E78.5","Hyperlipidemia","active","2015-03-10","2015-03-10","mild"),
    FHIRCondition("C004","PT002","J45.30","Mild persistent asthma","active","2001-08-20","2001-08-20","mild"),
    FHIRCondition("C005","PT002","M79.3","Fibromyalgia","active","2019-01-15","2019-01-15","moderate"),
    FHIRCondition("C006","PT003","I50.9","Heart Failure, unspecified","active","2020-05-12","2020-05-12","severe"),
    FHIRCondition("C007","PT003","N18.3","Chronic Kidney Disease, stage 3","active","2018-11-30","2018-11-30","moderate"),
    FHIRCondition("C008","PT003","I10","Essential Hypertension","active","2005-02-14","2005-02-14","moderate"),
    FHIRCondition("C009","PT004","O26.89","Gestational diabetes (resolved)","resolved","2022-04-01","2022-04-01","mild"),
    FHIRCondition("C010","PT004","F32.1","Major depressive disorder, moderate","active","2020-09-10","2020-09-10","moderate"),
    FHIRCondition("C011","PT005","E11.65","T2DM with hyperglycemia","active","2010-07-22","2010-07-22","moderate"),
    FHIRCondition("C012","PT005","N18.4","CKD, stage 4","active","2022-01-08","2022-01-08","severe"),
    FHIRCondition("C013","PT005","I25.10","Coronary artery disease","active","2019-03-15","2019-03-15","severe"),
]

SYNTHETIC_OBSERVATIONS: List[FHIRObservation] = [
    # PT001 - James Carter
    FHIRObservation("OB001","PT001","laboratory","4548-4","HbA1c",7.8,"%",None,7.0,"final","2024-10-15","high"),
    FHIRObservation("OB002","PT001","vital-signs","55284-4","Blood Pressure Systolic",142,"mmHg",None,130,"final","2024-10-15","high"),
    FHIRObservation("OB003","PT001","vital-signs","55284-5","Blood Pressure Diastolic",88,"mmHg",None,80,"final","2024-10-15","high"),
    FHIRObservation("OB004","PT001","laboratory","2093-3","Cholesterol, Total",215,"mg/dL",None,200,"final","2024-10-15","high"),
    FHIRObservation("OB005","PT001","vital-signs","8302-2","Body Height",175,"cm",None,None,"final","2024-10-15","normal"),
    FHIRObservation("OB006","PT001","vital-signs","29463-7","Body Weight",92,"kg",None,None,"final","2024-10-15","normal"),
    # PT002 - Maria Gonzalez
    FHIRObservation("OB007","PT002","vital-signs","59408-5","SpO2",94,"%",95,100,"final","2024-11-01","low"),
    FHIRObservation("OB008","PT002","laboratory","6690-2","WBC",11.2,"10^3/uL",4.5,11.0,"final","2024-11-01","high"),
    FHIRObservation("OB009","PT002","vital-signs","8867-4","Heart Rate",98,"bpm",60,100,"final","2024-11-01","normal"),
    # PT003 - Robert Thompson
    FHIRObservation("OB010","PT003","laboratory","2160-0","Creatinine",2.8,"mg/dL",0.7,1.3,"final","2024-10-28","critical-high"),
    FHIRObservation("OB011","PT003","laboratory","3094-0","BUN",45,"mg/dL",7,25,"final","2024-10-28","high"),
    FHIRObservation("OB012","PT003","laboratory","6299-2","eGFR",28,"mL/min",60,None,"final","2024-10-28","low"),
    FHIRObservation("OB013","PT003","laboratory","2823-3","Potassium",5.6,"mEq/L",3.5,5.1,"final","2024-10-28","critical-high"),
    FHIRObservation("OB014","PT003","vital-signs","55284-4","Blood Pressure Systolic",158,"mmHg",None,130,"final","2024-10-28","high"),
    # PT004 - Sarah Williams
    FHIRObservation("OB015","PT004","laboratory","2093-3","Cholesterol, Total",178,"mg/dL",None,200,"final","2024-09-20","normal"),
    FHIRObservation("OB016","PT004","laboratory","718-7","Hemoglobin",10.8,"g/dL",12.0,16.0,"final","2024-09-20","low"),
    FHIRObservation("OB017","PT004","vital-signs","29463-7","Body Weight",63,"kg",None,None,"final","2024-09-20","normal"),
    # PT005 - David Patel
    FHIRObservation("OB018","PT005","laboratory","4548-4","HbA1c",9.2,"%",None,7.0,"final","2024-11-10","critical-high"),
    FHIRObservation("OB019","PT005","laboratory","2160-0","Creatinine",3.6,"mg/dL",0.7,1.3,"final","2024-11-10","critical-high"),
    FHIRObservation("OB020","PT005","laboratory","6299-2","eGFR",16,"mL/min",60,None,"final","2024-11-10","critical-low"),
    FHIRObservation("OB021","PT005","laboratory","2823-3","Potassium",6.1,"mEq/L",3.5,5.1,"final","2024-11-10","critical-high"),
]

SYNTHETIC_MEDICATIONS: List[FHIRMedication] = [
    FHIRMedication("M001","PT001","Metformin","860975","1000mg","twice daily","oral","active","2015-03-10","Dr. Roberts","Type 2 Diabetes"),
    FHIRMedication("M002","PT001","Lisinopril","197884","10mg","once daily","oral","active","2012-06-15","Dr. Roberts","Hypertension"),
    FHIRMedication("M003","PT001","Atorvastatin","617310","40mg","once nightly","oral","active","2015-03-10","Dr. Roberts","Hyperlipidemia"),
    FHIRMedication("M004","PT002","Albuterol inhaler","745679","90mcg","as needed","inhalation","active","2001-08-20","Dr. Kim","Asthma"),
    FHIRMedication("M005","PT002","Fluticasone","746765","110mcg","twice daily","inhalation","active","2019-06-01","Dr. Kim","Asthma"),
    FHIRMedication("M006","PT002","Duloxetine","596926","30mg","once daily","oral","active","2019-01-20","Dr. Kim","Fibromyalgia"),
    FHIRMedication("M007","PT003","Furosemide","202991","40mg","twice daily","oral","active","2020-05-15","Dr. Roberts","Heart Failure"),
    FHIRMedication("M008","PT003","Carvedilol","200031","12.5mg","twice daily","oral","active","2020-05-15","Dr. Roberts","Heart Failure"),
    FHIRMedication("M009","PT003","Losartan","202429","50mg","once daily","oral","active","2018-11-30","Dr. Roberts","HTN/CKD"),
    FHIRMedication("M010","PT004","Sertraline","36437","50mg","once daily","oral","active","2020-09-15","Dr. Chen","Depression"),
    FHIRMedication("M011","PT005","Insulin glargine","274783","20 units","once nightly","subcutaneous","active","2021-03-10","Dr. Chang","T2DM"),
    FHIRMedication("M012","PT005","Amlodipine","17767","10mg","once daily","oral","active","2019-03-20","Dr. Chang","CAD/HTN"),
    FHIRMedication("M013","PT005","Sevelamer","196472","800mg","three times daily","oral","active","2022-01-10","Dr. Chang","CKD/Hyperphosphatemia"),
]

SYNTHETIC_NOTES: List[ClinicalNote] = [
    ClinicalNote(
        id="N001", patient_id="PT001", note_type="progress",
        author="Dr. Emily Roberts", date="2024-10-15",
        chief_complaint="Routine diabetic follow-up and hypertension management",
        subjective="Patient James Carter presents for 3-month follow-up. Reports occasional headaches in the morning. Denies chest pain, shortness of breath. Blood glucose readings at home averaging 160-180 mg/dL fasting. Diet compliance reported as 'fair'. Exercise limited to walking 20 min, 3x/week.",
        objective="BP 142/88 mmHg (elevated). HR 78 bpm regular. Weight 92kg, BMI 30.1. HbA1c 7.8% (target <7.0%). Total cholesterol 215 mg/dL. Fasting glucose 168 mg/dL. Creatinine 1.0 mg/dL (normal). eGFR >60.",
        assessment="1. T2DM — suboptimally controlled (HbA1c 7.8%, target <7.0%). 2. Hypertension — inadequately controlled (BP 142/88). 3. Hyperlipidemia — borderline elevated cholesterol.",
        plan="1. Increase Metformin to 1000mg BID (already at max; consider adding SGLT2i). 2. Increase Lisinopril to 20mg daily for BP control. 3. Intensify dietary counseling — refer to diabetes educator. 4. Recheck HbA1c, BMP, lipid panel in 3 months. 5. Continue Atorvastatin 40mg nightly. 6. Encourage DASH diet.",
    ),
    ClinicalNote(
        id="N002", patient_id="PT003", note_type="progress",
        author="Dr. Emily Roberts", date="2024-10-28",
        chief_complaint="Worsening shortness of breath, bilateral leg swelling",
        subjective="Robert Thompson, 79M with HFrEF, CKD stage 3, HTN. Reports 2-week progression of dyspnea on exertion (now with minimal activity). Orthopnea requiring 3 pillows. Bilateral ankle swelling worse than baseline. Weight gain of 6 lbs over past week. Denies chest pain. Reports compliance with medications.",
        objective="BP 158/96 mmHg. HR 92 bpm. SpO2 91% on room air. Weight 84kg (up 3kg from last visit). JVD present at 45°. Bilateral crackles at lung bases. 2+ pitting edema bilateral lower extremities. Creatinine 2.8 (up from 2.1). eGFR 28. BUN 45. K+ 5.6 (critical high). BNP pending.",
        assessment="1. Acute decompensated heart failure — worsening fluid overload. 2. AKI on CKD stage 3-4 — likely cardiorenal syndrome. 3. Hyperkalemia (K+ 5.6) — likely medication-related (Losartan + CKD). 4. Uncontrolled hypertension.",
        plan="1. ADMIT to cardiology service. 2. IV Furosemide 80mg now, then 40mg BID. 3. Hold Losartan given AKI + hyperkalemia. 4. Cardiology and nephrology consult. 5. Strict I&O, daily weights. 6. Low-potassium, fluid-restricted diet. 7. Repeat BMP tomorrow morning. 8. Echocardiogram to reassess EF.",
    ),
    ClinicalNote(
        id="N003", patient_id="PT005", note_type="progress",
        author="Dr. Lisa Chang", date="2024-11-10",
        chief_complaint="Poorly controlled diabetes, declining kidney function",
        subjective="David Patel, 61M with T2DM, CKD stage 4, CAD. HbA1c 9.2% at today's draw. Reports fatigue, nocturia x4, decreased appetite. Glucose readings 250-320 mg/dL consistently. Denies chest pain. No hypoglycemic episodes. Reports missing insulin doses occasionally.",
        objective="BP 148/92 mmHg. HR 76 bpm. Weight 78kg. HbA1c 9.2% (critical). Creatinine 3.6 mg/dL (critical high). eGFR 16 mL/min. K+ 6.1 mEq/L (critical high). Glucose 287 mg/dL. Hemoglobin 9.8 g/dL.",
        assessment="1. T2DM, severely uncontrolled (HbA1c 9.2%). 2. CKD stage 4-5 approaching ESRD — eGFR 16. 3. Life-threatening hyperkalemia (K+ 6.1). 4. Hypertension, uncontrolled. 5. Anemia of CKD.",
        plan="1. URGENT: Treat hyperkalemia — Kayexalate 30g oral now. 2. Nephrology referral — URGENT — ESRD planning, dialysis access discussion. 3. Intensify insulin: increase glargine to 30 units nightly, add mealtime Lispro 5 units TID. 4. Diabetes educator + dietitian — renal diet education. 5. Erythropoietin stimulating agent for anemia per nephrology. 6. Cardiology follow-up for CAD management. 7. Repeat BMP in 24 hours.",
    ),
]


# ── EHR Data Access Layer ──────────────────────────────────────────────────────

class EHRDatabase:
    """In-memory EHR database with FHIR-compliant access methods."""

    def __init__(self):
        self.patients = {p.id: p for p in SYNTHETIC_PATIENTS}
        self.conditions = SYNTHETIC_CONDITIONS
        self.observations = SYNTHETIC_OBSERVATIONS
        self.medications = SYNTHETIC_MEDICATIONS
        self.notes = SYNTHETIC_NOTES

    def get_patient(self, patient_id: str) -> Optional[FHIRPatient]:
        return self.patients.get(patient_id)

    def get_patient_by_mrn(self, mrn: str) -> Optional[FHIRPatient]:
        return next((p for p in self.patients.values() if p.mrn == mrn), None)

    def search_patients(self, query: str) -> List[FHIRPatient]:
        q = query.lower()
        return [p for p in self.patients.values()
                if q in p.full_name.lower() or q in p.mrn.lower()]

    def get_conditions(self, patient_id: str) -> List[FHIRCondition]:
        return [c for c in self.conditions if c.patient_id == patient_id]

    def get_active_conditions(self, patient_id: str) -> List[FHIRCondition]:
        return [c for c in self.conditions if c.patient_id == patient_id and c.clinical_status == "active"]

    def get_observations(self, patient_id: str, category: Optional[str] = None) -> List[FHIRObservation]:
        obs = [o for o in self.observations if o.patient_id == patient_id]
        if category:
            obs = [o for o in obs if o.category == category]
        return sorted(obs, key=lambda o: o.effective_date, reverse=True)

    def get_abnormal_observations(self, patient_id: str) -> List[FHIRObservation]:
        return [o for o in self.get_observations(patient_id) if o.is_abnormal]

    def get_medications(self, patient_id: str, active_only: bool = True) -> List[FHIRMedication]:
        meds = [m for m in self.medications if m.patient_id == patient_id]
        if active_only:
            meds = [m for m in meds if m.status == "active"]
        return meds

    def get_notes(self, patient_id: str, note_type: Optional[str] = None) -> List[ClinicalNote]:
        notes = [n for n in self.notes if n.patient_id == patient_id]
        if note_type:
            notes = [n for n in notes if n.note_type == note_type]
        return sorted(notes, key=lambda n: n.date, reverse=True)

    def get_full_summary(self, patient_id: str) -> Dict[str, Any]:
        """Return complete patient clinical summary."""
        patient = self.get_patient(patient_id)
        if not patient:
            return {}
        return {
            "patient": asdict(patient),
            "conditions": [asdict(c) for c in self.get_active_conditions(patient_id)],
            "observations": [asdict(o) for o in self.get_observations(patient_id)[:10]],
            "medications": [asdict(m) for m in self.get_medications(patient_id)],
            "recent_notes": [vars(n) for n in self.get_notes(patient_id)[:2]],
        }

    def list_all_patients(self) -> List[FHIRPatient]:
        return list(self.patients.values())


# Singleton
ehr_db = EHRDatabase()
