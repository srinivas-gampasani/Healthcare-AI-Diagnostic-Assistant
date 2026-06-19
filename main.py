#!/usr/bin/env python3
"""
Healthcare AI Diagnostic Assistant
===================================
Entry point.

Usage:
  python main.py                   # Start API server
  python main.py --demo            # Run offline demo (no API key needed)
  python main.py --test-hipaa      # Test HIPAA PHI masking
  python main.py --test-ehr        # Test EHR data access
"""
import sys
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║     Healthcare AI Diagnostic Assistant v1.0.0                ║
║     GPT-4 + RAG + HL7 FHIR · HIPAA Compliant                ║
╠══════════════════════════════════════════════════════════════╣
║  API:   http://localhost:8000                                 ║
║  Docs:  http://localhost:8000/docs                           ║
║  UI:    http://localhost:8000                                 ║
║  Demo:  python main.py --demo                                ║
╚══════════════════════════════════════════════════════════════╝
"""


def run_server():
    import uvicorn
    from config import settings
    print(BANNER)
    uvicorn.run(
        "api.server:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True,
        log_level="info",
    )


def run_demo():
    """Offline demo — shows all features without OpenAI API key."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.markdown import Markdown

    console = Console()
    console.print(Panel.fit("[bold blue]Healthcare AI Diagnostic Assistant — Offline Demo[/]",
                             subtitle="HIPAA Compliant · GPT-4 + RAG + FHIR"))

    # 1. EHR Data
    console.print("\n[bold yellow]1. EHR Patient Database (HL7 FHIR)[/]")
    from ehr.fhir_models import ehr_db
    patients = ehr_db.list_all_patients()
    table = Table(show_header=True, header_style="bold blue")
    table.add_column("ID"); table.add_column("Name"); table.add_column("Age")
    table.add_column("Conditions"); table.add_column("Meds"); table.add_column("Abnormal Labs")
    for p in patients:
        conds = ehr_db.get_active_conditions(p.id)
        meds  = ehr_db.get_medications(p.id)
        abn   = ehr_db.get_abnormal_observations(p.id)
        table.add_row(p.id, p.full_name, str(p.age),
                      str(len(conds)), str(len(meds)),
                      f"[red]{len(abn)}[/]" if abn else "[green]0[/]")
    console.print(table)

    # 2. HIPAA PHI Masking
    console.print("\n[bold yellow]2. HIPAA PHI Detection & Masking[/]")
    from core.hipaa import phi_masker
    test_texts = [
        "Patient SSN: 123-45-6789, call 555-867-5309",
        "DOB: 03/14/1958, MRN: 100001",
        "Email: james.carter@gmail.com, Account: ACC-998877",
        "Normal clinical note with no PHI present.",
    ]
    for t in test_texts:
        masked, found = phi_masker.mask(t)
        status = f"[red]PHI: {found}[/]" if found else "[green]Clean[/]"
        console.print(f"  {status}")
        console.print(f"  Original: {t}")
        console.print(f"  Masked:   {masked}\n")

    # 3. RAG Engine
    console.print("[bold yellow]3. Clinical RAG Engine — Guideline Retrieval[/]")
    from rag.engine import rag_engine
    rag_engine.initialize()
    queries = [
        "What is the HbA1c target for Type 2 Diabetes?",
        "How to manage hyperkalemia in CKD patients?",
        "Blood pressure target for heart failure patients?",
    ]
    for q in queries:
        results = rag_engine.retrieve(q, top_k=2)
        console.print(f"  Query: [cyan]{q}[/]")
        for r in results:
            console.print(f"    → [{r.source_type}] {r.title[:60]} (score: {r.score:.3f})")
        console.print()

    # 4. Critical Alerts
    console.print("[bold yellow]4. Clinical Decision Support Alerts[/]")
    from agents.clinical_assistant import assistant
    for pid in ["PT001", "PT003", "PT005"]:
        p = ehr_db.get_patient(pid)
        alerts = assistant._check_critical_alerts(pid)
        console.print(f"  [bold]{p.full_name}[/] ({pid}):")
        if alerts:
            for a in alerts:
                console.print(f"    [red]{a}[/]")
        else:
            console.print("    [green]No critical alerts[/]")
        console.print()

    # 5. Offline AI Response (fallback)
    console.print("[bold yellow]5. Offline Fallback Response (no API key)[/]")
    result = assistant.query(
        "What are the current lab concerns for this patient?",
        patient_id="PT005",
        user_id="demo_user",
        include_rag=False,
    )
    console.print(Panel(result["answer"][:800], title="AI Response (offline mode)", border_style="blue"))

    console.print("\n[bold green]✓ Demo complete. Run 'python main.py' with OPENAI_API_KEY for full AI capabilities.[/]")


def test_hipaa():
    from rich.console import Console
    console = Console()
    console.print("[bold]HIPAA PHI Masking Tests[/]\n")
    from core.hipaa import phi_masker, audit_logger

    tests = [
        ("SSN detection", "Patient SSN is 123-45-6789", ["ssn"]),
        ("Phone masking", "Call patient at 555-867-5309", ["phone"]),
        ("Email masking", "Email: patient@gmail.com", ["email"]),
        ("MRN masking", "MRN: 100001 admitted today", ["mrn"]),
        ("DOB masking", "DOB: 03/14/1958", ["dob"]),
        ("Clean text", "Blood pressure 140/90 mmHg today", []),
    ]

    passed = 0
    for name, text, expected_types in tests:
        masked, found = phi_masker.mask(text)
        ok = all(t in found for t in expected_types) and (bool(found) == bool(expected_types))
        icon = "[green]PASS[/]" if ok else "[red]FAIL[/]"
        console.print(f"  {icon} {name}")
        if ok: passed += 1

    # Audit log test
    audit_logger.log("test_user", "phi_test", "system", "test-001")
    recent = audit_logger.get_recent(1)
    audit_ok = len(recent) > 0
    console.print(f"  {'[green]PASS[/]' if audit_ok else '[red]FAIL[/]'} Audit logging")
    if audit_ok: passed += 1

    console.print(f"\n[bold]{passed}/{len(tests)+1} tests passed[/]")


def test_ehr():
    from rich.console import Console
    console = Console()
    console.print("[bold]EHR / FHIR Data Tests[/]\n")
    from ehr.fhir_models import ehr_db

    passed = 0; total = 0

    def chk(name, condition):
        nonlocal passed, total
        total += 1
        ok = bool(condition)
        console.print(f"  {'[green]PASS[/]' if ok else '[red]FAIL[/]'} {name}")
        if ok: passed += 1

    chk("5 patients loaded", len(ehr_db.list_all_patients()) == 5)
    chk("Get patient by ID", ehr_db.get_patient("PT001") is not None)
    chk("Get patient by MRN", ehr_db.get_patient_by_mrn("MRN-100001") is not None)
    chk("Search patients", len(ehr_db.search_patients("carter")) > 0)
    chk("Active conditions PT001", len(ehr_db.get_active_conditions("PT001")) > 0)
    chk("Observations PT003", len(ehr_db.get_observations("PT003")) > 0)
    chk("Abnormal obs PT005", len(ehr_db.get_abnormal_observations("PT005")) > 0)
    chk("Medications PT001", len(ehr_db.get_medications("PT001")) > 0)
    chk("Clinical notes PT001", len(ehr_db.get_notes("PT001")) > 0)
    chk("Full summary PT003", bool(ehr_db.get_full_summary("PT003")))
    pt = ehr_db.get_patient("PT001")
    chk("FHIR dict output", "resourceType" in pt.to_fhir_dict())
    chk("Patient age calculation", pt.age > 0)

    console.print(f"\n[bold]{passed}/{total} tests passed[/]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Healthcare AI Diagnostic Assistant")
    parser.add_argument("--demo",       action="store_true", help="Run offline demo")
    parser.add_argument("--test-hipaa", action="store_true", help="Test HIPAA PHI masking")
    parser.add_argument("--test-ehr",   action="store_true", help="Test EHR data layer")
    args = parser.parse_args()

    if args.demo:
        run_demo()
    elif args.test_hipaa:
        test_hipaa()
    elif args.test_ehr:
        test_ehr()
    else:
        run_server()
