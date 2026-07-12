"""
utils/student_history.py
--------------------------
Historical writing-sample management and Stylistic Signature Summary builder.

COS layout:
  student_history/<class_id>/<student_id>/roster.json
  student_history/<class_id>/<student_id>/<assignment_id>.txt

roster.json schema:
{
  "student_id": "s001",
  "full_name": "John Doe",
  "class_id": "CS101",
  "submissions": [
    { "assignment_id": "hw1", "title": "Sorting Algorithms", "submitted_at": "..." }
  ]
}

The Stylistic Signature Summary is a compressed textual snapshot assembled
from past submissions and fed to Granite as in-context RAG evidence.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

HISTORY_PREFIX = "student_history/"
MAX_SAMPLE_CHARS = 3_000   # characters per historical sample used in summary
MAX_SAMPLES = 5             # maximum past submissions to include


# ──────────────────────────────────────────────────────────────────────────────
# Roster helpers
# ──────────────────────────────────────────────────────────────────────────────

def _roster_key(class_id: str, student_id: str) -> str:
    return f"{HISTORY_PREFIX}{class_id}/{student_id}/roster.json"


def _submission_key(class_id: str, student_id: str, assignment_id: str) -> str:
    return f"{HISTORY_PREFIX}{class_id}/{student_id}/{assignment_id}.txt"


def list_students(cos, class_id: str) -> list[dict]:
    """Return all student roster records for *class_id*."""
    prefix = f"{HISTORY_PREFIX}{class_id}/"
    keys = cos.list_prefix(prefix)
    students: list[dict] = []
    for key in keys:
        if key.endswith("roster.json"):
            roster = cos.get_json(key)
            if roster:
                students.append(roster)
    return students


def get_student_roster(cos, class_id: str, student_id: str) -> Optional[dict]:
    """Fetch a single student's roster record."""
    return cos.get_json(_roster_key(class_id, student_id))


def save_student_roster(cos, class_id: str, student_id: str, roster: dict) -> None:
    """Persist / update a student's roster record in COS."""
    roster.setdefault("student_id", student_id)
    roster.setdefault("class_id", class_id)
    roster.setdefault("submissions", [])
    cos.put_json(_roster_key(class_id, student_id), roster)


def save_submission(
    cos,
    class_id: str,
    student_id: str,
    assignment_id: str,
    title: str,
    text: str,
) -> None:
    """
    Store a new assignment submission body and update the student's roster.
    """
    cos.put_text(_submission_key(class_id, student_id, assignment_id), text)

    # Update roster record
    roster = get_student_roster(cos, class_id, student_id) or {
        "student_id": student_id,
        "class_id": class_id,
        "submissions": [],
    }
    # Avoid duplicate entries
    existing_ids = {s["assignment_id"] for s in roster["submissions"]}
    if assignment_id not in existing_ids:
        roster["submissions"].append({
            "assignment_id": assignment_id,
            "title": title,
            "submitted_at": datetime.now(timezone.utc).isoformat(),
        })
    save_student_roster(cos, class_id, student_id, roster)


# ──────────────────────────────────────────────────────────────────────────────
# Direct COS upload for the History page form
# ──────────────────────────────────────────────────────────────────────────────

# Separate prefix used by the History page upload form so it never collides
# with the evaluation-pipeline path (student_history/).
HISTORY_RECORDS_PREFIX = "history_records/"


def upload_history_to_ibm_cos(
    cos,
    student_id: str,
    topic: str,
    text: str,
    full_name: str = "",
    class_id: str = "",
) -> tuple[bool, str]:
    """
    Upload a historical reference assignment directly to IBM Cloud Object Storage
    under the path:

        history_records/<student_id>/<topic_slug>.txt

    Also registers the submission in the student's roster (student_history/ path)
    so it immediately becomes available as RAG context during evaluations.

    Returns (True, cos_key) on success or (False, error_message) on failure.
    """
    if not student_id.strip():
        return False, "Student ID cannot be empty."
    if not topic.strip():
        return False, "Assignment topic cannot be empty."
    if not text.strip():
        return False, "Assignment text cannot be empty."

    # Build a safe filename slug from the topic string
    slug = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in topic.strip())
    slug = slug[:80]  # cap length

    cos_key = f"{HISTORY_RECORDS_PREFIX}{student_id.strip()}/{slug}.txt"

    try:
        # 1. Write the raw text body to history_records/ path
        cos.put_text(cos_key, text)

        # 2. Mirror into student_history/ so the RAG pipeline can find it.
        #    Use class_id if provided, otherwise fall back to a generic bucket.
        effective_class = class_id.strip() if class_id.strip() else "UPLOADED"
        save_submission(
            cos,
            class_id=effective_class,
            student_id=student_id.strip(),
            assignment_id=slug,
            title=topic.strip(),
            text=text,
        )

        # 3. If a full name was provided, backfill it into the roster record
        if full_name.strip():
            roster = get_student_roster(cos, effective_class, student_id.strip()) or {}
            if roster.get("full_name", "") != full_name.strip():
                roster["full_name"] = full_name.strip()
                save_student_roster(cos, effective_class, student_id.strip(), roster)

        return True, cos_key

    except Exception as exc:
        return False, str(exc)


# ──────────────────────────────────────────────────────────────────────────────
# Stylistic Signature Summary (RAG simulation)
# ──────────────────────────────────────────────────────────────────────────────

def build_stylistic_signature(cos, class_id: str, student_id: str) -> str:
    """
    Pull up to MAX_SAMPLES past submissions from COS and distil them into
    a textual Stylistic Signature Summary.

    The summary is a compact in-context representation of the student's
    historical writing patterns for use in Layer 2 of the plagiarism agent.
    Returns a fallback notice when no history exists.
    """
    roster = get_student_roster(cos, class_id, student_id)
    if not roster or not roster.get("submissions"):
        return (
            "NO HISTORICAL DATA: No prior submissions are available for this "
            "student. Style-deviation analysis cannot be performed for Layer 2."
        )

    submissions = roster["submissions"][-MAX_SAMPLES:]   # most recent first
    excerpts: list[str] = []

    for submission in reversed(submissions):   # oldest → newest for context
        aid = submission["assignment_id"]
        title = submission.get("title", aid)
        text = cos.get_text(_submission_key(class_id, student_id, aid))
        if text:
            excerpt = text[:MAX_SAMPLE_CHARS]
            excerpts.append(
                f"### Past Assignment: '{title}'\n"
                f"(Excerpt — {len(excerpt)} chars)\n{excerpt}"
            )

    if not excerpts:
        return (
            "NO HISTORICAL DATA: Submission records exist but file bodies "
            "could not be retrieved from storage."
        )

    full_name = roster.get("full_name", student_id)
    header = (
        f"STUDENT STYLISTIC SIGNATURE — {full_name} (ID: {student_id})\n"
        f"Based on {len(excerpts)} historical submission(s).\n"
        f"Use these samples as the reference baseline for style deviation analysis.\n"
        "─" * 60
    )

    return header + "\n\n" + "\n\n".join(excerpts)


def add_demo_student(cos, class_id: str = "DEMO101") -> None:
    """
    Seed a demo student with two synthetic past submissions so the app
    works end-to-end without pre-existing COS data.
    Called once on first app load when no students exist in the demo class.
    """
    demo_student_id = "demo_student_001"
    roster = get_student_roster(cos, class_id, demo_student_id)
    if roster:
        return  # already seeded

    sample_1 = """
In the domain of distributed systems, the concept of eventual consistency
represents a fundamental trade-off between availability and strong consistency
guarantees. Unlike linearizable systems, eventually consistent stores allow
temporary divergence between replicas, reconciling state through anti-entropy
protocols such as gossip or vector-clock-based merging. The CAP theorem, first
proposed by Brewer, formalises this constraint: no distributed system can
simultaneously guarantee consistency, availability, and partition tolerance.
My analysis focuses on how CRDTs sidestep the CAP limitation by designing
data structures whose merge operations are commutative, associative, and
idempotent, enabling conflict-free replication without coordination.
""".strip()

    sample_2 = """
Operating system scheduling algorithms balance throughput, latency, and
fairness. Round-Robin assigns equal CPU quanta, whereas Shortest Job First
minimises average waiting time at the cost of potential starvation for longer
processes. I tend to ground my arguments in concrete numerical examples: for
a workload of three processes (burst times 4, 2, 6 ms) under SJF, the average
waiting time is (0+4+6)/3 = ~3.33 ms compared to 5.33 ms under FCFS. This
quantitative reasoning style, paired with pseudocode illustrations, is the
analytical approach I consistently apply across problem sets.
""".strip()

    save_submission(cos, class_id, demo_student_id, "hw1",
                    "Distributed Systems & CRDTs", sample_1)
    save_submission(cos, class_id, demo_student_id, "hw2",
                    "OS Scheduling Algorithms", sample_2)

    # Update roster with full name
    roster = get_student_roster(cos, class_id, demo_student_id)
    roster["full_name"] = "Alex Johnson (Demo)"
    save_student_roster(cos, class_id, demo_student_id, roster)

    logger.info("Demo student seeded in class %s", class_id)
