"""
utils/instructor_config.py
----------------------------
Instructor dashboard configuration: grading patterns, preferences,
and the dynamic system-prompt constraint builder.

Configurations are persisted in COS under:
  instructor_configs/<username>.json

Schema:
{
  "username": "prof_smith",
  "grading_style": "Concept-focused",          # primary style tag
  "custom_tags": ["cite sources", "diagrams"],  # additional rubric tags
  "weight_originality": 40,                     # % weight (must sum to 100)
  "weight_ai_detection": 35,
  "weight_style": 25,
  "rubric_notes": "Free-text professor notes",
  "updated_at": "<ISO datetime>"
}
"""

import logging
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

CONFIG_PREFIX = "instructor_configs/"

# Predefined grading style vocabulary
GRADING_STYLES = [
    "Concept-focused",
    "Definition-heavy",
    "Structure / Diagram reliant",
    "Critical Analysis",
    "Problem-solving / Applied",
    "Literature Review",
    "Lab Report",
]

DEFAULT_TAGS: list[str] = []


# ──────────────────────────────────────────────────────────────────────────────
# COS helpers
# ──────────────────────────────────────────────────────────────────────────────

def _config_key(username: str) -> str:
    return f"{CONFIG_PREFIX}{username}.json"


def save_config(cos, username: str, config: dict) -> tuple[bool, str]:
    """Persist instructor config to COS."""
    config["username"] = username
    config["updated_at"] = datetime.now(timezone.utc).isoformat()
    try:
        cos.put_json(_config_key(username), config)
        return True, ""
    except Exception as exc:
        return False, str(exc)


def load_config(cos, username: str) -> Optional[dict]:
    """Load instructor config from COS; returns None if not configured yet."""
    return cos.get_json(_config_key(username))


# ──────────────────────────────────────────────────────────────────────────────
# System-prompt constraint builder
# ──────────────────────────────────────────────────────────────────────────────

def build_rubric_constraint(config: Optional[dict]) -> str:
    """
    Convert an instructor config dict into a natural-language constraint block
    ready to be injected into the Granite system prompt.

    Returns a placeholder string when *config* is None so the agent still
    functions even without a saved configuration.
    """
    if not config:
        return (
            "No instructor rubric has been configured. "
            "Apply standard academic integrity criteria."
        )

    style = config.get("grading_style", "General")
    tags = config.get("custom_tags", [])
    notes = config.get("rubric_notes", "").strip()
    w_orig = config.get("weight_originality", 40)
    w_ai = config.get("weight_ai_detection", 35)
    w_sty = config.get("weight_style", 25)

    tag_str = ", ".join(tags) if tags else "none"
    notes_str = f'\n  Additional rubric notes: "{notes}"' if notes else ""

    constraint = f"""INSTRUCTOR RUBRIC CONSTRAINTS (apply during Layer 3 evaluation):
  - Primary grading style: {style}
  - Required rubric elements: {tag_str}
  - Score weighting preference:
      Originality   → {w_orig}%
      AI Detection  → {w_ai}%
      Style Match   → {w_sty}%{notes_str}

When generating the rubric alignment checklist, verify whether the submission
satisfies each required rubric element listed above, and flag any element
that is missing or insufficiently addressed."""

    return constraint
