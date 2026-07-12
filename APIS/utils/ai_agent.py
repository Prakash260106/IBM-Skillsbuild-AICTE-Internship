"""
utils/ai_agent.py
------------------
Three-layer plagiarism evaluation agent (IBM WatsonX).

Responsibilities:
  1. Build an optimised multi-shot system prompt combining:
       - Layer 1  AI / paraphrase detection instructions
       - Layer 2  style-deviation analysis using student's Stylistic Signature
       - Layer 3  rubric alignment check using instructor grading constraints
  2. Call ibm-watsonx-ai with retries
  3. Parse the structured JSON response into a typed EvaluationReport

Models (tried in order — fastest first):
  1. mistralai/mistral-small-3-1-24b-instruct-2503  — fastest, primary
  2. meta-llama/llama-3-3-70b-instruct              — strong fallback
  3. mistralai/mistral-medium-2505                  — heavier fallback
  4. meta-llama/llama-4-maverick-17b-128e-instruct-fp8  — last resort
"""

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Optional

from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as Params
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Preferred models (tried in order)
# Models confirmed available on this WatsonX environment.
# ──────────────────────────────────────────────────────────────────────────────
PREFERRED_MODELS = [
    "mistralai/mistral-small-3-1-24b-instruct-2503",        # Fastest — primary model
    "meta-llama/llama-3-3-70b-instruct",                    # Strong fallback
    "mistralai/mistral-medium-2505",                        # Heavier fallback
    "meta-llama/llama-4-maverick-17b-128e-instruct-fp8",    # Last resort
]

# Human-readable display name for the primary model shown in the UI
PRIMARY_MODEL_DISPLAY = "Mistral Small 3.1 24B Instruct"


# ──────────────────────────────────────────────────────────────────────────────
# Typed result
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class EvaluationReport:
    """Structured output produced by the three-layer evaluation agent."""
    originality_score: float = 0.0          # 0–100 %
    ai_likelihood_score: float = 0.0        # 0–100 %
    style_consistency: str = ""             # qualitative label + explanation
    rubric_checklist: list[dict] = field(default_factory=list)
    overall_verdict: str = ""               # one-sentence summary
    detailed_analysis: str = ""             # raw LLM reasoning block
    model_used: str = ""
    error: Optional[str] = None


# ──────────────────────────────────────────────────────────────────────────────
# Prompt builder
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT_TEMPLATE = """You are an expert Academic Integrity AI assistant specialised in
plagiarism detection, AI-generated text identification, and writing-style analysis.
Your task is to evaluate a student assignment submission through three analytical layers
and produce a structured JSON report.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LAYER 1 — AI CONTENT & PARAPHRASE DETECTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Analyse the submission for signals of AI-generated or heavily paraphrased content:
  • Perplexity: AI text tends to have low perplexity — over-smooth, predictable phrasing.
  • Burstiness: Human writing shows high "burst" variation in sentence length and complexity;
    AI writing is uniformly structured.
  • Vocabulary entropy: AI text favours high-frequency, "safe" word choices.
  • Transition formulaic patterns: Phrases like "In conclusion, ...", "It is important to note
    that ...", "Furthermore, ..." used repetitively signal AI origin.
  • Semantic coherence without depth: Paragraphs that sound authoritative but lack
    concrete examples, personal opinion, or original synthesis.
  • Stylistic rephrasing fingerprints: Unusual synonym substitutions or sentence inversions
    that indicate paraphrasing tools.

Output an AI Likelihood Score as an integer (0 = definitely human, 100 = definitely AI).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LAYER 2 — STYLISTIC DEVIATION ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Compare the submission against the STUDENT HISTORICAL BASELINE provided below.
Look for deviations in:
  • Vocabulary sophistication and domain-specific terminology choices
  • Average sentence length and structural complexity
  • Punctuation density (comma rate, semicolons, em-dashes)
  • Rhetorical tone (formal / informal / academic / conversational)
  • Argumentative structure (inductive vs. deductive)
  • Use of first-person vs. passive voice
  • Citation / reference integration patterns

Assign one of: "Consistent", "Minor Deviation", "Moderate Deviation", "Significant Deviation"
and provide a concise qualitative explanation (2–3 sentences).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LAYER 3 — RUBRIC ALIGNMENT CHECK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{rubric_constraint}

Produce a checklist of each required rubric element as:
  [ ] Element name → "Met" / "Partially Met" / "Not Met" with a one-line rationale.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT (respond ONLY with valid JSON — no markdown fences, no prose outside JSON)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{{
  "originality_score": <integer 0-100>,
  "ai_likelihood_score": <integer 0-100>,
  "style_consistency": {{
    "label": "<Consistent | Minor Deviation | Moderate Deviation | Significant Deviation>",
    "explanation": "<2-3 sentence qualitative analysis>"
  }},
  "rubric_checklist": [
    {{ "item": "<rubric element>", "status": "<Met | Partially Met | Not Met>", "rationale": "<1 sentence>" }}
  ],
  "overall_verdict": "<1 sentence summary for the instructor>",
  "detailed_analysis": "<paragraph-length reasoning covering all three layers>"
}}
"""

USER_PROMPT_TEMPLATE = """
─── STUDENT HISTORICAL BASELINE ────────────────────────────────────────────
{stylistic_signature}

─── CURRENT SUBMISSION TO EVALUATE ─────────────────────────────────────────
{submission_text}

─── TASK ────────────────────────────────────────────────────────────────────
Perform the three-layer evaluation as instructed. Reply with the JSON report only.
"""


def build_messages(
    submission_text: str,
    stylistic_signature: str,
    rubric_constraint: str,
) -> list[dict]:
    """
    Construct the chat-formatted message list for Granite instruct models.
    """
    system = SYSTEM_PROMPT_TEMPLATE.format(rubric_constraint=rubric_constraint)
    user = USER_PROMPT_TEMPLATE.format(
        stylistic_signature=stylistic_signature,
        submission_text=submission_text,
    )
    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]


# ──────────────────────────────────────────────────────────────────────────────
# watsonx.ai client factory
# ──────────────────────────────────────────────────────────────────────────────

def _build_model(api_key: str, url: str, project_id: str, model_id: str, max_new_tokens: int = 900) -> ModelInference:
    """Instantiate a ModelInference client for the given model."""
    creds = Credentials(api_key=api_key, url=url)
    params = {
        Params.DECODING_METHOD: "greedy",
        Params.MAX_NEW_TOKENS:  max_new_tokens,
        Params.TEMPERATURE:     0.0,    # deterministic for evaluation tasks
        Params.REPETITION_PENALTY: 1.05,
    }
    return ModelInference(
        model_id=model_id,
        credentials=creds,
        project_id=project_id,
        params=params,
    )


# ──────────────────────────────────────────────────────────────────────────────
# JSON extractor — handles model responses that wrap JSON in prose/fences
# ──────────────────────────────────────────────────────────────────────────────

def _extract_json(raw: str) -> dict:
    """
    Attempt multiple strategies to locate a JSON object in *raw* text:
      1. Direct parse
      2. Strip markdown code fences (```json ... ```)
      3. Regex search for the first {...} block
    Raises ValueError if all strategies fail.
    """
    raw = raw.strip()

    # Strategy 1 — clean JSON
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Strategy 2 — strip fences
    fence_match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", raw)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    # Strategy 3 — find first balanced { ... }
    brace_match = re.search(r"\{[\s\S]+\}", raw)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not extract JSON from model response:\n{raw[:500]}")


# ──────────────────────────────────────────────────────────────────────────────
# Retry wrapper for the watsonx API call
# ──────────────────────────────────────────────────────────────────────────────

@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=1, max=4),
    reraise=True,
)
def _call_model(model: ModelInference, messages: list[dict]) -> str:
    """Send *messages* to the model and return the raw text response."""
    response = model.chat(messages=messages)
    # Extract content from the standard chat-completion response shape
    choices = response.get("choices", [])
    if not choices:
        raise ValueError("Model returned empty choices list.")
    return choices[0]["message"]["content"]


# ──────────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_submission(
    api_key: str,
    url: str,
    project_id: str,
    submission_text: str,
    stylistic_signature: str,
    rubric_constraint: str,
) -> EvaluationReport:
    """
    Run the full three-layer evaluation and return an EvaluationReport.

    Tries each model in PREFERRED_MODELS until one succeeds.
    """
    if not submission_text.strip():
        return EvaluationReport(error="Submission text is empty — nothing to evaluate.")

    messages = build_messages(submission_text, stylistic_signature, rubric_constraint)
    raw_response = ""
    last_error: Optional[Exception] = None

    for model_id in PREFERRED_MODELS:
        try:
            logger.info("Attempting evaluation with model: %s", model_id)
            model = _build_model(api_key, url, project_id, model_id)
            raw_response = _call_model(model, messages)
            logger.debug("Raw model response (first 300 chars): %s", raw_response[:300])
            break  # success — stop trying other models
        except Exception as exc:
            logger.warning("Model %s failed: %s", model_id, exc)
            last_error = exc
    else:
        return EvaluationReport(
            error=f"All model endpoints failed. Last error: {last_error}"
        )

    # ── Parse structured JSON ──────────────────────────────────────────────
    try:
        data = _extract_json(raw_response)
    except ValueError as exc:
        return EvaluationReport(
            detailed_analysis=raw_response,
            error=f"JSON parsing failed: {exc}",
            model_used=model_id,
        )

    # ── Populate report ───────────────────────────────────────────────────
    style_block = data.get("style_consistency", {})
    if isinstance(style_block, str):
        # Model collapsed the nested object to a plain string — normalise
        style_label = style_block
        style_explanation = ""
    else:
        style_label = style_block.get("label", "Unknown")
        style_explanation = style_block.get("explanation", "")

    style_str = style_label
    if style_explanation:
        style_str += f"\n{style_explanation}"

    return EvaluationReport(
        originality_score=float(data.get("originality_score", 0)),
        ai_likelihood_score=float(data.get("ai_likelihood_score", 0)),
        style_consistency=style_str,
        rubric_checklist=data.get("rubric_checklist", []),
        overall_verdict=data.get("overall_verdict", ""),
        detailed_analysis=data.get("detailed_analysis", raw_response),
        model_used=model_id,
    )
