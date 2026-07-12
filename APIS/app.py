"""
app.py  ── IntegriAI — Academic Integrity Intelligence Platform
═══════════════════════════════════════════════════════════════
Single-file Streamlit application with full UI/UX overhaul.

Navigation (authenticated):
  📊 Dashboard    — Instructor config & grading rubric
  🔍 Evaluate     — Upload + validate submissions
  🤖 AI Chat      — Free-form chat with the Granite model
  📚 Library      — Per-user record/history browser
"""

import os
import logging
import streamlit as st
from PIL import Image

# ── local utils ───────────────────────────────────────────────────────────────
from utils.cos_client      import COSClient
from utils.auth            import (
    authenticate_user, register_user, get_user, reset_password_by_email,
)
from utils.instructor_config import (
    GRADING_STYLES, load_config, save_config, build_rubric_constraint,
)
from utils.student_history import (
    list_students, get_student_roster, save_submission,
    build_stylistic_signature, add_demo_student,
    upload_history_to_ibm_cos,
)
from utils.document_parser import extract_text, truncate_text
from utils.ai_agent        import evaluate_submission, EvaluationReport, PRIMARY_MODEL_DISPLAY, PREFERRED_MODELS

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS & DESIGN SYSTEM
# ═════════════════════════════════════════════════════════════════════════════

CUSTOM_CSS = """
<style>
/* ── Google Font ──────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Root tokens ──────────────────────────────────────────────────────────── */
:root {
  --bg:          #0f0f1a;
  --surface:     #1a1a2e;
  --surface2:    #16213e;
  --border:      rgba(108,99,255,0.18);
  --accent:      #6c63ff;
  --accent2:     #a78bfa;
  --accent-glow: rgba(108,99,255,0.35);
  --text:        #e8e8f0;
  --muted:       #8b8ba7;
  --success:     #34d399;
  --warning:     #fbbf24;
  --danger:      #f87171;
  --radius:      14px;
  --radius-sm:   8px;
  --shadow:      0 8px 32px rgba(0,0,0,0.4);
  --font:        'Inter', system-ui, sans-serif;
  --mono:        'JetBrains Mono', monospace;
}

/* ── Base overrides ──────────────────────────────────────────────────────── */
html, body, [data-testid="stApp"] {
  background:
    radial-gradient(circle at 50% 8%, rgba(76,201,240,0.10), transparent 32rem),
    linear-gradient(180deg, #0b0b16 0%, var(--bg) 55%, #10131f 100%) !important;
  font-family: var(--font) !important;
  color: var(--text) !important;
  scroll-behavior: smooth;
}

/* Hide Streamlit's default chrome entirely — display:none (not
   visibility:hidden) so the elements are removed from layout/paint
   immediately, rather than just being invisible-but-present, which is
   what was causing the header/sidebar-arrow flash on load. */
#MainMenu,
footer,
header,
[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stStatusWidget"],
[data-testid="stDecoration"] {
  display: none !important;
  visibility: hidden !important;
  height: 0 !important;
}

/* Sidebar is permanently unused — top nav / login card handle all
   navigation. Hidden here, in the very first CSS injected after
   set_page_config(), so there is no flash-of-sidebar before this
   rule applies. */
[data-testid="stSidebar"],
[data-testid="collapsedControl"] {
  display: none !important;
  visibility: hidden !important;
}

/* Reclaim the top padding Streamlit normally reserves for its header,
   now that the header itself is gone. */
[data-testid="stAppViewContainer"] > .main {
  padding-top: 0 !important;
}

/* ── Sidebar ──────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 1px solid var(--border) !important;
  transition: all 0.3s ease;
}
[data-testid="stSidebar"] > div:first-child {
  padding: 1.5rem 1rem !important;
}

/* ── Logo area ────────────────────────────────────────────────────────────── */
.integri-logo {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 0.5rem 0 1rem 0;
  animation: fadeInDown 0.5s ease;
}
.integri-logo img {
  width: 54px !important;
  max-width: 54px !important;
  height: 540px !important;
  object-fit: contain !important;
  border-radius: 52px !important;
}
.integri-logo-icon {
  width: 760px;
  height: 760px;
  border-radius: 14px;
  background: linear-gradient(135deg, rgba(108,99,255,0.15) 0%, rgba(167,139,250,0.15) 100%);
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: 0 4px 14px var(--accent-glow);
  flex-shrink: 0;
  border: 1px solid rgba(167,139,250,0.3);
  transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.integri-logo-icon:hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 20px rgba(108,99,255,0.5);
}
.integri-logo-text {
  line-height: 1.15;
}
.integri-logo-name {
  font-size: 1.25rem;
  font-weight: 800;
  color: var(--text);
  letter-spacing: -0.3px;
  background: linear-gradient(135deg, #fff, var(--accent2));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
.integri-logo-tag {
  font-size: 0.7rem;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 1px;
  margin-top: 2px;
}

/* ── Nav buttons ─────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] .stButton > button {
  width: 100%;
  text-align: left !important;
  background: transparent !important;
  color: var(--muted) !important;
  border: none !important;
  border-radius: var(--radius-sm) !important;
  padding: 0.7rem 1rem !important;
  font-size: 0.9rem !important;
  font-weight: 500 !important;
  transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
  margin-bottom: 4px;
  position: relative;
  overflow: hidden;
}
[data-testid="stSidebar"] .stButton > button:hover {
  background: rgba(108,99,255,0.08) !important;
  color: var(--text) !important;
  transform: translateX(4px);
}
[data-testid="stSidebar"] .stButton > button[kind="primary"] {
  background: linear-gradient(90deg, rgba(108,99,255,0.2), rgba(167,139,250,0.05)) !important;
  color: var(--accent2) !important;
  border-left: 4px solid var(--accent) !important;
}

/* ── Main content Margins ────────────────────────────────────────────────── */
.block-container {
  padding: 3rem 4rem 4rem !important;
  max-width: 1200px !important;
  margin: 0 auto !important;
  animation: fadeIn 0.6s ease-out;
}
@media (max-width: 768px) {
  .block-container {
    padding: 1.5rem !important;
  }
}

/* ── Cards ───────────────────────────────────────────────────────────────── */
.card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.75rem;
  margin-bottom: 1.5rem;
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
  transition: transform 0.3s ease, box-shadow 0.3s ease;
  animation: fadeInUp 0.5s ease backwards;
}
.card:hover {
  transform: translateY(-3px);
  box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.2), 0 8px 10px -6px rgba(0, 0, 0, 0.1);
  border-color: rgba(167,139,250,0.4);
}
.card-glass {
  background: rgba(26,26,46,0.65);
  backdrop-filter: blur(12px);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.75rem;
  transition: all 0.3s ease;
}
.landing-card-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 1rem;
  margin-top: 1rem;
}
.landing-feature-card {
  min-height: 210px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: flex-start;
  text-align: center;
}
@media (max-width: 900px) {
  .landing-card-grid {
    grid-template-columns: 1fr;
  }
  .landing-feature-card {
    min-height: auto;
  }
}

/* ── Page titles ─────────────────────────────────────────────────────────── */
.page-header {
  display: flex;
  align-items: center;
  gap: 1rem;
  margin-bottom: 2.5rem;
  padding-bottom: 1.5rem;
  border-bottom: 1px solid rgba(108,99,255,0.1);
  animation: fadeInDown 0.6s ease backwards;
}
.page-header-icon {
  font-size: 2rem;
  background: rgba(108,99,255,0.1);
  padding: 10px;
  border-radius: 12px;
  display: flex;
}
.page-header h1 {
  font-size: 2rem !important;
  font-weight: 800 !important;
  color: var(--text) !important;
  margin: 0 !important;
  letter-spacing: -0.5px !important;
}
.page-header-sub {
  font-size: 0.9rem;
  color: var(--muted);
  margin-top: 4px;
}

/* ── Inputs ──────────────────────────────────────────────────────────────── */
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {
  background: rgba(0,0,0,0.2) !important;
  border: 1px solid rgba(255,255,255,0.1) !important;
  border-radius: var(--radius-sm) !important;
  color: var(--text) !important;
  font-family: var(--font) !important;
  padding: 0.75rem 1rem !important;
  transition: all 0.3s ease !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 4px rgba(108,99,255,0.15) !important;
  background: rgba(0,0,0,0.3) !important;
}

/* ── Buttons (main area) ─────────────────────────────────────────────────── */
.stButton > button {
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
}
.stButton > button[kind="primary"] {
  background: linear-gradient(135deg, var(--accent) 0%, #8b5cf6 100%) !important;
  color: white !important;
  border: none !important;
  border-radius: var(--radius-sm) !important;
  font-weight: 600 !important;
  letter-spacing: 0.3px !important;
  padding: 0.5rem 1.5rem !important;
  box-shadow: 0 4px 14px var(--accent-glow), inset 0 1px 0 rgba(255,255,255,0.2) !important;
}
.stButton > button[kind="primary"]:hover {
  transform: translateY(-2px) scale(1.02) !important;
  box-shadow: 0 6px 20px rgba(108,99,255,0.5), inset 0 1px 0 rgba(255,255,255,0.2) !important;
}
.stButton > button[kind="primary"]:active {
  transform: translateY(1px) !important;
}
.stButton > button[kind="secondary"] {
  background: rgba(255,255,255,0.03) !important;
  color: var(--text) !important;
  border: 1px solid rgba(255,255,255,0.1) !important;
  border-radius: var(--radius-sm) !important;
}
.stButton > button[kind="secondary"]:hover {
  background: rgba(255,255,255,0.08) !important;
  border-color: rgba(255,255,255,0.2) !important;
  transform: translateY(-1px) !important;
}

/* ── Metrics ─────────────────────────────────────────────────────────────── */
[data-testid="stMetric"] {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  padding: 1.5rem !important;
  box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
  transition: transform 0.3s ease, border-color 0.3s ease !important;
  animation: fadeInUp 0.5s ease backwards;
}
[data-testid="stMetric"]:hover {
  transform: translateY(-3px) !important;
  border-color: var(--accent) !important;
}
[data-testid="stMetricLabel"] { color: var(--muted) !important; font-size: 0.85rem !important; font-weight: 500 !important;}
[data-testid="stMetricValue"] { color: var(--text) !important; font-size: 2rem !important; font-weight: 800 !important; }

/* ── Progress bars ──────────────────────────────────────────────────────── */
[data-testid="stProgress"] > div > div {
  background: linear-gradient(90deg, var(--accent), var(--accent2)) !important;
  border-radius: 4px !important;
  box-shadow: 0 0 10px rgba(108,99,255,0.5) !important;
}

/* ── Chat messages ───────────────────────────────────────────────────────── */
[data-testid="stChatMessage"] {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  padding: 1.25rem !important;
  animation: fadeSlideUp 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
  margin-bottom: 1rem !important;
}

/* ── Expanders ───────────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
  background: var(--surface) !important;
  border: 1px solid rgba(255,255,255,0.05) !important;
  border-radius: var(--radius-sm) !important;
  transition: border-color 0.3s ease !important;
}
[data-testid="stExpander"]:hover {
  border-color: rgba(108,99,255,0.3) !important;
}
[data-testid="stExpander"] summary {
  color: var(--text) !important;
  font-weight: 600 !important;
  padding: 1rem !important;
}

/* ── Selectbox / Dropdown ────────────────────────────────────────────────── */
[data-testid="stSelectbox"] > div > div {
  background: rgba(0,0,0,0.2) !important;
  border: 1px solid rgba(255,255,255,0.1) !important;
  border-radius: var(--radius-sm) !important;
  color: var(--text) !important;
  transition: all 0.3s ease !important;
}
[data-testid="stSelectbox"] > div > div:hover {
  border-color: rgba(255,255,255,0.3) !important;
}

/* ── Tabs ────────────────────────────────────────────────────────────────── */
[data-testid="stTabs"] [role="tab"] {
  color: var(--muted) !important;
  font-weight: 600 !important;
  padding: 0.75rem 1.5rem !important;
  transition: color 0.3s ease !important;
}
[data-testid="stTabs"] [role="tab"]:hover {
  color: var(--text) !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
  color: var(--accent2) !important;
  border-bottom: 3px solid var(--accent) !important;
}

/* ── Alerts ──────────────────────────────────────────────────────────────── */
[data-testid="stAlert"] {
  border-radius: var(--radius-sm) !important;
  border: 1px solid rgba(255,255,255,0.1) !important;
  animation: fadeIn 0.4s ease !important;
}

/* ── Divider ─────────────────────────────────────────────────────────────── */
hr { 
  border-color: rgba(255,255,255,0.05) !important;
  margin: 2.5rem 0 !important;
}

/* ── Score badges ────────────────────────────────────────────────────────── */
.score-badge {
  display: inline-block;
  padding: 0.3rem 0.8rem;
  border-radius: 20px;
  font-size: 0.8rem;
  font-weight: 700;
  letter-spacing: 0.5px;
  box-shadow: 0 2px 5px rgba(0,0,0,0.2);
}
.score-good  { background: rgba(52,211,153,0.15); color: #34d399; border: 1px solid rgba(52,211,153,0.3); }
.score-warn  { background: rgba(251,191,36,0.15);  color: #fbbf24; border: 1px solid rgba(251,191,36,0.3); }
.score-bad   { background: rgba(248,113,113,0.15); color: #f87171; border: 1px solid rgba(248,113,113,0.3); }

/* ── Library record cards ────────────────────────────────────────────────── */
.record-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  padding: 1.25rem 1.5rem;
  margin-bottom: 0.85rem;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  animation: fadeInUp 0.4s ease backwards;
}
.record-card:hover { 
  border-color: var(--accent); 
  transform: translateX(4px) translateY(-2px);
  box-shadow: -4px 4px 15px rgba(0,0,0,0.2);
}

/* ── User profile chip ───────────────────────────────────────────────────── */
.profile-chip {
  display: flex;
  align-items: center;
  gap: 0.85rem;
  padding: 0.75rem 1rem;
  background: rgba(108,99,255,0.08);
  border: 1px solid rgba(108,99,255,0.2);
  border-radius: var(--radius);
  margin-bottom: 1rem;
  transition: all 0.3s ease;
}
.profile-chip:hover {
  background: rgba(108,99,255,0.12);
  border-color: rgba(108,99,255,0.4);
}
.profile-avatar {
  width: 38px;
  height: 38px;
  border-radius: 50%;
  background: linear-gradient(135deg, var(--accent), #a78bfa);
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: 800;
  font-size: 0.95rem;
  color: white;
  flex-shrink: 0;
  box-shadow: 0 2px 8px rgba(108,99,255,0.4);
}
.profile-name { font-size: 0.95rem; font-weight: 700; color: var(--text); }
.profile-role { font-size: 0.75rem; color: var(--muted); text-transform: capitalize; font-weight: 500;}
.sidebar-mini-nav {
  display: grid;
  gap: 0.75rem;
  margin-top: 0.75rem;
}
.sidebar-nav-card {
  background: rgba(108,99,255,0.08);
  border: 1px solid rgba(108,99,255,0.2);
  border-radius: var(--radius-sm);
  padding: 0.85rem 0.95rem;
  color: var(--text);
  animation: slideInSidebar 0.45s ease backwards;
}
.sidebar-nav-card:nth-child(2) { animation-delay: 0.08s; }
.sidebar-nav-card:nth-child(3) { animation-delay: 0.16s; }
.sidebar-nav-card-title {
  font-size: 0.88rem;
  font-weight: 700;
  margin-bottom: 0.25rem;
}
.sidebar-nav-card-copy {
  color: var(--muted);
  font-size: 0.76rem;
  line-height: 1.45;
}
.sidebar-login-hint {
  margin-top: 1rem;
  padding: 0.9rem;
  border-radius: var(--radius-sm);
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(255,255,255,0.08);
  color: var(--muted);
  font-size: 0.78rem;
  line-height: 1.45;
}

/* ── Login page specific ─────────────────────────────────────────────────── */
.landing-nav-pane {
  margin: 1rem auto 0;
  max-width: 920px;
  padding: 1.25rem;
  border-radius: var(--radius);
  background: linear-gradient(135deg, rgba(15,23,42,0.92), rgba(20,36,54,0.74));
  border: 1px solid rgba(76,201,240,0.24);
  box-shadow: 0 16px 40px rgba(0,0,0,0.22);
  animation: fadeInUp 0.65s ease both;
}
.landing-nav-title {
  color: var(--text);
  font-size: 1.05rem;
  font-weight: 800;
  margin-bottom: 0.25rem;
}
.landing-nav-copy {
  color: #9fb4c7;
  font-size: 0.86rem;
  line-height: 1.55;
  margin-bottom: 1rem;
}
.landing-nav-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 0.75rem;
}
.landing-nav-item {
  min-height: 112px;
  padding: 0.95rem;
  border-radius: var(--radius-sm);
  background: rgba(255,255,255,0.045);
  border: 1px solid rgba(255,255,255,0.09);
}
.landing-nav-icon {
  font-size: 1.35rem;
  margin-bottom: 0.45rem;
}
.landing-nav-name {
  color: var(--text);
  font-weight: 800;
  font-size: 0.9rem;
  margin-bottom: 0.25rem;
}
.landing-nav-desc {
  color: #8fa2b5;
  font-size: 0.76rem;
  line-height: 1.4;
}
@media (max-width: 860px) {
  .landing-nav-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}
@media (max-width: 540px) {
  .landing-nav-grid {
    grid-template-columns: 1fr;
  }
}

.login-hero {
  text-align: center;
  padding: 4rem 1rem 3rem;
  animation: fadeInDown 0.8s ease;
}
.login-hero-icon {
  font-size: 4rem;
  margin-bottom: 1rem;
  animation: floatIcon 4s ease-in-out infinite;
  display: inline-block;
}
.login-hero-icon img {
  width: min(220px, 78vw) !important;
  max-height: 160px !important;
  object-fit: contain !important;
}
.login-hero h1 {
  font-size: 2.75rem !important;
  font-weight: 800 !important;
  background: linear-gradient(135deg, #fff 0%, var(--accent2) 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin: 0 !important;
  letter-spacing: -1px;
}
.login-hero p {
  color: var(--muted);
  font-size: 1.1rem;
  margin-top: 0.5rem;
  max-width: 400px;
  margin-left: auto;
  margin-right: auto;
  line-height: 1.5;
}
.feature-pills {
  display: flex;
  flex-wrap: wrap;
  gap: 0.75rem;
  justify-content: center;
  margin-top: 2rem;
  animation: fadeInUp 0.8s ease 0.2s backwards;
}
.feature-pill {
  background: rgba(108,99,255,0.1);
  border: 1px solid rgba(108,99,255,0.3);
  border-radius: 30px;
  padding: 0.4rem 1rem;
  font-size: 0.85rem;
  color: var(--accent2);
  font-weight: 600;
  transition: all 0.3s ease;
}
.feature-pill:hover {
  background: rgba(108,99,255,0.2);
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(108,99,255,0.2);
}
/* ── Spinner / AI processing animation ──────────────────────────────────── */
.ai-processing {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  padding: 1rem 1.5rem;
  background: rgba(108,99,255,0.08);
  border: 1px solid rgba(108,99,255,0.25);
  border-radius: var(--radius-sm);
  animation: glowPulse 2s ease-in-out infinite;
}
.ai-processing-dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background: var(--accent2);
  animation: dotPulse 1.4s ease-in-out infinite;
}
.ai-processing-dot:nth-child(2) { animation-delay: 0.2s; }
.ai-processing-dot:nth-child(3) { animation-delay: 0.4s; }

/* ── Keyframes ───────────────────────────────────────────────────────────── */
@keyframes fadeIn {
  from { opacity: 0; }
  to   { opacity: 1; }
}
@keyframes fadeInUp {
  from { opacity: 0; transform: translateY(15px); }
  to   { opacity: 1; transform: translateY(0); }
}
@keyframes fadeInDown {
  from { opacity: 0; transform: translateY(-15px); }
  to   { opacity: 1; transform: translateY(0); }
}
@keyframes fadeSlideUp {
  from { opacity: 0; transform: translateY(10px); }
  to   { opacity: 1; transform: translateY(0); }
}
@keyframes floatIcon {
  0%, 100%  { transform: translateY(0px) rotate(0deg); }
  50%       { transform: translateY(-8px) rotate(2deg); }
}
@keyframes glowPulse {
  0%, 100% { box-shadow: 0 0 0 0 rgba(108,99,255,0.1); }
  50%      { box-shadow: 0 0 20px 0 rgba(108,99,255,0.3); }
}
@keyframes dotPulse {
  0%, 100% { transform: scale(0.8); opacity: 0.5; }
  50%      { transform: scale(1.2); opacity: 1; }
}
@keyframes slideInSidebar {
  from { transform: translateX(-30px); opacity: 0; }
  to   { transform: translateX(0);     opacity: 1; }
}
@keyframes logoutFade {
  from { opacity: 1; transform: scale(1); }
  to   { opacity: 0; transform: scale(0.98); }
}
.sidebar-animate {
  animation: slideInSidebar 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275);
}

/* ── Top nav bar ─────────────────────────────────────────────────────────── */
div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] > div .stButton > button {
  border-radius: var(--radius-sm) !important;
}
.topnav-brand img {
  width: 64px !important;
  height: 64px !important;
  object-fit: contain !important;
  border-radius: 12px !important;
  box-shadow: 0 3px 10px rgba(0,0,0,0.25);
}
</style>
"""

import base64
import io

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
LOGO_PATH = os.path.join(ASSETS_DIR, "logo.png")
DASHBOARD_HERO_PATH = os.path.join(ASSETS_DIR, "dashboard_hero.png")


def _load_image_safe(path: str, label: str):
    """Load an image into memory as a PIL.Image, with detailed diagnostics
    on failure (rather than a bare emoji fallback with no explanation).

    Returns (PIL.Image | None, diagnostic_message | None).
    """
    if os.path.exists(path):
        try:
            img = Image.open(path)
            img.load()  # force-read pixel data now so any error surfaces here
            return img, None
        except Exception as e:
            msg = f"{label}: file exists at `{path}` but could not be opened as an image ({e})."
            logger.warning(msg)
            return None, msg

    # File wasn't found at the expected path — look for likely causes.
    if not os.path.isdir(ASSETS_DIR):
        msg = (
            f"{label}: the `assets/` folder itself is missing "
            f"(expected at `{ASSETS_DIR}`). It may not have been committed/deployed."
        )
        logger.warning(msg)
        return None, msg

    # assets/ exists but the exact filename doesn't — check for a
    # case-mismatch or near-match, which is a common cross-platform bug
    # (works on Windows/Mac, breaks on case-sensitive Linux deployments).
    wanted = os.path.basename(path).lower()
    try:
        siblings = os.listdir(ASSETS_DIR)
    except Exception:
        siblings = []
    close_match = next((f for f in siblings if f.lower() == wanted), None)

    if close_match:
        msg = (
            f"{label}: found `{close_match}` in `assets/` but the code looks for "
            f"`{os.path.basename(path)}` — filename casing doesn't match "
            f"(this passes on Windows/Mac, fails on case-sensitive Linux deployments)."
        )
    else:
        msg = (
            f"{label}: `{os.path.basename(path)}` not found in `assets/`. "
            f"Files present: {siblings or '(folder is empty)'}"
        )
    logger.warning(msg)
    return None, msg


LOGO_IMAGE, LOGO_LOAD_ERROR = _load_image_safe(LOGO_PATH, "Logo")
DASHBOARD_HERO_IMAGE, DASHBOARD_HERO_LOAD_ERROR = _load_image_safe(
    DASHBOARD_HERO_PATH, "Dashboard hero image"
)

# The sidebar logo / login logo / topnav logo are embedded inline as base64
# <img> tags (not st.image), so they still need raw bytes — read those from
# the already-validated PIL image via an in-memory buffer rather than
# re-opening the file path a second time.
if LOGO_IMAGE is not None:
    _buf = io.BytesIO()
    LOGO_IMAGE.save(_buf, format="PNG")
    _b64 = base64.b64encode(_buf.getvalue()).decode()
    LOGO_SVG = f'<img src="data:image/png;base64,{_b64}" alt="IntegriAI logo">'
    LOGIN_LOGO = f'<img src="data:image/png;base64,{_b64}" alt="IntegriAI logo">'
    TOPNAV_LOGO = f'<img src="data:image/png;base64,{_b64}" alt="IntegriAI logo">'
else:
    LOGO_SVG = "🛡️"
    LOGIN_LOGO = "🛡️"
    TOPNAV_LOGO = "🛡️"

# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _get_secret(key: str, fallback_env: str = "") -> str:
    try:
        return st.secrets[key]
    except (KeyError, FileNotFoundError):
        return os.environ.get(fallback_env or key, "")


def _load_credentials() -> dict:
    return {
        "watsonx_api_key": _get_secret("WATSONX_APIKEY"),
        "watsonx_url":     _get_secret("WATSONX_URL") or "https://eu-de.ml.cloud.ibm.com",
        "project_id":      _get_secret("PROJECT_ID"),
        "cos_endpoint":    _get_secret("COS_ENDPOINT"),
        "cos_api_key":     _get_secret("COS_API_KEY_ID"),
        "cos_crn":         _get_secret("COS_INSTANCE_CRN"),
        "cos_bucket":      _get_secret("COS_BUCKET_NAME") or "plagiarism-intelligence",
    }


def _credentials_valid(creds: dict) -> tuple[bool, list[str]]:
    required = ["watsonx_api_key", "project_id", "cos_endpoint", "cos_api_key", "cos_crn"]
    missing = [k for k in required if not creds.get(k)]
    return (len(missing) == 0, missing)


# ═════════════════════════════════════════════════════════════════════════════
# COS CLIENT
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Connecting to cloud storage…")
def get_cos(endpoint: str, api_key: str, crn: str, bucket: str) -> COSClient:
    return COSClient(endpoint=endpoint, api_key=api_key,
                     instance_crn=crn, bucket=bucket)


# ═════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═════════════════════════════════════════════════════════════════════════════

def _init_session():
    defaults = {
        "authenticated":    False,
        "user_profile":     None,
        "active_page":      "Dashboard",
        "chat_messages":    [],
        "last_report":      None,
        "submission_text":  "",
        "selected_student": None,
        "selected_class":   "DEMO101",
        "ai_chat_messages": [],
        "show_forgot_pw":   False,
        "_login_error":     None,
        "_register_error":  None,
        "_register_success": None,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════

def render_sidebar(cos: COSClient):
    with st.sidebar:
        # ── Logo ──────────────────────────────────────────────────────────────
        st.markdown(
            f'<div class="integri-logo sidebar-animate">'
            f'  {LOGO_SVG}'
            f'  <div class="integri-logo-text">'
            f'    <div class="integri-logo-name">IntegriAI</div>'
            f'    <div class="integri-logo-tag">Academic Integrity</div>'
            f'  </div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.divider()

        if not st.session_state["authenticated"]:
            # FIX: previously called _render_auth_panel(cos) here, which drew a
            # second full login/register form in the sidebar — duplicating the
            # one already rendered by page_login_landing() in the main area.
            # The informational panel below points users to that main-area form
            # instead of re-implementing login in two places.
            _render_public_nav_panel()
        else:
            _render_nav_panel()


def _render_public_nav_panel():
    """Pre-login navigation/overview pane."""
    st.markdown(
        """
        <div class="sidebar-mini-nav sidebar-animate">
          <div class="sidebar-nav-card">
            <div class="sidebar-nav-card-title">Dashboard</div>
            <div class="sidebar-nav-card-copy">Configure rubrics, grading weights, and academic integrity controls after sign in.</div>
          </div>
          <div class="sidebar-nav-card">
            <div class="sidebar-nav-card-title">Evaluate</div>
            <div class="sidebar-nav-card-copy">Upload student work and run AI, style, and rubric checks from one workspace.</div>
          </div>
          <div class="sidebar-nav-card">
            <div class="sidebar-nav-card-title">Library</div>
            <div class="sidebar-nav-card-copy">Review saved submissions and student writing history in the authenticated area.</div>
          </div>
        </div>
        <div class="sidebar-login-hint">
          Use the sign-in panel on the main page to launch the app.
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_auth_panel(cos: COSClient):
    """Login / Register / Forgot Password tabs.

    NOTE: no longer called from render_sidebar() — kept here in case a
    sidebar-based login is wanted again in the future. The live login flow
    lives in page_login_landing().
    """
    if st.session_state.get("show_forgot_pw"):
        _render_forgot_password(cos)
        return

    tab_login, tab_register = st.tabs(["Sign In", "Create Account"])

    # ── Login tab ──────────────────────────────────────────────────────────
    with tab_login:
        username = st.text_input("Username", key="login_user",
                                 placeholder="e.g. prof_smith",
                                 label_visibility="visible")
        password = st.text_input("Password", type="password",
                                 key="login_pass",
                                 placeholder="••••••••")

        col_login, col_forgot = st.columns([3, 2])
        with col_login:
            do_login = st.button("Sign In", use_container_width=True, type="primary")
        with col_forgot:
            if st.button("Forgot?", use_container_width=True):
                st.session_state["show_forgot_pw"] = True
                st.rerun()

        if do_login:
            if not username or not password:
                st.error("Both fields required.")
            else:
                with st.spinner("Verifying…"):
                    ok, profile = authenticate_user(cos, username, password)
                if ok:
                    st.session_state["authenticated"] = True
                    st.session_state["user_profile"]  = profile
                    st.session_state["active_page"]   = "Dashboard"
                    st.rerun()
                else:
                    st.error("Invalid credentials.")

    # ── Register tab ───────────────────────────────────────────────────────
    with tab_register:
        r_full  = st.text_input("Full Name",  key="reg_name",
                                placeholder="Dr. Alice Smith")
        r_email = st.text_input("Email",      key="reg_email",
                                placeholder="alice@university.edu")
        r_user  = st.text_input("Username",   key="reg_user",
                                placeholder="prof_alice")
        r_pass  = st.text_input("Password",   type="password", key="reg_pass1",
                                placeholder="At least 6 characters")
        r_pass2 = st.text_input("Confirm",    type="password", key="reg_pass2",
                                placeholder="Repeat password")
        if st.button("Create Account", use_container_width=True, type="primary"):
            if not r_full.strip():
                st.error("Full Name is required.")
            elif not r_email.strip() or "@" not in r_email:
                st.error("A valid email address is required.")
            elif not r_user.strip():
                st.error("Username is required.")
            elif len("".join(c for c in r_user if c.isalnum() or c in ("_", "-"))) == 0:
                st.error("Username may only contain letters, numbers, _ or -.")
            elif r_pass != r_pass2:
                st.error("Passwords do not match.")
            elif len(r_pass) < 6:
                st.error("Password must be at least 6 characters.")
            else:
                with st.spinner("Creating account…"):
                    ok, err = register_user(cos, r_user, r_pass,
                                            r_full, r_email, "instructor")
                if ok:
                    st.success("Account created! Sign in above.")
                else:
                    st.error(err or "Registration failed.")


def _render_forgot_password(cos: COSClient):
    """Inline forgot-password flow inside the sidebar."""
    st.markdown("**Reset Password**")
    st.caption("Enter the email address linked to your account.")

    fp_email  = st.text_input("Email", key="fp_email",
                               placeholder="alice@university.edu")
    fp_pass1  = st.text_input("New Password", type="password", key="fp_pass1",
                               placeholder="New password")
    fp_pass2  = st.text_input("Confirm New Password", type="password", key="fp_pass2",
                               placeholder="Repeat password")

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Reset", use_container_width=True, type="primary"):
            if not fp_email.strip():
                st.error("Email is required.")
            elif fp_pass1 != fp_pass2:
                st.error("Passwords do not match.")
            elif len(fp_pass1) < 6:
                st.error("Min 6 characters.")
            else:
                with st.spinner("Resetting…"):
                    ok, result = reset_password_by_email(cos, fp_email, fp_pass1)
                if ok:
                    st.success(f"Password reset for **{result}**. Sign in above.")
                    st.session_state["show_forgot_pw"] = False
                    st.rerun()
                else:
                    st.error(result)
    with col_b:
        if st.button("Cancel", use_container_width=True):
            st.session_state["show_forgot_pw"] = False
            st.rerun()


def _render_nav_panel():
    """Logged-in user profile chip + navigation."""
    profile = st.session_state["user_profile"]
    name    = profile.get("full_name", profile["username"])
    initials = "".join(w[0] for w in name.split()[:2]).upper()
    role   = profile.get("role", "instructor")

    st.markdown(
        f'<div class="profile-chip sidebar-animate">'
        f'  <div class="profile-avatar">{initials}</div>'
        f'  <div>'
        f'    <div class="profile-name">{name}</div>'
        f'    <div class="profile-role">{role}</div>'
        f'  </div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    pages = [
        ("📊", "Dashboard",  "Dashboard"),
        ("🔍", "Evaluate",   "Evaluate"),
        ("🤖", "AI Chat",    "AIChat"),
        ("📚", "Library",    "Library"),
    ]
    for icon, label, page_key in pages:
        active = st.session_state["active_page"] == page_key
        btn_label = f"{icon}  {label}"
        if st.button(btn_label, use_container_width=True,
                     type="primary" if active else "secondary",
                     key=f"nav_{page_key}"):
            st.session_state["active_page"] = page_key
            st.rerun()

    st.divider()
    if st.button("⏻  Sign Out", use_container_width=True):
        st.markdown(
            '<style>[data-testid="stApp"]{animation:logoutFade 0.5s ease forwards;}</style>',
            unsafe_allow_html=True,
        )
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# HELPER: Page header
# ═════════════════════════════════════════════════════════════════════════════

def _page_header(icon: str, title: str, subtitle: str = ""):
    sub_html = f'<div class="page-header-sub">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f'<div class="page-header">'
        f'  <div class="page-header-icon">{icon}</div>'
        f'  <div><h1>{title}</h1>{sub_html}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════

def page_dashboard(cos: COSClient):
    # ── Hero Section ──────────────────────────────────────────────────────────
    col_text, col_img = st.columns([1.2, 1], gap="large")
    
    with col_text:
        st.markdown(
            '<div style="padding-top: 1rem; animation: fadeIn 0.6s ease;">'
            '<div style="font-size:0.85rem;color:var(--accent2);text-transform:uppercase;font-weight:700;letter-spacing:1px;margin-bottom:0.5rem;">🛡️ AI-Powered · IBM WatsonX</div>'
            '<h1 style="font-size:2.6rem;font-weight:800;line-height:1.2;margin-bottom:1rem;background:linear-gradient(135deg, #fff, var(--accent2));-webkit-background-clip:text;-webkit-text-fill-color:transparent;">INTEGRIAI GRADING<br>DASHBOARD</h1>'
            '<p style="color:var(--muted);font-size:1.05rem;line-height:1.6;margin-bottom:1.5rem;">Configure your automated grading rubric and adjust AI evaluation weights to detect nuanced forms of plagiarism. Set parameters for original content, stylistic signature matching, and AI detection.</p>'
            '</div>',
            unsafe_allow_html=True
        )
    
    with col_img:
        if DASHBOARD_HERO_IMAGE is not None:
            st.image(DASHBOARD_HERO_IMAGE, use_container_width=True)
        elif DASHBOARD_HERO_LOAD_ERROR:
            st.info(f"🖼️ Hero image not loaded — {DASHBOARD_HERO_LOAD_ERROR}")

    st.markdown("<hr>", unsafe_allow_html=True)

    username = st.session_state["user_profile"]["username"]
    existing = load_config(cos, username) or {}

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("**Grading Style**")
        grading_style = st.selectbox(
            "Primary Grading Style",
            options=GRADING_STYLES,
            index=GRADING_STYLES.index(existing.get("grading_style", GRADING_STYLES[0])),
            label_visibility="collapsed",
        )

        st.markdown("**Rubric Notes**")
        rubric_notes = st.text_area(
            "Rubric Notes",
            value=existing.get("rubric_notes", ""),
            placeholder="e.g. Students must cite ≥3 primary sources, include a system diagram, define all key terms.",
            height=130,
            label_visibility="collapsed",
        )

        st.markdown("**Custom Rubric Tags** *(comma-separated)*")
        custom_tags_raw = st.text_input(
            "Custom Tags",
            value=", ".join(existing.get("custom_tags", [])),
            placeholder="cite sources, include diagrams, define terms",
            label_visibility="collapsed",
        )
        custom_tags = [t.strip() for t in custom_tags_raw.split(",") if t.strip()]

    with col2:
        st.markdown("**Score Weighting** *(must total 100 %)*")
        w_orig = st.slider("Originality Weight %",   0, 100,
                           existing.get("weight_originality", 40), step=5)
        w_ai   = st.slider("AI Detection Weight %",  0, 100,
                           existing.get("weight_ai_detection", 35), step=5)
        w_sty  = st.slider("Style Match Weight %",   0, 100,
                           existing.get("weight_style", 25), step=5)
        total  = w_orig + w_ai + w_sty

        badge_cls  = "score-good" if total == 100 else "score-bad"
        badge_text = f"Total: {total}%" + (" ✓" if total == 100 else " ✗")
        st.markdown(
            f'<span class="score-badge {badge_cls}">{badge_text}</span>',
            unsafe_allow_html=True,
        )
        if total != 100:
            st.warning(f"Weights sum to {total}% — adjust sliders to reach 100%.")

    st.divider()

    with st.expander("Preview AI Prompt Constraint", expanded=False):
        preview_config = {
            "grading_style":       grading_style,
            "custom_tags":         custom_tags,
            "rubric_notes":        rubric_notes,
            "weight_originality":  w_orig,
            "weight_ai_detection": w_ai,
            "weight_style":        w_sty,
        }
        st.code(build_rubric_constraint(preview_config), language="text")

    if st.button("💾  Save Configuration", type="primary"):
        ok, err = save_config(cos, username, preview_config)
        if ok:
            st.success("Configuration saved.")
        else:
            st.error(f"Save failed: {err}")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — EVALUATE
# ═════════════════════════════════════════════════════════════════════════════

def page_evaluate(cos: COSClient, creds: dict):
    _page_header("🔍", "Evaluate", "Upload a submission and run three-layer AI analysis")
    st.markdown(
        f'<div style="display:inline-flex;align-items:center;gap:0.5rem;'
        f'background:rgba(108,99,255,0.08);border:1px solid rgba(108,99,255,0.25);'
        f'border-radius:20px;padding:0.3rem 0.9rem;margin-bottom:1rem;">'
        f'  <span style="font-size:0.75rem;color:var(--muted);font-weight:500;">Active model</span>'
        f'  <span style="font-size:0.8rem;font-weight:700;color:var(--accent2);">{PRIMARY_MODEL_DISPLAY}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    col_student, col_class = st.columns([3, 1])

    with col_class:
        class_id = st.text_input(
            "Class ID",
            value=st.session_state.get("selected_class", "DEMO101"),
            help="Course identifier for student history lookup.",
        )
        st.session_state["selected_class"] = class_id

    with col_student:
        with st.spinner("Loading roster…"):
            add_demo_student(cos, class_id)
            students = list_students(cos, class_id)

        student_options = {
            s.get("full_name", s["student_id"]): s["student_id"]
            for s in students
        }
        student_options["— No student selected —"] = None
        display_names = list(student_options.keys())
        selected_name = st.selectbox(
            "Student History Profile",
            options=display_names,
            index=len(display_names) - 1,
        )
        selected_id = student_options[selected_name]
        st.session_state["selected_student"] = selected_id

    if selected_id:
        with st.expander("📖 Stylistic Signature (RAG Context)", expanded=False):
            sig = build_stylistic_signature(cos, class_id, selected_id)
            st.text(sig[:3000] + ("…" if len(sig) > 3000 else ""))

    st.divider()
    st.markdown("**Upload Submission**")
    uploaded = st.file_uploader(
        "Drop the student's assignment here",
        type=["txt", "md", "pdf"],
        label_visibility="collapsed",
    )

    if uploaded:
        with st.spinner(f"Parsing '{uploaded.name}'…"):
            text, parse_error = extract_text(uploaded)
        if parse_error:
            st.error(f"Parse error: {parse_error}")
        else:
            st.session_state["submission_text"] = truncate_text(text)
            char_count = len(st.session_state["submission_text"])
            st.success(f"Loaded **{uploaded.name}** — {char_count:,} characters")
            with st.expander("Preview"):
                st.text(st.session_state["submission_text"][:1500])

    st.divider()
    st.markdown("**AI Evaluation Chat**")

    for msg in st.session_state["chat_messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    col_btn, col_hint = st.columns([2, 5])
    with col_btn:
        run_validation = st.button(
            "✅  Validate Contextually",
            type="primary",
            disabled=not bool(st.session_state.get("submission_text")),
        )
    with col_hint:
        if not st.session_state.get("submission_text"):
            st.info("Upload an assignment file to enable validation.")

    if run_validation:
        _run_evaluation(cos, creds, class_id)

    user_input = st.chat_input(
        "Ask a follow-up about the report…",
        disabled=not bool(st.session_state.get("last_report")),
    )
    if user_input:
        _handle_chat_followup(user_input, creds)

    if st.session_state.get("last_report"):
        _render_report(st.session_state["last_report"])


def _run_evaluation(cos: COSClient, creds: dict, class_id: str):
    submission_text = st.session_state["submission_text"]
    selected_id     = st.session_state.get("selected_student")
    username        = st.session_state["user_profile"]["username"]

    with st.spinner("Pulling stylistic signature…"):
        sig = (
            build_stylistic_signature(cos, class_id, selected_id)
            if selected_id
            else "NO HISTORICAL DATA: No student profile selected."
        )

    with st.spinner("Loading instructor rubric…"):
        instructor_config = load_config(cos, username)
        rubric = build_rubric_constraint(instructor_config)

    user_msg = (
        "📤 **New submission received.** Running three-layer contextual evaluation…\n\n"
        f"- Student profile: **{selected_id or 'None'}**\n"
        f"- Submission size: **{len(submission_text):,} characters**\n"
        f"- Instructor rubric: **{'configured' if instructor_config else 'default'}**"
    )
    st.session_state["chat_messages"].append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.markdown(user_msg)

    with st.chat_message("assistant"):
        # AI processing animation
        st.markdown(
            '<div class="ai-processing">'
            '  <div class="ai-processing-dot"></div>'
            '  <div class="ai-processing-dot"></div>'
            '  <div class="ai-processing-dot"></div>'
            '  <span style="font-size:0.85rem;color:var(--muted)">Granite AI is analysing…</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        with st.spinner("Granite AI is evaluating the submission…"):
            report: EvaluationReport = evaluate_submission(
                api_key=creds["watsonx_api_key"],
                url=creds["watsonx_url"],
                project_id=creds["project_id"],
                submission_text=submission_text,
                stylistic_signature=sig,
                rubric_constraint=rubric,
            )
        st.session_state["last_report"] = report

        if report.error:
            msg = f"⚠️ **Evaluation Error**\n\n{report.error}"
            st.error(msg)
            st.session_state["chat_messages"].append({"role": "assistant", "content": msg})
        else:
            summary = _report_summary_markdown(report)
            st.markdown(summary)
            st.session_state["chat_messages"].append({"role": "assistant", "content": summary})

    if selected_id and not report.error:
        import time as _time
        aid = f"eval_{int(_time.time())}"
        try:
            save_submission(cos, class_id, selected_id, aid,
                            "Evaluated Submission", submission_text)
        except Exception as exc:
            logger.warning("Could not persist submission: %s", exc)

    st.rerun()


def _handle_chat_followup(user_input: str, creds: dict):
    from utils.ai_agent import _build_model, _call_model, PREFERRED_MODELS

    st.session_state["chat_messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    report: EvaluationReport = st.session_state.get("last_report")
    report_context = ""
    if report and not report.error:
        report_context = (
            f"Originality Score: {report.originality_score}%\n"
            f"AI Likelihood Score: {report.ai_likelihood_score}%\n"
            f"Style Consistency: {report.style_consistency}\n"
            f"Overall Verdict: {report.overall_verdict}\n"
            f"Detailed Analysis: {report.detailed_analysis}"
        )

    messages = [
        {
            "role": "system",
            "content": (
                "You are an academic integrity assistant. Answer follow-up questions "
                "about the plagiarism evaluation report concisely and helpfully.\n"
                "Keep your answers short, direct, and under 2-3 paragraphs max.\n\n"
                f"REPORT CONTEXT:\n{report_context}"
            ),
        },
        {"role": "user", "content": user_input},
    ]

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                for model_id in PREFERRED_MODELS:
                    try:
                        model = _build_model(
                            creds["watsonx_api_key"], creds["watsonx_url"],
                            creds["project_id"], model_id, max_new_tokens=400
                        )
                        reply = _call_model(model, messages)
                        break
                    except Exception:
                        continue
                else:
                    reply = "⚠️ Could not reach the AI model. Please retry."
            except Exception as exc:
                reply = f"⚠️ Error: {exc}"
        st.markdown(reply)
        st.session_state["chat_messages"].append({"role": "assistant", "content": reply})


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 — AI CHAT (standalone)
# ═════════════════════════════════════════════════════════════════════════════

def page_ai_chat(creds: dict):
    _page_header("🤖", "AI Chat", f"Free-form conversation powered by {PRIMARY_MODEL_DISPLAY}")
    st.markdown(
        f'<div style="display:inline-flex;align-items:center;gap:0.5rem;'
        f'background:rgba(108,99,255,0.08);border:1px solid rgba(108,99,255,0.25);'
        f'border-radius:20px;padding:0.3rem 0.9rem;margin-bottom:1rem;">'
        f'  <span style="font-size:0.75rem;color:var(--muted);font-weight:500;">Active model</span>'
        f'  <span style="font-size:0.8rem;font-weight:700;color:var(--accent2);">{PRIMARY_MODEL_DISPLAY}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    from utils.ai_agent import _build_model, _call_model, PREFERRED_MODELS

    # Render history
    for msg in st.session_state["ai_chat_messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if not st.session_state["ai_chat_messages"]:
        st.markdown(
            '<div class="card" style="text-align:center;padding:2rem;">'
            '<div style="font-size:2.5rem;margin-bottom:0.5rem">🤖</div>'
            '<div style="font-size:1rem;font-weight:600;color:var(--text)">Start a conversation</div>'
            '<div style="font-size:0.85rem;color:var(--muted);margin-top:0.3rem">'
            'Ask anything about academic integrity, plagiarism detection, or writing analysis.</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    user_input = st.chat_input("Ask IntegriAI anything…")
    if user_input:
        st.session_state["ai_chat_messages"].append(
            {"role": "user", "content": user_input}
        )
        with st.chat_message("user"):
            st.markdown(user_input)

        system_msg = (
            "You are IntegriAI, an expert academic integrity and plagiarism detection assistant. "
            "Answer questions about plagiarism, AI-generated text, "
            "writing style analysis, academic integrity policies, and related topics. "
            "IMPORTANT: Be concise, accurate, and helpful. Keep responses under 2-3 paragraphs and avoid long essays."
        )

        messages = [
            {"role": "system", "content": system_msg},
        ]
        # Append recent history (last 10 turns) for context
        for m in st.session_state["ai_chat_messages"][-10:]:
            messages.append(m)

        with st.chat_message("assistant"):
            st.markdown(
                '<div class="ai-processing">'
                '<div class="ai-processing-dot"></div>'
                '<div class="ai-processing-dot"></div>'
                '<div class="ai-processing-dot"></div>'
                '<span style="font-size:0.82rem;color:var(--muted)">AI is thinking…</span>'
                '</div>',
                unsafe_allow_html=True,
            )
            with st.spinner(""):
                try:
                    last_model_error = None
                    for model_id in PREFERRED_MODELS:
                        try:
                            model = _build_model(
                                creds["watsonx_api_key"], creds["watsonx_url"],
                                creds["project_id"], model_id, max_new_tokens=400
                            )
                            reply = _call_model(model, messages)
                            last_model_error = None
                            break
                        except Exception as model_exc:
                            logger.warning("AI Chat — model %s failed: %s", model_id, model_exc)
                            last_model_error = model_exc
                            continue
                    else:
                        reply = (
                            f"⚠️ AI model is unavailable. Last error: `{last_model_error}`\n\n"
                            "**Possible causes:**\n"
                            "- Invalid or expired WatsonX API key\n"
                            "- Wrong `WATSONX_URL` region (check your IBM Cloud region)\n"
                            "- Project ID not linked to WatsonX.ai\n"
                            "- Model not available on your plan"
                        )
                except Exception as exc:
                    logger.error("AI Chat — unexpected error: %s", exc, exc_info=True)
                    reply = f"⚠️ Unexpected error: {exc}"
            st.markdown(reply)
            st.session_state["ai_chat_messages"].append(
                {"role": "assistant", "content": reply}
            )

    if st.session_state["ai_chat_messages"]:
        if st.button("🗑  Clear Conversation", type="secondary"):
            st.session_state["ai_chat_messages"] = []
            st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 4 — LIBRARY (Record browser for current user)
# ═════════════════════════════════════════════════════════════════════════════

def page_library(cos: COSClient):
    _page_header("📚", "Library", "Your submission records and student history")

    username = st.session_state["user_profile"]["username"]

    # ── Upload panel ────────────────────────────────────────────────────────
    with st.expander("➕ Add Record to Library", expanded=False):
        with st.form("upload_history_form", clear_on_submit=True):
            col_a, col_b = st.columns(2)
            with col_a:
                form_student_id = st.text_input("Student ID *",
                                                placeholder="e.g. s001 or jane_doe")
                form_full_name  = st.text_input("Student Full Name",
                                                placeholder="Jane Doe")
                form_class_id   = st.text_input("Class ID",
                                                placeholder="e.g. CS101")
            with col_b:
                form_topic = st.text_input("Assignment Title *",
                                           placeholder="Binary Search Trees")
                form_file  = st.file_uploader("Text File *  (.txt · .md · .pdf)", type=["txt", "md", "pdf"])

            submitted = st.form_submit_button(
                "☁️  Save to Library",
                use_container_width=True,
                type="primary",
            )

        if submitted:
            errors: list[str] = []
            if not form_student_id.strip():
                errors.append("Student ID is required.")
            if not form_topic.strip():
                errors.append("Assignment title is required.")
            if form_file is None:
                errors.append("Upload a file.")
            if errors:
                for e in errors:
                    st.error(e)
            else:
                upload_text, parse_err = extract_text(form_file)
                upload_text = upload_text.strip()
                if parse_err:
                    st.error(f"Could not read file: {parse_err}")
                if upload_text:
                    with st.spinner("Saving…"):
                        ok, result = upload_history_to_ibm_cos(
                            cos=cos,
                            student_id=form_student_id,
                            topic=form_topic,
                            text=upload_text,
                            full_name=form_full_name,
                            class_id=form_class_id,
                        )
                    if ok:
                        st.success(f"Saved  →  `{result}`")
                    else:
                        st.error(f"Upload failed: {result}")

    st.divider()

    # ── Browser ─────────────────────────────────────────────────────────────
    col_class, _ = st.columns([2, 3])
    with col_class:
        class_id = st.text_input(
            "Filter by Class ID",
            value=st.session_state.get("selected_class", "DEMO101"),
            key="library_class",
        )
    st.session_state["selected_class"] = class_id

    students = list_students(cos, class_id)
    if not students:
        st.markdown(
            '<div class="card" style="text-align:center;padding:2rem;">'
            '<div style="font-size:2rem;margin-bottom:0.5rem">📭</div>'
            '<div style="font-size:0.95rem;color:var(--muted)">'
            f'No records found for class <strong>{class_id}</strong>.<br>'
            'Use the panel above to add your first record.</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    student_map = {s.get("full_name", s["student_id"]): s["student_id"] for s in students}
    selected_name = st.selectbox("Select Student", list(student_map.keys()),
                                  key="library_student")
    selected_id   = student_map[selected_name]

    roster = get_student_roster(cos, class_id, selected_id)
    if not roster or not roster.get("submissions"):
        st.warning("No submission records for this student.")
        return

    # ── Student summary card ─────────────────────────────────────────────────
    st.markdown(
        f'<div class="card">'
        f'  <div style="display:flex;justify-content:space-between;align-items:start">'
        f'    <div>'
        f'      <div style="font-size:1.05rem;font-weight:700">{roster.get("full_name", selected_id)}</div>'
        f'      <div style="font-size:0.8rem;color:var(--muted)">ID: {selected_id} &nbsp;·&nbsp; Class: {class_id}</div>'
        f'    </div>'
        f'    <span class="score-badge score-good">{len(roster["submissions"])} records</span>'
        f'  </div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    for sub in reversed(roster["submissions"]):
        submitted_date = sub.get("submitted_at", "")[:10] or "unknown date"
        with st.expander(
            f"📝 {sub.get('title', sub['assignment_id'])}  —  {submitted_date}",
            expanded=False,
        ):
            text = cos.get_text(
                f"student_history/{class_id}/{selected_id}/{sub['assignment_id']}.txt"
            )
            if text:
                st.text_area(
                    "Content",
                    value=text,
                    height=200,
                    disabled=True,
                    label_visibility="collapsed",
                )
                st.caption(
                    f"Path: `student_history/{class_id}/{selected_id}/{sub['assignment_id']}.txt` "
                    f"· {len(text):,} chars"
                )
            else:
                st.warning("File not found in storage.")


# ═════════════════════════════════════════════════════════════════════════════
# REPORT RENDERING HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _score_color(score: float, invert: bool = False) -> str:
    if invert:
        if score >= 70: return "#f87171"
        if score >= 40: return "#fbbf24"
        return "#34d399"
    else:
        if score >= 70: return "#34d399"
        if score >= 40: return "#fbbf24"
        return "#f87171"


def _report_summary_markdown(report: EvaluationReport) -> str:
    checklist_lines = ""
    for item in report.rubric_checklist:
        icon = {"Met": "✅", "Partially Met": "🟡", "Not Met": "❌"}.get(
            item.get("status", ""), "•"
        )
        checklist_lines += (
            f"\n  {icon} **{item.get('item', '')}** — "
            f"{item.get('status', '')} _{item.get('rationale', '')}_"
        )

    return (
        f"## 📋 Evaluation Report\n\n"
        f"| Metric | Score |\n"
        f"|---|---|\n"
        f"| 🏆 Originality Score | **{report.originality_score:.0f}%** |\n"
        f"| 🤖 AI Likelihood Score | **{report.ai_likelihood_score:.0f}%** |\n"
        f"| ✍️ Style Consistency | **{report.style_consistency.splitlines()[0]}** |\n\n"
        f"**Overall Verdict:** {report.overall_verdict}\n\n"
        f"### Rubric Checklist{checklist_lines}\n\n"
        f"### Detailed Analysis\n{report.detailed_analysis}\n\n"
        f"---\n_Model: `{report.model_used}`_"
    )


def _render_report(report: EvaluationReport):
    if report.error:
        return

    st.divider()
    st.markdown("#### 📊 Full Evaluation Report")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("🏆 Originality", f"{report.originality_score:.0f}%",
                  help="Higher = more original.")
        st.progress(int(report.originality_score) / 100)
    with c2:
        st.metric("🤖 AI Likelihood", f"{report.ai_likelihood_score:.0f}%",
                  help="Higher = more likely AI-generated.")
        st.progress(int(report.ai_likelihood_score) / 100)
    with c3:
        label = report.style_consistency.splitlines()[0] if report.style_consistency else "N/A"
        st.metric("✍️ Style", label)
        label_map = {
            "Consistent": 1.0, "Minor Deviation": 0.75,
            "Moderate Deviation": 0.45, "Significant Deviation": 0.15,
        }
        st.progress(label_map.get(label, 0.5))

    st.info(f"**Verdict:** {report.overall_verdict}")

    if report.rubric_checklist:
        st.markdown("**Rubric Alignment**")
        for item in report.rubric_checklist:
            status = item.get("status", "")
            icon   = {"Met": "✅", "Partially Met": "🟡", "Not Met": "❌"}.get(status, "•")
            st.markdown(
                f"{icon} **{item.get('item', '')}** — *{status}*  \n"
                f"  _{item.get('rationale', '')}_"
            )

    with st.expander("🔬 Full Detailed Analysis", expanded=False):
        st.markdown(report.detailed_analysis)

    if "\n" in report.style_consistency:
        lines = report.style_consistency.split("\n", 1)
        with st.expander("✍️ Style Consistency Explanation", expanded=False):
            st.markdown(lines[1].strip())


# ═════════════════════════════════════════════════════════════════════════════
# TOP NAVIGATION BAR (authenticated users)
# ═════════════════════════════════════════════════════════════════════════════

def _render_topnav():
    """Horizontal nav bar rendered at the top of every authenticated page."""
    profile  = st.session_state.get("user_profile", {})
    name     = profile.get("full_name", profile.get("username", "User"))
    initials = "".join(w[0] for w in name.split()[:2]).upper()
    active   = st.session_state.get("active_page", "Dashboard")

    pages = [
        ("📊", "Dashboard", "Dashboard"),
        ("🔍", "Evaluate",  "Evaluate"),
        ("🤖", "AI Chat",   "AIChat"),
        ("📚", "Library",   "Library"),
    ]

    # Brand + page buttons
    cols = st.columns([1.6] + [1] * len(pages) + [1.5, 0.8])

    with cols[0]:
        st.markdown(
            f'<div class="topnav-brand" style="display:flex;align-items:center;gap:10px;padding:2px 0">'  
            f'  <span style="font-size:1.8rem;display:flex;align-items:center;">{TOPNAV_LOGO}</span>'  
            f'  <span style="font-weight:700;color:var(--accent2);font-size:1.1rem">IntegriAI</span>'  
            f'</div>',
            unsafe_allow_html=True,
        )

    for i, (icon, label, key) in enumerate(pages):
        with cols[i + 1]:
            is_active = active == key
            btn_style = (
                "background:linear-gradient(135deg,rgba(108,99,255,0.25),rgba(167,139,250,0.18));"
                "color:#a78bfa;border-left:3px solid #6c63ff;"
                if is_active else ""
            )
            if st.button(
                f"{icon} {label}",
                key=f"topnav_{key}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                st.session_state["active_page"] = key
                st.rerun()

    with cols[-2]:
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:6px;justify-content:flex-end">'
            f'  <div style="width:30px;height:30px;border-radius:50%;'
            f'background:linear-gradient(135deg,#6c63ff,#a78bfa);'
            f'display:flex;align-items:center;justify-content:center;'
            f'font-weight:700;font-size:0.78rem;color:white">{initials}</div>'
            f'  <span style="font-size:0.82rem;color:var(--text);font-weight:500">{name.split()[0]}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with cols[-1]:
        if st.button("⏻ Sign Out", key="topnav_signout", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    st.divider()


# ═════════════════════════════════════════════════════════════════════════════
# LOGIN LANDING (unauthenticated)
# ═════════════════════════════════════════════════════════════════════════════

def page_login_landing(cos):
    # ── Asset diagnostics (visible, not just in server logs) ───────────────
    if LOGO_LOAD_ERROR:
        with st.expander("⚠️ Logo image failed to load — click for details", expanded=False):
            st.warning(LOGO_LOAD_ERROR)

    # ── Centred auth card (top) ───────────────────────────────────────────────
    _, center_col, _ = st.columns([1, 2, 1])
    with center_col:
        st.markdown(
            f'<div style="text-align:center;padding:1rem 0 0.5rem;">'
            f'  <div style="display:inline-block;width:108px;height:108px;margin-bottom:0.85rem;">'
            f'    <style>.login-logo-img img{{width:108px!important;height:108px!important;object-fit:contain!important;border-radius:18px!important;box-shadow:0 6px 20px rgba(0,0,0,0.3);}}</style>'
            f'    <div class="login-logo-img">{LOGIN_LOGO}</div>'
            f'  </div>'
            f'  <div style="font-size:1.5rem;font-weight:800;background:linear-gradient(135deg,#fff,var(--accent2));'
            f'-webkit-background-clip:text;-webkit-text-fill-color:transparent;">IntegriAI</div>'
            f'  <div style="font-size:0.85rem;color:var(--muted);margin-top:0.25rem;">Academic Integrity Intelligence Platform</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="card-glass" style="padding:2rem;margin-top:1rem;animation:fadeIn 0.6s ease;">'
            '  <div style="text-align:center;margin-bottom:1.2rem;">'
            '    <div style="font-size:1.15rem;font-weight:700;color:var(--text);">Welcome back</div>'
            '    <div style="font-size:0.8rem;color:var(--muted);margin-top:2px;">Sign in to your account or create a new one</div>'
            '  </div>',
            unsafe_allow_html=True,
        )

        if st.session_state.get("show_forgot_pw"):
            # ── Forgot password inline ────────────────────────────────────────
            st.markdown("**Reset Password**")
            st.caption("Enter the email address linked to your account.")
            fp_email = st.text_input("Email", key="fp_email_main", placeholder="alice@university.edu")
            fp_pass1 = st.text_input("New Password", type="password", key="fp_pass1_main", placeholder="New password")
            fp_pass2 = st.text_input("Confirm New Password", type="password", key="fp_pass2_main", placeholder="Repeat password")
            col_reset, col_cancel = st.columns(2)
            with col_reset:
                if st.button("Reset Password", use_container_width=True, type="primary", key="main_reset_btn"):
                    if not fp_email.strip():
                        st.error("Email is required.")
                    elif fp_pass1 != fp_pass2:
                        st.error("Passwords do not match.")
                    elif len(fp_pass1) < 6:
                        st.error("Min 6 characters.")
                    else:
                        with st.spinner("Resetting…"):
                            ok, result = reset_password_by_email(cos, fp_email, fp_pass1)
                        if ok:
                            st.success(f"Password reset for **{result}**. Sign in below.")
                            st.session_state["show_forgot_pw"] = False
                            st.rerun()
                        else:
                            st.error(result)
            with col_cancel:
                if st.button("Cancel", use_container_width=True, key="main_cancel_btn"):
                    st.session_state["show_forgot_pw"] = False
                    st.rerun()
        else:
            tab_login, tab_register = st.tabs(["🔑  Sign In", "✨  Create Account"])

            # ── Sign In tab ───────────────────────────────────────────────────
            with tab_login:
                # Show persisted login error (survives rerun)
                if st.session_state.get("_login_error"):
                    st.error(st.session_state.pop("_login_error"))

                username = st.text_input("Username", key="main_login_user",
                                         placeholder="e.g. prof_smith")
                password = st.text_input("Password", type="password",
                                         key="main_login_pass",
                                         placeholder="••••••••")
                col_signin, col_forgot = st.columns([3, 2])
                with col_signin:
                    do_login = st.button("Sign In", use_container_width=True,
                                         type="primary", key="main_sign_in_btn")
                with col_forgot:
                    if st.button("Forgot password?", use_container_width=True,
                                 key="main_forgot_btn"):
                        st.session_state.pop("_login_error", None)
                        st.session_state["show_forgot_pw"] = True
                        st.rerun()

                if do_login:
                    if not username or not password:
                        st.session_state["_login_error"] = "Both fields are required."
                        st.rerun()
                    else:
                        with st.spinner("Verifying…"):
                            ok, profile = authenticate_user(cos, username, password)
                        if ok:
                            st.session_state.pop("_login_error", None)
                            st.session_state["authenticated"] = True
                            st.session_state["user_profile"]  = profile
                            st.session_state["active_page"]   = "Dashboard"
                            st.rerun()
                        else:
                            st.session_state["_login_error"] = "❌ Invalid username or password. Please try again."
                            st.rerun()

            # ── Create Account tab ────────────────────────────────────────────
            with tab_register:
                # Show persisted register error
                if st.session_state.get("_register_error"):
                    st.error(st.session_state.pop("_register_error"))
                if st.session_state.get("_register_success"):
                    st.success(st.session_state.pop("_register_success"))

                r_full  = st.text_input("Full Name",  key="main_reg_name",
                                        placeholder="Dr. Alice Smith")
                r_email = st.text_input("Email",      key="main_reg_email",
                                        placeholder="alice@university.edu")
                r_user  = st.text_input("Username",   key="main_reg_user",
                                        placeholder="prof_alice")
                r_pass  = st.text_input("Password",   type="password",
                                        key="main_reg_pass1",
                                        placeholder="At least 6 characters")
                r_pass2 = st.text_input("Confirm Password", type="password",
                                        key="main_reg_pass2",
                                        placeholder="Repeat password")
                if st.button("Create Account", use_container_width=True,
                             type="primary", key="main_create_btn"):
                    if not r_full.strip():
                        st.session_state["_register_error"] = "Full Name is required."
                        st.rerun()
                    elif not r_email.strip() or "@" not in r_email:
                        st.session_state["_register_error"] = "A valid email address is required."
                        st.rerun()
                    elif not r_user.strip():
                        st.session_state["_register_error"] = "Username is required."
                        st.rerun()
                    elif len("".join(c for c in r_user if c.isalnum() or c in ("_", "-"))) == 0:
                        st.session_state["_register_error"] = "Username may only contain letters, numbers, _ or -."
                        st.rerun()
                    elif r_pass != r_pass2:
                        st.session_state["_register_error"] = "Passwords do not match."
                        st.rerun()
                    elif len(r_pass) < 6:
                        st.session_state["_register_error"] = "Password must be at least 6 characters."
                        st.rerun()
                    else:
                        with st.spinner("Creating account…"):
                            ok, err = register_user(cos, r_user, r_pass,
                                                    r_full, r_email, "instructor")
                        if ok:
                            st.session_state["_register_success"] = "✅ Account created! Switch to Sign In to log in."
                            st.rerun()
                        else:
                            st.session_state["_register_error"] = err or "Registration failed."
                            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Hero / feature section (below auth card) ──────────────────────────────
    st.markdown(
        f'<div class="feature-pills" style="justify-content:center;margin-bottom:1.5rem;">'
        f'  <span class="feature-pill">AI Detection</span>'
        f'  <span class="feature-pill">Style Analysis</span>'
        f'  <span class="feature-pill">Rubric Alignment</span>'
        f'  <span class="feature-pill">Cloud Storage</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Feature cards ─────────────────────────────────────────────────────────
    col_a, col_b, col_c = st.columns(3)
    cards = [
        ("🧠", "Layer 1", "AI & Paraphrase Detection",
         "Detects AI-generated text using perplexity, burstiness, and vocabulary entropy signals."),
        ("✍️", "Layer 2", "Stylistic Deviation",
         "Compares submission against the student's historical writing signature via RAG."),
        ("📋", "Layer 3", "Rubric Alignment",
         "Verifies submission against your configured grading criteria and checklist."),
    ]
    for col, (icon, tag, title, desc) in zip([col_a, col_b, col_c], cards):
        with col:
            st.markdown(
                f'<div class="card" style="text-align:center;animation:fadeIn 0.5s ease">'
                f'  <div style="font-size:1.8rem;margin-bottom:0.4rem">{icon}</div>'
                f'  <div style="font-size:0.7rem;color:var(--accent2);text-transform:uppercase;'
                f'letter-spacing:0.8px;font-weight:600;margin-bottom:0.3rem">{tag}</div>'
                f'  <div style="font-size:0.95rem;font-weight:700;margin-bottom:0.4rem">{title}</div>'
                f'  <div style="font-size:0.8rem;color:var(--muted);line-height:1.5">{desc}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ═════════════════════════════════════════════════════════════════════════════
# SETUP GUIDE (missing credentials)
# ═════════════════════════════════════════════════════════════════════════════

def page_setup_guide(missing: list[str]):
    _page_header("⚠️", "Setup Required", "Cloud credentials are not configured")
    st.error(
        f"Missing secrets: `{'`, `'.join(missing)}`"
    )
    with st.expander("How to configure credentials", expanded=True):
        st.markdown("""
**Option A — `.streamlit/secrets.toml`** *(recommended)*
```toml
WATSONX_APIKEY      = "your-watsonx-api-key"
WATSONX_URL         = "https://eu-de.ml.cloud.ibm.com"
PROJECT_ID          = "your-watsonx-project-id"
COS_ENDPOINT        = "https://s3.eu-de.cloud-object-storage.appdomain.cloud"
COS_API_KEY_ID      = "your-cos-api-key"
COS_INSTANCE_CRN    = "crn:v1:bluemix:public:cloud-object-storage:global:..."
COS_BUCKET_NAME     = "plagiarism-intelligence"
```

**Option B — Environment Variables**
```bash
export WATSONX_APIKEY="..."
export PROJECT_ID="..."
export COS_ENDPOINT="..."
export COS_API_KEY_ID="..."
export COS_INSTANCE_CRN="..."
```
        """)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(
        page_title="IntegriAI — Academic Integrity",
        page_icon=LOGO_IMAGE if LOGO_IMAGE is not None else "🛡️",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # Inject global CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    _init_session()

    creds = _load_credentials()
    valid, missing = _credentials_valid(creds)

    if not valid:
        page_setup_guide(missing)
        st.stop()

    try:
        cos = get_cos(
            endpoint=creds["cos_endpoint"],
            api_key=creds["cos_api_key"],
            crn=creds["cos_crn"],
            bucket=creds["cos_bucket"],
        )
    except Exception as exc:
        st.error(f"Cannot connect to cloud storage: {exc}")
        st.stop()

    if not st.session_state["authenticated"]:
        # No sidebar on the pre-login screen — page_login_landing() is the
        # single, full-width login/register surface. (Sidebar is already
        # hidden globally via CUSTOM_CSS above.)
        page_login_landing(cos)
        return

    # ── Authenticated: no sidebar — top nav is the only navigation ─────────
    # (Sidebar is already hidden globally via CUSTOM_CSS above.)
    _render_topnav()

    page = st.session_state["active_page"]

    if page == "Dashboard":
        page_dashboard(cos)
    elif page == "Evaluate":
        page_evaluate(cos, creds)
    elif page == "AIChat":
        page_ai_chat(creds)
    elif page == "Library":
        page_library(cos)


if __name__ == "__main__":
    main()
