# AI-Driven Plagiarism Intelligence for Assignments

> A production-ready Streamlit application powered by **IBM watsonx.ai (Granite)** and
> **IBM Cloud Object Storage** that performs contextual, adaptive plagiarism and
> AI-text detection on student assignment submissions.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Streamlit Frontend                        │
│  ┌──────────────┐  ┌───────────────────┐  ┌──────────────────┐  │
│  │  Dashboard   │  │  Evaluate (Chat)  │  │     History      │  │
│  │  (Rubric     │  │  • RAG selector   │  │  • Browse past   │  │
│  │   Config)    │  │  • File upload    │  │    submissions   │  │
│  └──────────────┘  │  • Chat UI        │  └──────────────────┘  │
│                    │  • Validation     │                         │
│                    └───────────────────┘                         │
└────────────────────────────┬────────────────────────────────────┘
                             │
           ┌─────────────────┼────────────────────┐
           ▼                 ▼                    ▼
   ┌───────────────┐  ┌─────────────┐   ┌────────────────────┐
   │  IBM watsonx  │  │  IBM Cloud  │   │  IBM Cloud Object  │
   │  (Granite 3)  │  │  IAM Auth   │   │  Storage (COS)     │
   │               │  └─────────────┘   │  instructor_profiles│
   │  3-Layer      │                    │  instructor_configs │
   │  Evaluation   │                    │  student_history    │
   └───────────────┘                    └────────────────────┘
```

### Three-Layer Evaluation Agent

| Layer | Focus | Signals Analysed |
|-------|-------|-----------------|
| **Layer 1** | AI & Paraphrase Detection | Perplexity, burstiness, vocabulary entropy, transition patterns |
| **Layer 2** | Stylistic Deviation | Vocabulary, sentence structure, tone vs. student's historical baseline |
| **Layer 3** | Rubric Alignment | Submission vs. instructor's configured grading pattern & checklist |

---

## Project Structure

```
.
├── app.py                          # Main Streamlit application
├── requirements.txt
├── .streamlit/
│   ├── secrets.toml                # Credentials (gitignored)
│   └── config.toml                 # UI theme
└── utils/
    ├── __init__.py
    ├── cos_client.py               # IBM COS wrapper with retry logic
    ├── auth.py                     # COS-backed authentication
    ├── instructor_config.py        # Grading config + prompt builder
    ├── student_history.py          # Historical writing samples + RAG
    ├── document_parser.py          # .txt / .md / .pdf extractor
    └── ai_agent.py                 # Granite evaluation engine
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Credentials

Edit `.streamlit/secrets.toml`:

```toml
WATSONX_APIKEY      = "your-watsonx-api-key"
WATSONX_URL         = "https://us-south.ml.cloud.ibm.com"
PROJECT_ID          = "your-watsonx-project-id"

COS_ENDPOINT        = "https://s3.us-south.cloud-object-storage.appdomain.cloud"
COS_API_KEY_ID      = "your-cos-api-key"
COS_INSTANCE_CRN    = "crn:v1:bluemix:public:cloud-object-storage:global:a/..."
COS_BUCKET_NAME     = "plagiarism-intelligence"
```

> **Note:** Never commit `secrets.toml` to version control.
> Add it to `.gitignore`.

### 3. IBM Cloud Setup

#### watsonx.ai
1. Create a [Watson Studio project](https://cloud.ibm.com/catalog/services/watson-studio)
2. Add the **watsonx.ai Runtime** service
3. Copy the **Project ID** from *Manage → General*
4. Generate an **IBM Cloud API Key** from *Manage → Access (IAM)*

#### Cloud Object Storage (Lite Tier — Free)
1. Create a [Cloud Object Storage](https://cloud.ibm.com/catalog/services/cloud-object-storage) instance
2. Create a bucket (e.g. `plagiarism-intelligence`) in `us-south`
3. Under *Service Credentials*, create credentials with `Writer` role (HMAC optional)
4. Copy `apikey` → `COS_API_KEY_ID` and `resource_instance_id` → `COS_INSTANCE_CRN`
5. Copy the bucket's **public endpoint** → `COS_ENDPOINT`

### 4. Run the Application

```bash
streamlit run app.py
```

---

## Feature Walkthrough

### Authentication
- Register with full name, email, and password
- Passwords are stored as **SHA-256 hashes** — never plaintext
- Profiles persisted to `instructor_profiles/<username>.json` in COS

### Dashboard Configuration
- Select your **primary grading style** (Concept-focused, Definition-heavy, etc.)
- Add **custom rubric tags** (e.g. "cite sources", "include diagrams")
- Set **score weighting** across originality, AI detection, and style match
- Preview the exact **system-prompt constraint** that will be injected into Granite

### Evaluate (Chat Workspace)
1. Select a **student history profile** (demo student pre-seeded on first run)
2. Upload an assignment file (`.txt`, `.md`, or `.pdf`)
3. Click **Validate Contextually** to trigger IBM Granite evaluation
4. View the structured report: scores, rubric checklist, detailed analysis
5. Ask **follow-up questions** about the report in natural language via chat

### History Browser
- Browse all past submissions for any student in any class
- Supports multiple class IDs (e.g. `CS101`, `DEMO101`)

---

## COS Data Layout

```
plagiarism-intelligence/          ← bucket root
├── instructor_profiles/
│   └── prof_smith.json
├── instructor_configs/
│   └── prof_smith.json
└── student_history/
    └── DEMO101/
        └── demo_student_001/
            ├── roster.json
            ├── hw1.txt
            └── hw2.txt
```

---

## Evaluation Report Schema

The IBM Granite model returns (and the app renders):

```json
{
  "originality_score": 85,
  "ai_likelihood_score": 12,
  "style_consistency": {
    "label": "Consistent",
    "explanation": "Vocabulary and sentence patterns match historical baseline."
  },
  "rubric_checklist": [
    { "item": "cite sources", "status": "Met", "rationale": "3 citations present." },
    { "item": "include diagrams", "status": "Not Met", "rationale": "No diagrams found." }
  ],
  "overall_verdict": "High-confidence human-authored work with minor rubric gaps.",
  "detailed_analysis": "..."
}
```

---

## Error Handling

| Scenario | Handling |
|----------|---------|
| Missing credentials | Setup guide page shown before any other content |
| COS connection failure | Caught at startup; descriptive error displayed |
| Empty file upload | Validated before enabling the Evaluate button |
| Model API failure | Retried 3× with exponential back-off; fallback to next model |
| JSON parse failure | Raw model response preserved in `detailed_analysis` field |
| Scanned PDF | Warning message; text extraction returns error string |

---

## Environment Variables (Alternative to secrets.toml)

```bash
export WATSONX_APIKEY="..."
export WATSONX_URL="https://us-south.ml.cloud.ibm.com"
export PROJECT_ID="..."
export COS_ENDPOINT="..."
export COS_API_KEY_ID="..."
export COS_INSTANCE_CRN="..."
export COS_BUCKET_NAME="plagiarism-intelligence"
```

---

## License

MIT — see [LICENSE](LICENSE) for details.
