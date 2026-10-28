# Multi-Agent Customer Support — LangGraph + Streamlit + Fine-tuned Qwen LoRA

This is a beginner-friendly starter project that loads an existing **Qwen2.5-1.5B-Instruct LoRA adapter** and wraps it in a LangGraph support workflow.

## What is included

- **Coordinator:** routes requests to knowledge, ticket creation/tracking, account lookup, or human escalation using deterministic intent rules in `src/routing.py`.
- **Knowledge Agent (ChromaDB Vector RAG):** indexes local `.md`, `.txt`, and `.pdf` files into a persistent **ChromaDB** vector store (`data/chroma_db/`) using `BAAI/bge-small-en-v1.5` embeddings and cosine similarity (with lexical verification and automatic TF-IDF fallback), passing retrieved passages to Qwen.
- **Ticket Agent:** creates tickets (with duplicate prevention) and checks status in SQLite (`data/support.db`). Ticket access is strictly scoped to the signed-in customer.
- **User Agent:** displays verified account information only after the customer ID/PIN check succeeds.
- **Audio Processor (Dual Voice Input):** speech-to-text via cached `faster-whisper` (`int8` quantized `base` model with VAD filtering). Supports **both**:
  1. **🔴 Direct Browser Microphone Recording:** tap-to-record directly in the browser (`st.audio_input`) with manual send or automatic send upon finishing recording.
  2. **📁 Audio File Upload:** upload existing audio files (`WAV`, `MP3`, `M4A`, `OGG`, `FLAC`, `WEBM`) or test immediately with the 3 built-in downloadable sample `.wav` recordings.
- **Human Escalation & Admin Portal:** generates structured, issue-specific handoff briefings (`src/handoff.py`) with category, urgency, extracted entities (orders, tickets, dollar amounts, error codes), and recommended actions, stored in SQLite and manageable from the sidebar Admin Dashboard.
- **Streamlit UI:** live chat, customer sign-in, dual-tabbed voice & audio input, ticket list, and interactive admin escalation portal.

## Important note about the Coordinator

Routing uses explicit rules in `src/routing.py` (orchestrated by `src/graph.py`) for predictable behavior. The fine-tuned Qwen adapter is used to write customer-facing answers using retrieved knowledge and verified tool results. This is intentional: a general customer-support adapter trained on conversation data is not automatically a reliable structured intent classifier. Later, you can fine-tune or evaluate a separate intent classifier if needed.

## 1. Project structure

```text
customer-support-AI-agent/
├── app.py                         # Streamlit live chat, dual voice input & admin portal
├── requirements.txt
├── .env.example
├── README.md
├── qwen25_support_adapter/        # Fine-tuned Qwen 2.5-1.5B LoRA adapter & tokenizer
├── data/
│   ├── chroma_db/                 # Persistent ChromaDB vector store (auto-created/synced)
│   ├── knowledge/
│   │   ├── company_faq.md
│   │   └── company_policies.md
│   ├── sample_password_reset.wav  # Sample 1: Password FAQ audio
│   ├── sample_human_agent.wav     # Sample 2: Live Agent escalation audio
│   ├── sample_report_issue.wav    # Sample 3: Bug report / ticket audio
│   └── support.db                 # SQLite database (created automatically at first run)
├── src/
│   ├── __init__.py
│   ├── config.py                  # Environment & model/database paths
│   ├── model.py                   # Loads base Qwen model + PEFT/LoRA adapter
│   ├── database.py                # SQLite users, tickets, escalations, interactions
│   ├── knowledge.py               # ChromaDB semantic search + TF-IDF fallback (RAG)
│   ├── audio.py                   # Cached faster-whisper speech-to-text transcription
│   ├── routing.py                 # Rule-based intent routing & safety guardrails
│   ├── handoff.py                 # Structured human handoff briefing generation
│   └── graph.py                   # LangGraph workflow, coordinator, and agent nodes
└── tests/
    └── test_agents.py             # Automated unit & end-to-end test suite (31 tests)
```

## 2. Before running: locate your adapter

The adapter is a folder containing at least `adapter_config.json` and adapter weights (usually `adapter_model.safetensors`). It is not the full base model.

For your Colab workflow, the folder may be:

```text
/content/drive/MyDrive/customer_support_qwen25_15b_lora/
```

If running on your Windows computer, download/copy that entire adapter folder and use its Windows path in `.env`. Do not point `SUPPORT_ADAPTER_DIR` at the ZIP file or at the parent folder if the adapter files are one level deeper.

## 3. Set up in VS Code (Windows PowerShell)

Open a terminal in this project directory, then run:

```powershell
py -m venv myenv
.\myenv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

If PowerShell blocks activation, use Command Prompt and run:

```bat
.venv\Scripts\activate.bat
```

Edit `.env` and set `SUPPORT_ADAPTER_DIR` to your actual adapter folder. If you are only testing the UI before copying the adapter, set `SUPPORT_DISABLE_MODEL=1`. The application will use safe fallback responses for routes that do not require generated language. RAG answers will show retrieved passages rather than a fluent Qwen answer.

## 4. Run the app

```powershell
streamlit run app.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`.

## 5. Demo sign-in

- Customer ID: `CUST-1001`
- Demo PIN: `2468`

Second demo account:
- Customer ID: `CUST-1002`
- Demo PIN: `1357`

These are demonstration credentials, not secure production authentication. The database stores demo PINs in plain text so the starter is easy to inspect. Replace this with a proper identity provider and hashed credentials before real use.

## 6. Test the agents

1. Ask: `How do I reset my password?` → Knowledge Agent retrieves the answer from ChromaDB (`data/chroma_db/`).
2. Sign in, then ask: `I cannot log in and get an error message` → Ticket Agent creates a ticket in `data/support.db`.
3. Ask: `Show my ticket status` → Ticket Agent lists your tickets.
4. Ask: `Track ticket #1 status` → Ticket Agent checks that ticket only if it belongs to your signed-in account.
5. Ask: `What email is on my account?` → User Agent returns profile information only after sign-in.
6. Ask: `I want to talk to a human` → Escalation Agent creates a structured handoff record visible in the sidebar **Admin Portal -> Open Admin Dashboard**.
7. Expand **🎙️ Voice & Audio Input** and test either tab:
   - **🔴 Record with Browser Microphone:** Click the red microphone button to record directly in your browser, then click **🚀 Send Voice Message** (or enable **Auto-send when recording finishes**).
   - **📁 Upload Audio File:** Upload any `WAV`, `MP3`, `M4A`, `OGG`, `FLAC`, or `WEBM` file (or download one of the 3 built-in sample `.wav` files) and click **🚀 Process Uploaded Audio**.

To run the automated unit and integration test suite:

```powershell
python -m unittest tests/test_agents.py -v
```

## 7. Voice & Audio Transcription (`faster-whisper`)

Audio input is powered by `faster-whisper` (included in `requirements.txt`):

```powershell
pip install faster-whisper
```

`src/audio.py` caches the `int8` Whisper model in memory (`@lru_cache`) with Voice Activity Detection (`vad_filter=True`) for fast (~1 second) transcription. Both recorded microphone audio and uploaded audio files are preserved with an inline audio player inside the chat history alongside their transcript.

## 8. Knowledge Base & ChromaDB Vector Search

Copy company `.pdf`, `.txt`, or `.md` files into `data/knowledge/`.
- `src/knowledge.py` splits documents into chunks using `RecursiveCharacterTextSplitter` and stores `BAAI/bge-small-en-v1.5` vector embeddings in `data/chroma_db/` (`hnsw:space = cosine`).
- Document changes are detected automatically via MD5 content hashing—whenever you add or edit files in `data/knowledge/`, ChromaDB automatically re-syncs the collection on the next query.
- If ChromaDB is unavailable, retrieval automatically falls back to `scikit-learn` TF-IDF cosine similarity.

## 9. How the workflow works

```text
Customer text/audio
        |
        v
Audio Processor (only for audio)
        |
        v
Coordinator (rule-based routing)
  |        |        |         |
  v        v        v         v
Knowledge Ticket   User    Escalation
  Agent   Agent    Agent      Agent
  |        |        |         |
  +--------+--------+---------+
                   |
                   v
        Response to customer
```

## 10. Model loading and GPU notes

- The app loads `Qwen/Qwen2.5-1.5B-Instruct` and then attaches the LoRA adapter with PEFT.
- CUDA is used with 4-bit quantization when available; CPU mode is provided for testing but may be slow and memory-intensive.
- On a local Windows PC without an NVIDIA CUDA setup, consider testing routing/RAG with `SUPPORT_DISABLE_MODEL=1` first, then run inference in Colab or a GPU host.
- Keep Transformers, PEFT, Accelerate, PyTorch, and bitsandbytes versions compatible. If model loading fails, read the original exception and install a compatible set rather than repeatedly reinstalling unrelated packages.

## 11. What must change before production

- Replace demo PIN authentication with real secure authentication and authorization.
- Add rate limiting, secrets management, audit logs, retention rules, and privacy protections.
- Connect escalation records to a real helpdesk or notification channel; currently it only stores records in SQLite.
- Connect ticket creation and tracking to the company's real ticket system if one exists.
- Add tests, monitoring, retries, and human review for high-risk requests.
- Evaluate the adapter on your actual support FAQs and test that it does not invent policies or private data.

## Troubleshooting

**Adapter not found:** check that the directory directly contains `adapter_config.json` and adapter weights.

**Qwen does not load:** the model is downloaded from Hugging Face on first use; check internet access and package compatibility. On a GPU host, verify `torch.cuda.is_available()`.

**RAG says it cannot find an answer:** add the relevant FAQ/policy to `data/knowledge/`, then restart Streamlit.

**Audio does not work:** install `faster-whisper`; confirm the audio file is supported and contains speech.

**Ticket lookup fails:** sign in first and ensure the ticket belongs to the current customer. This is an intentional privacy check.
