# Multi-Agent Customer Support — LangGraph + Streamlit + Fine-tuned Qwen LoRA

This is a beginner-friendly starter project that loads an existing **Qwen2.5-1.5B-Instruct LoRA adapter** and wraps it in a LangGraph support workflow.

## What is included

- **Coordinator:** routes requests to knowledge, ticket creation/tracking, account lookup, or human escalation.
- **Knowledge Agent (RAG):** searches local `.md`, `.txt`, and `.pdf` files with TF-IDF retrieval and passes retrieved passages to Qwen.
- **Ticket Agent:** creates tickets and checks status in SQLite. Ticket access is scoped to the signed-in customer.
- **User Agent:** displays demo account information only after the customer ID/PIN check succeeds.
- **Audio Processor:** optional speech-to-text via `faster-whisper`; the transcript follows the same workflow as typed text.
- **Human escalation:** stores a handoff summary and recent conversation in SQLite. No real notification is sent in this demo.
- **Streamlit UI:** chat, sign-in, audio upload, ticket list, and workflow details.

## Important note about the Coordinator

Routing uses explicit rules in `src/graph.py` for predictable behavior. The fine-tuned Qwen adapter is used to write customer-facing answers using retrieved knowledge and verified tool results. This is intentional: a general customer-support adapter trained on conversation data is not automatically a reliable structured intent classifier. Later, you can fine-tune or evaluate a separate intent classifier if needed.

## 1. Project structure

```text
customer_support_multi_agent/
├── app.py
├── requirements.txt
├── .env.example
├── README.md
├── data/
│   ├── knowledge/
│   │   ├── company_faq.md
│   │   └── company_policies.md
│   └── support.db                 # created automatically at first run
└── src/
    ├── __init__.py
    ├── config.py
    ├── model.py                   # loads base model + your LoRA adapter
    ├── database.py                # demo users, tickets, escalations
    ├── knowledge.py               # local document retrieval (RAG)
    ├── audio.py                   # optional audio transcription
    └── graph.py                   # LangGraph coordinator and agents
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
py -m venv .venv
.\.venv\Scripts\Activate.ps1
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

1. Ask: `How do I reset my password?` → Knowledge Agent searches the FAQ/policy files.
2. Sign in, then ask: `I cannot log in and get an error message` → Ticket Agent creates a ticket.
3. Ask: `Show my ticket status` → Ticket Agent lists your tickets.
4. Ask: `Track ticket #1 status` → Ticket Agent checks that ticket only if it belongs to your signed-in account.
5. Ask: `What email is on my account?` → User Agent returns profile information only after sign-in.
6. Ask: `I want to talk to a human` → Human escalation record is created with context.
7. Upload an audio file and click **Process uploaded audio** → Audio Processor transcribes it, then the Coordinator routes the transcript.

## 7. Optional audio installation

Audio is optional. To enable transcription:

```powershell
pip install faster-whisper
```

The first transcription may download the Whisper model. CPU transcription can be slow. If `faster-whisper` is not installed, text chat continues to work.

## 8. Add your company PDFs

Copy company `.pdf`, `.txt`, or `.md` files into `data/knowledge/`. Restart the app after adding documents because the in-process TF-IDF index is cached. The included RAG implementation is intentionally small and local; for larger document collections, replace it with a vector database and embedding model.

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
