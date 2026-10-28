import os
import hashlib
import importlib
import streamlit as st
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".env", override=True)
import src.config as cfg
importlib.reload(cfg)
import src.model
importlib.reload(src.model)
import src.knowledge
import src.routing
import src.handoff
import src.graph
importlib.reload(src.knowledge)
importlib.reload(src.routing)
importlib.reload(src.handoff)
importlib.reload(src.graph)
from src.database import init_db, authenticate_customer, list_tickets, list_escalations, update_escalation_status, list_interactions
from src.graph import handle_support_request
from src.audio import transcribe_audio

st.set_page_config(page_title="Multi-Agent Customer Support", page_icon="💬", layout="wide")
init_db()

# Initialize session state for dashboard visibility
if "show_admin_dashboard" not in st.session_state:
    st.session_state.show_admin_dashboard = False

# --- SIDEBAR: CONTROLS & DASHBOARD BUTTON ---
with st.sidebar:
    st.markdown("### 🛡️ Admin Portal")

    escalations = list_escalations()
    pending_count = sum(1 for e in escalations if e["status"] == "pending_human")
    pending_badge = f" ({pending_count} Pending)" if pending_count > 0 else ""

    # Dashboard Button
    if not st.session_state.show_admin_dashboard:
        if st.button(f"📊 Open Admin Dashboard{pending_badge}", use_container_width=True, type="primary"):
            st.session_state.show_admin_dashboard = True
            st.rerun()
    else:
        if st.button("✖️ Close Admin Dashboard", use_container_width=True):
            st.session_state.show_admin_dashboard = False
            st.rerun()

    # Content revealed ONLY after clicking the Dashboard button
    if st.session_state.show_admin_dashboard:
        st.divider()
        st.subheader("🛡️ Conversation Handoffs")
        st.caption("Inspect AI agent escalations and conversation history.")

        if not escalations:
            st.info("No conversation handoffs recorded yet.")
        else:
            # 1. Conversation History Table: cus id, route, when created
            table_rows = []
            for e in escalations:
                table_rows.append({
                    "ID": f"#{e['escalation_id']}",
                    "cus id": e["customer_id"] or "Guest",
                    "route": "escalation",
                    "when created": e["created_at"][:19].replace("T", " "),
                })
            df_escalations = pd.DataFrame(table_rows)

            st.markdown("##### 📋 Conversation History Table")
            st.dataframe(
                df_escalations,
                use_container_width=True,
                hide_index=True
            )

            # 2. Select conversation from dropdown
            st.markdown("##### 🔍 Select Conversation")
            options = {
                f"Esc #{e['escalation_id']} · {e['customer_id'] or 'Guest'} · {e['created_at'][:16].replace('T', ' ')}": e
                for e in escalations
            }
            selected_label = st.selectbox(
                "Choose conversation:",
                list(options.keys()),
                help="Select a conversation to view its handoff summary."
            )
            selected_esc = options[selected_label]

            # 3. Based on the selected conversation, show concise handoff summary
            st.markdown("##### 📝 Handoff Summary")
            status_color = "red" if selected_esc["status"] == "pending_human" else "green"
            st.markdown(
                f"**Status:** :{status_color}[**{selected_esc['status'].upper()}**] · "
                f"**Customer:** `{selected_esc['customer_id'] or 'Guest'}` · "
                f"**Route:** `escalation`"
            )
            raw_summary = selected_esc["summary"] or ""
            if "Conversation context:" in raw_summary:
                raw_summary = raw_summary.split("Conversation context:")[0].strip()
            st.info(raw_summary)

            # 4. Expandable full conversation context
            with st.expander("💬 View Full Conversation Context", expanded=False):
                st.text_area(
                    "Chat transcript leading up to escalation:",
                    value=selected_esc["conversation"],
                    height=200,
                    key=f"side_conv_{selected_esc['escalation_id']}",
                    disabled=True
                )

            # 5. Status action buttons
            btn_col1, btn_col2 = st.columns(2)
            if selected_esc["status"] == "pending_human":
                if btn_col1.button("✅ Resolve", key=f"side_res_{selected_esc['escalation_id']}", use_container_width=True):
                    update_escalation_status(selected_esc["escalation_id"], "resolved")
                    st.success("Marked resolved!")
                    st.rerun()
                if btn_col2.button("🔄 In Progress", key=f"side_inp_{selected_esc['escalation_id']}", use_container_width=True):
                    update_escalation_status(selected_esc["escalation_id"], "in_progress")
                    st.info("Marked in progress!")
                    st.rerun()
            else:
                if btn_col1.button("Re-open", key=f"side_reopen_{selected_esc['escalation_id']}", use_container_width=True):
                    update_escalation_status(selected_esc["escalation_id"], "pending_human")
                    st.rerun()

        st.divider()

    # --- CUSTOMER SIGN-IN (DEMO) ---
    st.markdown("### 👤 Customer Sign-In (Demo)")
    customer_id_input = st.text_input("Customer ID", value="CUST-1001", key="side_cust_id")
    pin_input = st.text_input("Verification PIN", value="2468", type="password", key="side_pin")
    if st.button("Sign in", use_container_width=True, key="side_btn_signin"):
        user = authenticate_customer(customer_id_input, pin_input)
        if user:
            st.session_state["verified_user"] = user
            st.success(f"Signed in as {user['display_name']}.")
            st.rerun()
        else:
            st.session_state.pop("verified_user", None)
            st.error("Customer ID or PIN is incorrect.")

    if st.session_state.get("verified_user"):
        user = st.session_state["verified_user"]
        st.success(f"Signed in: {user['display_name']}")
        if st.button("Sign out", use_container_width=True, key="side_btn_signout"):
            st.session_state.pop("verified_user", None)
            st.rerun()
        st.subheader("Your tickets")
        for ticket in list_tickets(user["customer_id"]):
            st.write(f"**#{ticket['ticket_id']}** · {ticket['subject']} · {ticket['status']}")

    st.divider()
    # Model Configuration: Always run the fine-tuned model
    os.environ["SUPPORT_DISABLE_MODEL"] = "0"
    cfg.DISABLE_MODEL = False

    st.markdown("### 🤖 Active Model")
    if cfg.USE_ADAPTER:
        st.success("✨ **Fine-Tuned Qwen 2.5 (LoRA)**")
        with st.expander("⚙️ Model Details", expanded=False):
            st.caption(f"**Base Model:** `{cfg.BASE_MODEL}`")
            st.caption(f"**Adapter:** `{cfg.ADAPTER_DIR}`")
            st.caption("**Inference:** Neural Text Generation Active")
    else:
        st.info("ℹ️ **Qwen 2.5 Base Model**")
        with st.expander("⚙️ Model Details", expanded=False):
            st.caption(f"**Base Model:** `{cfg.BASE_MODEL}`")


# --- MAIN PAGE: CUSTOMER LIVE CHAT ---
st.title("💬 Multi-Agent Customer Support")
st.caption("LangGraph • Fine-tuned Qwen LoRA • RAG • Tickets • Account lookup • Audio • Human escalation")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- VOICE & AUDIO INPUT SECTION ---
with st.expander("🎙️ Voice & Audio Input (Direct Microphone or File Upload)", expanded=False):
    tab_mic, tab_upload = st.tabs([
        "🔴 Record with Browser Microphone",
        "📁 Upload Audio File"
    ])

    with tab_mic:
        st.caption("Click the red microphone to record directly in your browser:")
        recorded_audio = st.audio_input("Record voice message", key="browser_mic_input")
        mic_c1, mic_c2 = st.columns([1, 2])
        submit_mic = mic_c1.button(
            "🚀 Send Voice Message",
            disabled=recorded_audio is None,
            type="primary",
            use_container_width=True,
            key="btn_submit_mic"
        )
        auto_send_mic = mic_c2.checkbox("Auto-send when recording finishes", value=False, key="chk_auto_send_mic")

    with tab_upload:
        st.caption("Upload an existing voice message or test with our sample audio recordings:")
        audio_file = st.file_uploader(
            "Upload an audio message (WAV, MP3, M4A, etc.)",
            type=["wav", "mp3", "m4a", "ogg", "flac", "webm"],
            key="audio_file_uploader"
        )
        if audio_file is not None:
            st.audio(audio_file)
        submit_upload = st.button(
            "🚀 Process Uploaded Audio",
            disabled=audio_file is None,
            use_container_width=False,
            key="btn_submit_upload"
        )

        st.markdown("##### 🎧 Download Testing Audio Samples")
        c1, c2, c3 = st.columns(3)
        p_reset = Path("data/sample_password_reset.wav")
        if p_reset.exists():
            c1.download_button(
                "📥 Sample 1: Password FAQ",
                data=p_reset.read_bytes(),
                file_name="sample_password_reset.wav",
                mime="audio/wav",
                use_container_width=True,
                help="Speaks: 'How do I reset my password? I forgot my credentials.' -> Routes to Knowledge Agent"
            )
        p_human = Path("data/sample_human_agent.wav")
        if p_human.exists():
            c2.download_button(
                "📥 Sample 2: Live Agent",
                data=p_human.read_bytes(),
                file_name="sample_human_agent.wav",
                mime="audio/wav",
                use_container_width=True,
                help="Speaks: 'I want to speak with a human agent please, this issue is urgent.' -> Routes to Escalation Agent"
            )
        p_report = Path("data/sample_report_issue.wav")
        if p_report.exists():
            c3.download_button(
                "📥 Sample 3: Bug Report",
                data=p_report.read_bytes(),
                file_name="sample_report_issue.wav",
                mime="audio/wav",
                use_container_width=True,
                help="Speaks: 'The checkout button is broken and not working. Please create a support ticket.' -> Routes to Ticket Agent"
            )

# Resolve submitted audio from either microphone or file upload
target_audio = None
if submit_mic and recorded_audio is not None:
    target_audio = recorded_audio
    st.session_state["last_processed_audio_hash"] = hashlib.md5(recorded_audio.getvalue()).hexdigest()
elif auto_send_mic and recorded_audio is not None:
    rec_bytes = recorded_audio.getvalue()
    rec_hash = hashlib.md5(rec_bytes).hexdigest()
    if rec_hash != st.session_state.get("last_processed_audio_hash"):
        st.session_state["last_processed_audio_hash"] = rec_hash
        target_audio = recorded_audio
elif submit_upload and audio_file is not None:
    target_audio = audio_file

user_message = st.chat_input("Ask about a policy, report a problem, track a ticket, or request a human...")

# --- PROCESS INPUTS (TEXT OR AUDIO) ---
if user_message or target_audio is not None:
    current_user = st.session_state.get("verified_user")
    if user_message:
        display_message = user_message.strip()
        if not display_message:
            st.warning("Please type a message.")
            st.stop()
        st.session_state.chat_history.append({"role": "user", "content": display_message})
        with st.chat_message("user"):
            st.markdown(display_message)
        conversation_text = "\n".join(f"{x['role'].title()}: {x['content']}" for x in st.session_state.chat_history[-12:])
        with st.chat_message("assistant"):
            with st.spinner("Coordinator is routing your request..."):
                result = handle_support_request(display_message, customer=current_user, conversation=conversation_text)
                answer = result.get("response", "I couldn't prepare a response. Please try again or request human support.")
                st.markdown(answer)
                meta = {
                    "Route": result.get("route", "not specified"),
                    "Routing reason": result.get("intent_reason", ""),
                    "Knowledge sources": result.get("knowledge_sources", []),
                    "Ticket ID": result.get("ticket_id", ""),
                    "Escalation ID": result.get("escalation_id", ""),
                }
                with st.expander("Workflow details"):
                    st.write(meta)
        st.session_state.chat_history.append({"role": "assistant", "content": answer, "meta": meta})
        st.rerun()
    elif target_audio is not None:
        try:
            with st.spinner("🎙️ Transcribing audio with Whisper (~1 second)..."):
                transcript = transcribe_audio(target_audio)
        except Exception as exc:
            st.error(f"❌ Could not transcribe audio: {exc}")
            st.stop()

        if not transcript or not transcript.strip():
            st.warning("⚠️ No speech could be detected in this audio.")
            st.stop()

        # Capture audio bytes so audio player is preserved in chat history
        audio_bytes = target_audio.getvalue() if hasattr(target_audio, "getvalue") else target_audio.read()
        user_transcript_msg = f"🎙️ **Audio transcript:** \"{transcript.strip()}\""

        # 1. User message displays: Audio FIRST, then Transcribed Text AFTER
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_transcript_msg,
            "audio": audio_bytes
        })
        with st.chat_message("user"):
            st.audio(audio_bytes)
            st.markdown(user_transcript_msg)

        conversation_text = "\n".join(f"{x['role'].title()}: {x['content']}" for x in st.session_state.chat_history[-12:])

        # 2. Assistant response displays AFTER the audio and transcript
        with st.chat_message("assistant"):
            with st.spinner("🤖 Coordinator is routing to specialized agent..."):
                result = handle_support_request(transcript.strip(), customer=current_user, conversation=conversation_text)
                answer = result.get("response", "I couldn't prepare a response. Please try again or request human support.")
                st.markdown(answer)
                meta = {
                    "Route": result.get("route", "not specified"),
                    "Routing reason": result.get("intent_reason", ""),
                    "Knowledge sources": result.get("knowledge_sources", []),
                    "Ticket ID": result.get("ticket_id", ""),
                    "Escalation ID": result.get("escalation_id", ""),
                }
                with st.expander("Workflow details"):
                    st.write(meta)
        st.session_state.chat_history.append({"role": "assistant", "content": answer, "meta": meta})
        st.rerun()

# --- CONVERSATION HISTORY ---
st.divider()
conv_header_col, clear_btn_col = st.columns([5, 1])
conv_header_col.markdown("#### 💬 Conversation")
if st.session_state.chat_history:
    if clear_btn_col.button("Clear chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

for item in st.session_state.chat_history:
    with st.chat_message(item["role"]):
        if item.get("audio"):
            st.audio(item["audio"])
        st.markdown(item["content"])
        if item.get("meta"):
            with st.expander("Workflow details"):
                st.write(item["meta"])

st.divider()
st.caption("Demo notes: tickets and escalations are stored in a local SQLite database. Escalation records are saved, but this starter project does not send a real email/SMS/CRM notification.")
