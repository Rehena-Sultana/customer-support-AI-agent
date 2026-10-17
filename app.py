import streamlit as st
from src.database import init_db, authenticate_customer, list_tickets
from src.graph import handle_support_request
from src.audio import transcribe_audio
from src.config import ADAPTER_DIR, DISABLE_MODEL

st.set_page_config(page_title="Multi-Agent Customer Support", page_icon="💬", layout="wide")
init_db()

st.title("💬 Multi-Agent Customer Support")
st.caption("LangGraph • Fine-tuned Qwen LoRA • RAG • Tickets • Account lookup • Audio • Human escalation")

with st.sidebar:
    st.header("Customer sign-in (demo)")
    st.caption("Demo accounts only. Replace this PIN-based sign-in with your real authentication provider before production.")
    customer_id_input = st.text_input("Customer ID", value="CUST-1001")
    pin_input = st.text_input("Verification PIN", value="2468", type="password")
    if st.button("Sign in", use_container_width=True):
        user = authenticate_customer(customer_id_input, pin_input)
        if user:
            st.session_state["verified_user"] = user
            st.success(f"Signed in as {user['display_name']}.")
        else:
            st.session_state.pop("verified_user", None)
            st.error("Customer ID or PIN is incorrect.")
    if st.session_state.get("verified_user"):
        user = st.session_state["verified_user"]
        st.success(f"Signed in: {user['display_name']}")
        if st.button("Sign out", use_container_width=True):
            st.session_state.pop("verified_user", None)
            st.rerun()
        st.subheader("Your tickets")
        for ticket in list_tickets(user["customer_id"]):
            st.write(f"**#{ticket['ticket_id']}** · {ticket['subject']} · {ticket['status']}")
    st.divider()
    st.subheader("Model status")
    st.write("Model disabled" if DISABLE_MODEL else "LoRA adapter configured")
    st.caption(f"Adapter path: `{ADAPTER_DIR}`")
    st.caption("If the adapter is missing or model loading fails, the app uses safe fallback responses where possible.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for item in st.session_state.chat_history:
    with st.chat_message(item["role"]):
        st.markdown(item["content"])
        if item.get("meta"):
            with st.expander("Workflow details"):
                st.write(item["meta"])

st.markdown("#### Send a message")
audio_file = st.file_uploader("Or upload an audio message (WAV, MP3, M4A, etc.)", type=["wav", "mp3", "m4a", "ogg", "flac", "webm"])
user_message = st.chat_input("Ask about a policy, report a problem, track a ticket, or request a human...")

submitted_audio = st.button("Process uploaded audio", disabled=audio_file is None, use_container_width=False)
if user_message or submitted_audio:
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
    elif submitted_audio and audio_file is not None:
        conversation_text = "\n".join(f"{x['role'].title()}: {x['content']}" for x in st.session_state.chat_history[-12:])
        with st.spinner("Audio Processor is transcribing your message..."):
            result = handle_support_request("", customer=current_user, conversation=conversation_text, audio_file=audio_file)
        transcript = result.get("transcript", "")
        if transcript:
            st.session_state.chat_history.append({"role": "user", "content": f"🎙️ **Audio transcript:** {transcript}"})
            with st.chat_message("user"):
                st.markdown(f"🎙️ **Audio transcript:** {transcript}")
        with st.chat_message("assistant"):
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

if st.session_state.chat_history:
    if st.button("Clear conversation"):
        st.session_state.chat_history = []
        st.rerun()

st.divider()
st.caption("Demo notes: tickets and escalations are stored in a local SQLite database. Escalation records are saved, but this starter project does not send a real email/SMS/CRM notification.")
