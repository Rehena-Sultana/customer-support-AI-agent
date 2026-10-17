"""LangGraph workflow: audio (optional) -> coordinator -> specialized support capability."""
from typing import TypedDict, Optional, Any
import re
from langgraph.graph import StateGraph, START, END
from .audio import transcribe_audio
from .database import create_ticket, get_ticket, list_tickets, create_escalation
from .knowledge import retrieve
from .model import try_generate_response


class SupportState(TypedDict, total=False):
    message: str
    audio_file: Any
    transcript: str
    customer_id: str
    authorized: bool
    verified_user: Optional[dict]
    conversation: str
    route: str
    intent_reason: str
    knowledge_context: str
    knowledge_sources: list[str]
    tool_result: str
    response: str
    escalation_required: bool
    escalation_id: int
    ticket_id: int


def _rule_route(message: str) -> tuple[str, str]:
    text = message.lower()
    angry = any(p in text for p in ["very angry", "furious", "unacceptable", "complaint", "report you", "legal action"])
    asks_human = any(p in text for p in ["human", "real person", "live agent", "talk to an agent", "manager", "representative"])
    if angry or asks_human:
        return "escalation", "Customer requested a person or expressed strong frustration."
    if any(p in text for p in ["my account", "my email", "what is my email", "my profile", "account information", "change my email", "my personal details"]):
        return "account", "Request appears to concern private account information."
    ticket_id_match = re.search(r"(?:ticket\s*(?:id|number|#)?\s*[:#-]?\s*)(\d+)", text)
    tracking = any(p in text for p in ["ticket status", "track ticket", "status of ticket", "existing ticket", "ticket update"])
    if tracking or (ticket_id_match and any(p in text for p in ["status", "track", "update"])):
        return "track_ticket", "Customer wants to check an existing ticket."
    create_words = ["create a ticket", "open a ticket", "raise a ticket", "submit a ticket", "file a ticket", "report a problem", "not working", "doesn't work", "does not work", "cannot login", "cannot log in", "can't log in", "can't login", "error message", "broken", "issue with"]
    if any(p in text for p in create_words):
        return "create_ticket", "Request appears to report a problem or create a support ticket."
    return "knowledge", "General question routed to company knowledge search."


def audio_processor(state: SupportState) -> SupportState:
    try:
        transcript = transcribe_audio(state["audio_file"])
        return {"message": transcript, "transcript": transcript}
    except Exception as exc:
        return {"response": f"I couldn't transcribe that audio: {exc}", "route": "audio_error"}


def coordinator(state: SupportState) -> SupportState:
    if state.get("route") == "audio_error":
        return {}
    route, reason = _rule_route(state.get("message", ""))
    return {"route": route, "intent_reason": reason}


def knowledge_agent(state: SupportState) -> SupportState:
    context, sources = retrieve(state.get("message", ""))
    if context:
        answer = try_generate_response(state.get("message", ""), context=context)
        if not answer:
            answer = "I found these relevant company knowledge excerpts:\n\n" + context + "\n\nPlease let me know if you would like me to open a support ticket."
    else:
        answer = (
            "I couldn't find a reliable answer in the current company knowledge files. "
            "Please add the relevant FAQ or policy document, rephrase your question, or ask me to connect you with human support."
        )
    return {"knowledge_context": context, "knowledge_sources": sources, "response": answer}


def ticket_agent(state: SupportState) -> SupportState:
    if not state.get("authorized") or not state.get("customer_id"):
        return {"response": "Please sign in using your customer ID and verification PIN before I access or create account-linked tickets."}
    message = state.get("message", "").strip()
    if state.get("route") == "track_ticket":
        match = re.search(r"(?:ticket\s*(?:id|number|#)?\s*[:#-]?\s*)(\d+)", message.lower())
        if not match:
            items = list_tickets(state["customer_id"])
            if not items:
                return {"response": "I couldn't find any tickets associated with your account."}
            summary = "Your recent tickets:\n" + "\n".join(
                f"Ticket #{t['ticket_id']}: {t['subject']} — {t['status']}" for t in items
            )
            return {"tool_result": summary, "response": summary}
        ticket = get_ticket(int(match.group(1)), state["customer_id"])
        if not ticket:
            return {"response": "I couldn't find that ticket under your signed-in account. Please check the ticket number."}
        result = f"Ticket #{ticket['ticket_id']} | Subject: {ticket['subject']} | Status: {ticket['status']} | Created: {ticket['created_at']}"
        answer = try_generate_response(message, tool_result=result) or result
        return {"tool_result": result, "response": answer, "ticket_id": ticket["ticket_id"]}

    subject = message[:100].strip() or "Customer support request"
    ticket_id = create_ticket(state["customer_id"], subject, message)
    result = f"Ticket #{ticket_id} was created successfully. Current status: open."
    answer = try_generate_response(message, tool_result=result) or (
        f"I've created support ticket #{ticket_id} for you. Its current status is open. "
        "Please keep the ticket number for future updates."
    )
    return {"tool_result": result, "response": answer, "ticket_id": ticket_id}


def user_agent(state: SupportState) -> SupportState:
    # Account data is only queried after successful authentication in the UI.
    if not state.get("authorized") or not state.get("customer_id"):
        return {"response": "For your privacy, please sign in with your customer ID and verification PIN before I can look up account information."}
    # authenticate_customer returned only safe display fields; the app passes those as a trusted result.
    user = state.get("verified_user")
    if not user:
        return {"response": "I couldn't access a verified account profile. Please sign in again."}
    result = f"Verified account: customer ID {user['customer_id']}; name {user['display_name']}; email {user['email']}."
    answer = try_generate_response(state.get("message", ""), tool_result=result) or result
    return {"tool_result": result, "response": answer}


def escalation_agent(state: SupportState) -> SupportState:
    conversation = state.get("conversation", "")
    message = state.get("message", "")
    reason = state.get("intent_reason", "Human support requested.")
    summary = f"Reason: {reason}\nCustomer message/transcript: {message}\nConversation context:\n{conversation[-5000:]}"
    escalation_id = create_escalation(
        state.get("customer_id") if state.get("authorized") else None,
        reason,
        summary,
        conversation + f"\nCustomer: {message}",
    )
    return {
        "escalation_required": True,
        "escalation_id": escalation_id,
        "response": (
            f"I've forwarded your request to human support as escalation #{escalation_id}. "
            "The handoff includes your message and conversation context. A human representative will need to review it; this demo does not send a real notification."
        ),
    }


def _route_after_coordinator(state: SupportState) -> str:
    route = state.get("route", "knowledge")
    return {
        "knowledge": "knowledge_agent",
        "create_ticket": "ticket_agent",
        "track_ticket": "ticket_agent",
        "account": "user_agent",
        "escalation": "escalation_agent",
        "audio_error": END,
    }.get(route, "knowledge_agent")


def build_graph():
    workflow = StateGraph(SupportState)
    workflow.add_node("audio_processor", audio_processor)
    workflow.add_node("coordinator", coordinator)
    workflow.add_node("knowledge_agent", knowledge_agent)
    workflow.add_node("ticket_agent", ticket_agent)
    workflow.add_node("user_agent", user_agent)
    workflow.add_node("escalation_agent", escalation_agent)
    workflow.add_conditional_edges(START, lambda s: "audio_processor" if s.get("audio_file") else "coordinator",
                                   {"audio_processor": "audio_processor", "coordinator": "coordinator"})
    workflow.add_conditional_edges("audio_processor", lambda s: END if s.get("route") == "audio_error" else "coordinator",
                                   {END: END, "coordinator": "coordinator"})
    workflow.add_conditional_edges("coordinator", _route_after_coordinator, {
        "knowledge_agent": "knowledge_agent", "ticket_agent": "ticket_agent",
        "user_agent": "user_agent", "escalation_agent": "escalation_agent", END: END,
    })
    for node in ["knowledge_agent", "ticket_agent", "user_agent", "escalation_agent"]:
        workflow.add_edge(node, END)
    return workflow.compile()


support_graph = build_graph()


def handle_support_request(message: str, customer: dict | None = None, conversation: str = "", audio_file=None) -> dict:
    customer = customer or {}
    state: SupportState = {
        "message": message or "",
        "audio_file": audio_file,
        "customer_id": customer.get("customer_id", ""),
        "authorized": bool(customer.get("customer_id")),
        "verified_user": customer if customer.get("customer_id") else None,
        "conversation": conversation,
    }
    result = support_graph.invoke(state)
    return result
