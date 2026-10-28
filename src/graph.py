"""LangGraph workflow: audio (optional) -> coordinator -> specialized support capability.

Orchestrates multi-agent routing between Knowledge, Ticket, User, and Escalation agents.
Intent classification rules are in src/routing.py; human handoff generation is in src/handoff.py.
"""
import logging
import re
from typing import TypedDict, Optional, Any

from langgraph.graph import StateGraph, START, END

from .audio import transcribe_audio
from .database import create_ticket, get_ticket, list_tickets, create_escalation, log_interaction
from .knowledge import retrieve, find_direct_answer, extract_best_paragraph
from .model import try_generate_response
from .routing import (
    _norm,
    _rule_route,
    _model_text_ok,
    _is_grounded,
    TICKET_ID_RE,
    EMAIL_RE,
)
from .handoff import (
    generate_handoff_summary,
    _customer_statements,
    _is_generic_handoff,
)

logger = logging.getLogger(__name__)

MAX_MESSAGE_CHARS = 2000
OPEN_STATUSES = {"open", "in_progress"}
NO_ANSWER_MESSAGE = (
    "I couldn't find a reliable answer to that in our help articles. You can rephrase your question, "
    "say **'create a ticket'** and I'll log it for our support team, or ask me to **connect you with a human**."
)


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


# ======================================================================================
# Ticket helpers
# ======================================================================================
def _ticket_text(message: str, conversation: str) -> str:
    """Text to store in the ticket. For a bare 'create a ticket' use the customer's earlier problem."""
    stripped = re.sub(
        r"\b(?:please|can you|could you|i want to|i would like to|i'd like to|i need to|for me|about|this|that|it|my|"
        r"the|issue|problem|report|a|an|new|support|ticket|create|open|raise|submit|file|make|start|log)\b",
        " ", _norm(message))
    if len(stripped.split()) >= 3:
        return message.strip()
    earlier = [s for s in _customer_statements(conversation)
               if _norm(s) != _norm(message) and len(s.split()) >= 3 and not _is_generic_handoff(s)]
    return earlier[-1] if earlier else (message.strip() or "Customer support request")


def _make_subject(text: str, limit: int = 80) -> str:
    first = re.split(r"(?<=[.!?])\s+", text.strip())[0] or "Customer support request"
    if len(first) <= limit:
        return first
    return first[:limit].rsplit(" ", 1)[0].rstrip(",;:") + "..."


# ======================================================================================
# Graph nodes
# ======================================================================================
def audio_processor(state: SupportState) -> SupportState:
    try:
        transcript = transcribe_audio(state["audio_file"])
        return {"message": transcript, "transcript": transcript}
    except Exception as exc:
        err_msg = f"I couldn't transcribe that audio: {exc}"
        return {"response": err_msg, "route": "audio_error", "transcript": f"[Audio transcription error: {exc}]"}


def coordinator(state: SupportState) -> SupportState:
    if state.get("route") == "audio_error":
        return {}
    route, reason = _rule_route(state.get("message", ""))
    return {"route": route, "intent_reason": reason}


def knowledge_agent(state: SupportState) -> SupportState:
    query = state.get("message", "").strip()
    context, sources = retrieve(query)
    direct_answer = find_direct_answer(query)

    if direct_answer:
        answer = direct_answer
    elif context:
        answer = try_generate_response(query, context=context)
        if not (_model_text_ok(answer) and _is_grounded(answer, context)):
            answer = extract_best_paragraph(query, context)
        if not answer:
            answer = NO_ANSWER_MESSAGE
    else:
        answer = NO_ANSWER_MESSAGE
    return {"knowledge_context": context, "knowledge_sources": sources, "response": answer}


def ticket_agent(state: SupportState) -> SupportState:
    if not state.get("authorized") or not state.get("customer_id"):
        return {"response": "Please sign in using your customer ID and verification PIN before I access or create account-linked tickets."}
    message = state.get("message", "").strip()
    customer_id = state["customer_id"]

    if state.get("route") == "track_ticket":
        match = TICKET_ID_RE.search(_norm(message))
        if not match:
            items = list_tickets(customer_id)
            if not items:
                return {"response": "I couldn't find any tickets associated with your account."}
            summary = "Your recent tickets:\n" + "\n".join(
                f"Ticket #{t['ticket_id']}: {t['subject']} — {t['status']}" for t in items
            )
            return {"tool_result": summary, "response": summary}
        ticket = get_ticket(int(match.group(1)), customer_id)
        if not ticket:
            return {"response": "I couldn't find that ticket under your signed-in account. Please check the ticket number."}
        result = f"Ticket #{ticket['ticket_id']} | Subject: {ticket['subject']} | Status: {ticket['status']} | Created: {ticket['created_at']}"
        answer = try_generate_response(message, tool_result=result)
        if not _model_text_ok(answer, required=[ticket["ticket_id"]]):
            answer = f"Ticket **#{ticket['ticket_id']}** ('{ticket['subject']}') is currently **{ticket['status']}** (created on {ticket['created_at']})."
        return {"tool_result": result, "response": answer, "ticket_id": ticket["ticket_id"]}

    # Create a ticket
    description = _ticket_text(message, state.get("conversation", ""))
    subject = _make_subject(description)

    for t in list_tickets(customer_id, limit=20):
        if t["subject"].strip().lower() == subject.strip().lower() and t["status"] in OPEN_STATUSES:
            return {
                "tool_result": f"Existing ticket #{t['ticket_id']} ({t['status']}).",
                "ticket_id": t["ticket_id"],
                "response": (f"You already have an open ticket for this: **#{t['ticket_id']}** ('{t['subject']}'), "
                             f"status **{t['status']}**. I haven't created a duplicate; our team is already on it."),
            }

    ticket_id = create_ticket(customer_id, subject, description)
    result = f"Ticket #{ticket_id} was created successfully. Current status: open."
    answer = try_generate_response(message, tool_result=result)
    if not _model_text_ok(answer, required=[ticket_id]):
        answer = (
            f"I have created support ticket **#{ticket_id}** for you with status **open**.\n\n"
            f"**Subject:** {subject}\n\n"
            "Our support team has logged your report and will investigate. You can track updates directly in your account dashboard."
        )
    return {"tool_result": result, "response": answer, "ticket_id": ticket_id}


def user_agent(state: SupportState) -> SupportState:
    if not state.get("authorized") or not state.get("customer_id"):
        return {"response": "For your privacy, please sign in with your customer ID and verification PIN before I can look up account information."}
    user = state.get("verified_user")
    if not user:
        return {"response": "I couldn't access a verified account profile. Please sign in again."}
    result = f"Verified account: customer ID {user['customer_id']}; name {user['display_name']}; email {user['email']}."
    msg = state.get("message", "").lower()
    answer = try_generate_response(state.get("message", ""), tool_result=result)

    wants_details = any(k in msg for k in ["email", "address", "profile", "account"])
    required = [user["email"]] if wants_details else []
    if not _model_text_ok(answer, required=required):
        answer = f"Hello {user['display_name']}! Your registered account email address is **{user['email']}** (Customer ID: {user['customer_id']})."
    return {"tool_result": result, "response": answer}


def escalation_agent(state: SupportState) -> SupportState:
    conversation = state.get("conversation", "")
    message = state.get("message", "").strip()
    reason = state.get("intent_reason", "Human support requested.")
    user = state.get("verified_user")
    customer_id = state.get("customer_id") if state.get("authorized") else None

    summary = generate_handoff_summary(
        message=message,
        conversation=conversation,
        reason=reason,
        user=user,
        customer_id=customer_id,
    )
    escalation_id = create_escalation(
        customer_id,
        reason,
        summary,
        conversation + (f"\nCustomer: {message}" if message else ""),
    )
    return {
        "escalation_required": True,
        "escalation_id": escalation_id,
        "response": (
            f"I've forwarded your request to human support as escalation #{escalation_id}. "
            "A detailed issue briefing and your conversation history have been forwarded to a human representative."
        ),
    }


# ======================================================================================
# Graph wiring & execution
# ======================================================================================
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

    workflow.add_conditional_edges(
        START,
        lambda s: "audio_processor" if s.get("audio_file") else "coordinator",
        {"audio_processor": "audio_processor", "coordinator": "coordinator"}
    )
    workflow.add_conditional_edges(
        "audio_processor",
        lambda s: END if s.get("route") == "audio_error" else "coordinator",
        {END: END, "coordinator": "coordinator"}
    )
    workflow.add_conditional_edges("coordinator", _route_after_coordinator, {
        "knowledge_agent": "knowledge_agent",
        "ticket_agent": "ticket_agent",
        "user_agent": "user_agent",
        "escalation_agent": "escalation_agent",
        END: END,
    })
    for node in ["knowledge_agent", "ticket_agent", "user_agent", "escalation_agent"]:
        workflow.add_edge(node, END)
    return workflow.compile()


support_graph = build_graph()


def handle_support_request(message: str, customer: dict | None = None, conversation: str = "", audio_file=None) -> dict:
    customer = customer or {}
    state: SupportState = {
        "message": (message or "")[:MAX_MESSAGE_CHARS],
        "audio_file": audio_file,
        "customer_id": customer.get("customer_id", ""),
        "authorized": bool(customer.get("customer_id")),
        "verified_user": customer if customer.get("customer_id") else None,
        "conversation": conversation,
    }
    result = support_graph.invoke(state)

    try:
        route = result.get("route", "knowledge")
        user_msg = state.get("message") or (f"[Audio] {result.get('transcript', '')}" if audio_file else "User request")
        resp_text = result.get("response", "")
        if route == "account":
            resp_text = "[account details shown to the verified customer]"
        else:
            resp_text = EMAIL_RE.sub("[email]", resp_text)

        summary_map = {
            "escalation": result.get("intent_reason", "Human escalation requested"),
            "knowledge": f"Retrieved knowledge from: {', '.join(result.get('knowledge_sources', []))}" if result.get("knowledge_sources") else "Knowledge search",
            "create_ticket": f"Created support ticket #{result.get('ticket_id', '')}",
            "track_ticket": f"Checked status for ticket #{result.get('ticket_id', '')}",
            "account": f"Account inquiry for {state.get('customer_id', 'user')}",
            "audio_error": "Audio transcription failure",
        }
        summary_text = summary_map.get(route, "")

        log_interaction(
            customer_id=state.get("customer_id") if state.get("authorized") else None,
            route=route,
            user_message=user_msg,
            response=resp_text,
            summary=summary_text,
            escalation_id=result.get("escalation_id"),
        )
    except Exception:
        logger.warning("Failed to write interaction log", exc_info=True)

    return result
