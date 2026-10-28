"""Human hand-off briefing generation for customer support escalations.

Builds structured, issue-specific briefings including category, urgency,
customer profile, root cause summary, entities (orders, tickets, amounts),
and actionable next steps for human agents.
"""
import re
from typing import Optional

from .routing import (
    _norm,
    _has,
    HUMAN_RE,
    MANAGER_RE,
    LEGAL_RE,
    BILLING_RES,
)

_STATEMENT_RE = re.compile(r"^\*{0,2}(?:user|customer|human)\*{0,2}\s*:\s*\*{0,2}\s*(.+)$", re.IGNORECASE)
_GENERIC_HANDOFF = {
    "human", "agent", "human please", "i need a human", "live agent", "real person",
    "representative", "operator", "human agent"
}

BILLING_KW = [
    "refund", "refunds", "charged", "charge", "charges", "payment", "billing", "credit card", "invoice",
    "receipt", "overcharged", "double charge", "duplicate charge", "fee", "fees", "money back"
]
ACCESS_KW = [
    "login", "log in", "logged out", "password", "locked out", "credentials", "pin", "2fa",
    "two-factor", "sign in", "verification code"
]
SUBSCRIPTION_KW = ["cancel", "cancellation", "subscription", "unsubscribe", "renew", "renewal"]
DELIVERY_KW = ["package", "shipping", "shipment", "tracking", "delivered", "delivery", "return", "returns"]
TECH_KW = [
    "broken", "error", "crash", "crashes", "crashing", "not working", "doesn't work", "does not work",
    "bug", "failed", "glitch", "button", "freezing", "timeout"
]
LEGAL_KW = ["legal", "sue", "suing", "lawyer", "attorney", "court", "report you", "take to court", "lawsuit"]
COMPLAINT_KW = ["complaint", "furious", "angry", "unacceptable", "horrible", "terrible", "worst service"]
ORDER_RE = re.compile(r"(?<!in )\borders?\b(?! to\b)")
ERROR_CODE_RE = re.compile(r"\b(?:error|status|code|http)\s*(?:code\s*)?(404|500|502|503)\b|\b(404|500|502|503)\s*(?:error|page)\b")


def _is_generic_handoff(statement: str) -> bool:
    """True for 'talk to a human' style messages that contain no real problem description."""
    s = _norm(statement)
    return s in _GENERIC_HANDOFF or (bool(HUMAN_RE.search(s) or MANAGER_RE.search(s)) and len(s.split()) <= 8)


def _customer_statements(conversation: str) -> list[str]:
    statements = []
    for line in (conversation or "").splitlines():
        m = _STATEMENT_RE.match(line.strip())
        if m:
            text = re.sub(r"^🎙️\s*\*\*Audio transcript:\*\*\s*[\"']?|[\"']$", "", m.group(1).strip()).strip()
            if text:
                statements.append(text)
    return statements


def generate_handoff_summary(
    message: str,
    conversation: str = "",
    reason: str = "",
    user: Optional[dict] = None,
    customer_id: Optional[str] = None,
) -> str:
    """Generate a structured, issue-specific handoff briefing for a human support representative."""
    # 1. All customer statements in the conversation
    customer_statements = _customer_statements(conversation)
    if message and (not customer_statements or message.strip() != customer_statements[-1]):
        customer_statements.append(message.strip())
    combined_text = _norm(" ".join(customer_statements))

    # 2. Category (most serious first) and recommended action
    category = "General Support Handoff"
    rec_action = "Review full conversation transcript and reach out directly to assist the customer."
    if _has(combined_text, LEGAL_KW) or LEGAL_RE.search(combined_text):
        category = "Legal / Compliance Escalation"
        rec_action = "Treat with high priority. Handoff to compliance/legal relations or senior customer retention management."
    elif _has(combined_text, BILLING_KW) or any(p.search(combined_text) for p in BILLING_RES):
        category = "Billing & Financial Dispute"
        rec_action = "Inspect customer transaction logs in payment gateway. Verify any disputed/duplicate charges and issue refund or correction as appropriate."
    elif _has(combined_text, ACCESS_KW):
        category = "Account Access & Authentication"
        rec_action = "Verify customer identity and check account security status. Provide secure password reset or unlock assistance."
    elif _has(combined_text, SUBSCRIPTION_KW):
        category = "Subscription & Cancellation"
        rec_action = "Confirm the customer's plan and billing date, process the cancellation or change, and confirm it in writing."
    elif ORDER_RE.search(combined_text) or _has(combined_text, DELIVERY_KW):
        category = "Order Status & Fulfillment"
        rec_action = "Look up order in fulfillment system, check logistics tracking status, and provide updated delivery timeline or replacement."
    elif _has(combined_text, TECH_KW) or ERROR_CODE_RE.search(combined_text):
        category = "Technical Bug / Platform Issue"
        rec_action = "Review reported platform error, check system status/logs, and escalate to engineering or create an internal bug ticket."
    elif _has(combined_text, COMPLAINT_KW):
        category = "Customer Dissatisfaction & Formal Complaint"
        rec_action = "Contact customer promptly with priority outreach. Acknowledge frustration, investigate service failure, and provide resolution."

    # 3. Urgency
    urgency = "NORMAL"
    if _has(combined_text, LEGAL_KW) or LEGAL_RE.search(combined_text) or _has(combined_text, ["fraud", "stolen", "hacked"]):
        urgency = "CRITICAL (Legal / Compliance Risk)"
    elif (_has(combined_text, ["unacceptable", "furious", "angry", "double charged", "duplicate charge", "overcharged",
                               "locked out", "urgent", "emergency"])
          or re.search(r"\bcharged\b.*\btwice\b", combined_text)
          or any(p.search(combined_text) for p in BILLING_RES[:5])):
        urgency = "HIGH (Immediate Attention Required)"
    elif _has(combined_text, ["manager", "supervisor", "complaint", "not working", "broken", "dispute", "cannot"]) \
            or ERROR_CODE_RE.search(combined_text):
        urgency = "MEDIUM (Escalated Request)"

    # 4. Core issue (ignore messages that only say 'talk to a human')
    substantive = [s for s in customer_statements if len(s.split()) >= 3 and not _is_generic_handoff(s)]
    if substantive:
        core_issue = substantive[0]
        if len(substantive) > 1 and substantive[-1] != substantive[0]:
            core_issue += f" (Follow-up: \"{substantive[-1]}\")"
    else:
        core_issue = message if message else "Customer requested immediate human intervention without providing initial details."

    # 5. Key entities
    details = []
    orders = re.findall(r"order\s*(?:id|number|no\.?|#)?\s*[:#-]?\s*(\d{3,})", combined_text)
    if orders:
        details.append(f"- **Order Number(s):** #{', #'.join(sorted(set(orders)))}")
    tickets = re.findall(r"ticket\s*(?:id|number|no\.?|#)?\s*[:#-]?\s*(\d+)", combined_text)
    if tickets:
        details.append(f"- **Ticket Number(s):** #{', #'.join(sorted(set(tickets)))}")
    amounts = re.findall(r"\$\s*\d+(?:,\d{3})*(?:\.\d{2})?|\b\d+(?:\.\d{2})?\s*(?:dollars|usd)\b", combined_text)
    if amounts:
        details.append(f"- **Mentioned Amount(s):** {', '.join(sorted(set(amounts)))}")
    codes = {a or b for a, b in ERROR_CODE_RE.findall(combined_text)} | set(re.findall(r"\b(timeout|timed out)\b", combined_text))
    if codes:
        details.append(f"- **Error Detected:** {', '.join(sorted(codes)).upper()}")

    # 6. Customer identity
    if user and user.get("customer_id"):
        cust_info = f"**{user.get('display_name', 'Customer')}** (ID: `{user['customer_id']}`) | `{user.get('email', 'No email')}`"
    elif customer_id:
        cust_info = f"Customer ID: `{customer_id}`"
    else:
        cust_info = "Guest User (Unauthenticated)"

    # 7. Assemble
    lines = [
        f"**Category:** {category}",
        f"**Urgency:** [{urgency}]",
        f"**Customer:** {cust_info}",
        f"**Handoff Trigger:** {reason}",
        "",
        "**Core Customer Problem:**",
        f"> {core_issue}",
    ]
    if details:
        lines += ["", "**Key Entities & Details:**", *details]
    lines += ["", f"**Recommended Action:**\n{rec_action}"]
    return "\n".join(lines)
