"""Rule-based routing and safety verification for customer support requests.

Orders rules from most sensitive/specific (legal, escalation, billing) to most general.
All routing logic uses word boundaries and deterministic patterns.
"""
import re

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_RE = re.compile(r"(?<!\w)\+?\d[\d\s().-]{7,}\d(?!\w)")
MENTION_RE = re.compile(r"(?<![\w.])@\w+")
DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?(?:\+\d{2}:\d{2}|Z)?)?")
DM_RE = re.compile(r"\b(?:dm|dms|direct message)\b", re.I)
BROKEN_RE = re.compile(r"\b(?:via|through|using|at|visit)\s+(?:so|and|or|,|\.|!)", re.I)
TICKET_ID_RE = re.compile(r"ticket\s*(?:id|number|no\.?|#)?\s*[:#-]?\s*(\d+)")

# Intent & routing regex patterns
_LEAD_RE = re.compile(r"^(?:hi|hello|hey|please|ok|okay|so|and|also)[,!. ]+")
INFO_Q = re.compile(
    r"^(?:how (?:do|can|could|should|would|to|long|much)|how's|what(?:'s| is| are| does| do)|"
    r"where (?:do|can)|when (?:do|can|will|does)|why (?:is|are|does|do)|which|who|is there|are there|"
    r"is it|do you|does (?:the|your)|can i|could i|do i|should i)\b"
)
PROC_Q = re.compile(
    r"^(?:how (?:do|can|could|should|would|to)\b|where (?:do|can) i\b|is it possible to\b|"
    r"what(?:'s| is) the (?:process|procedure|way|steps?)\b)"
)

LEGAL_RE = re.compile(
    r"\b(?:legal action|lawsuit|lawyer|attorney|sue|suing|take (?:you|this|it|the company) to court|"
    r"go to court|report you|consumer protection|regulator)\b"
)
HUMAN_RE = re.compile(
    r"\b(?:real person|live agent|human agent|human support|human being|representative please|"
    r"customer service (?:agent|rep|representative)|connect me (?:to|with)|transfer me|"
    r"(?:speak|talk|chat) (?:to|with) (?:a |an |the )?(?:human|person|agent|representative|rep|operator|someone)|"
    r"human(?!\s+(?:resources|rights|error|factors)))\b"
)
MANAGER_RE = re.compile(r"\b(?:manager|supervisor)\b")
WANTS_RE = re.compile(r"\b(?:speak|talk|want|need|connect|escalate|give me|get me|let me|contact)\b")
COMPLAINT_RE = re.compile(r"\b(?:complaint|complain)\b")
ANGER_STRONG_RE = re.compile(
    r"\b(?:furious|unacceptable|pissed|outraged|disgusting|fed up|very angry|extremely angry|so angry|"
    r"(?:horrible|terrible|awful|worst) (?:service|support|experience|company))\b"
)
ANGER_WEAK_RE = re.compile(r"\b(?:terrible|horrible|awful|ridiculous|useless|angry|livid|appalling)\b")

BILLING_RES = [re.compile(p) for p in (
    r"\b(?:charged|billed)\b.{0,15}\b(?:twice|two times|double|again|extra|wrongly|incorrectly)\b",
    r"\bdouble[- ]?(?:charged|billed|charge)\b",
    r"\bover[- ]?charged\b",
    r"\bunauthori[sz]ed (?:charge|transaction|payment|purchase)s?\b",
    r"\bchargeback\b",
    r"\b(?:i want|i need|i'd like|i would like|give me|issue me|process my)\b.{0,30}\brefund\b",
    r"\brefund (?:me|my)\b|\bwhere is my refund\b|\bmoney back\b",
)]

CREATE_TICKET_RE = re.compile(
    r"\b(?:create|open|raise|submit|file|make|start|log|need|want)\s+(?:me\s+)?(?:an? |new |another )*"
    r"(?:support |help |service )?ticket\b|\breport (?:an? |the )(?:problem|issue|bug)\b"
)
TRACK_RE = re.compile(
    r"\b(?:ticket status|track(?:ing)? (?:my |the )?ticket|status of (?:my |the )?ticket|existing ticket|"
    r"ticket update|my tickets|open tickets|update on (?:my |the )?ticket|check (?:my |the )?ticket)\b"
)

DELIVERY_RES = [re.compile(p) for p in (
    r"\bwhere(?:'s| is) my (?:order|package|parcel|delivery|shipment|items?)\b",
    r"\b(?:order|package|parcel|delivery|shipment|item)\b.{0,40}\b(?:hasn't|has not|haven't|have not|never|not)\s+"
    r"(?:yet\s+)?(?:arrived|come|been delivered|delivered|shown up)\b",
    r"\b(?:hasn't|has not|never|not yet)\s+(?:arrived|been delivered|come)\b",
    r"\b(?:late|delayed|missing|lost|damaged|wrong) (?:order|package|parcel|delivery|shipment|item)\b",
    r"\bstill waiting (?:for|on) my\b",
)]
PROBLEM_RES = [re.compile(p) for p in (
    r"\b(?:can'?t|cannot|unable to|not able to|won'?t let me|couldn'?t)\s+"
    r"(?:log ?in|sign ?in|log into|sign into|access|open|reset|connect|pay|check ?out|download|upload|use)\b",
    r"\b(?:not|isn'?t|aren'?t|wasn'?t|doesn'?t|does not|don'?t|do not|stopped|won'?t|will not)\s+"
    r"(?:working|work|loading|load|responding|opening|open|syncing)\b",
    r"\b(?:crash(?:es|ed|ing)?|freez(?:es|ing)|froze|broken|glitch(?:es|ing)?|bug)\b",
    r"\berror(?: message| code)?\b",
    r"\b(?:locked out|account (?:is )?locked)\b",
    r"\b(?:having|have|got|facing|experiencing|there's|there is) (?:an? )?(?:issue|problem)s? with\b",
)]
ACCOUNT_RE = re.compile(
    r"\bmy (?:e-?mail(?: address)?|profile|customer id|(?:personal|contact) (?:details|info|information)|account)\b|"
    r"\b(?:e-?mail|details|profile)\b.{0,25}\b(?:on|in|linked to|with|for)\s+(?:my\s+)?account\b|"
    r"\baccount (?:information|info|details)\b|\bwho am i\b"
)
CHANGE_RE = re.compile(r"\b(?:change|update|reset|edit|switch|modify|delete|remove)\b")


def _norm(text: str) -> str:
    """Lower-case, turn curly quotes into straight ones, collapse whitespace."""
    text = (text or "").lower().replace("\u2019", "'").replace("\u2018", "'").replace("`", "'")
    return re.sub(r"\s+", " ", text).strip()


def _has(text: str, words) -> bool:
    """True if any word/phrase appears as a WHOLE word (so 'sue' does not match 'issue')."""
    return any(re.search(rf"\b{re.escape(w)}\b", text) for w in words)


def _model_text_ok(answer: str, required=()) -> bool:
    """Safety check for text generated by the model.

    Rejects empty text, invented links / phone numbers / @mentions, 'please DM us' replies,
    sentences cut off by data cleaning, and replies missing a required fact (e.g. ticket id).
    """
    if not answer or not answer.strip():
        return False
    a = answer.strip()
    scan = DATE_RE.sub(" ", a)
    if URL_RE.search(scan) or PHONE_RE.search(scan) or MENTION_RE.search(scan):
        return False
    if DM_RE.search(a) or a.endswith(":") or BROKEN_RE.search(a):
        return False
    return all(str(r) in a for r in required)


def _is_grounded(answer: str, context: str, threshold: float = 0.4) -> bool:
    """Cheap grounding check: enough of the answer's content words must appear in context."""
    tokens = set(re.findall(r"[a-z0-9']{4,}", answer.lower()))
    if not tokens:
        return False
    ctx = set(re.findall(r"[a-z0-9']{4,}", context.lower()))
    return len(tokens & ctx) / len(tokens) >= threshold


def _rule_route(message: str) -> tuple[str, str]:
    """Return (route, reason). Order matters: most sensitive / most specific first."""
    text = _norm(message)
    if not text:
        return "knowledge", "Empty message routed to company knowledge search."
    core = _LEAD_RE.sub("", text)
    info_q = bool(INFO_Q.match(core))
    procedural = bool(PROC_Q.match(core))

    # 1. Escalation to a human
    if LEGAL_RE.search(text):
        return "escalation", "Customer mentioned potential legal action or regulatory reporting."
    if HUMAN_RE.search(text):
        return "escalation", "Customer explicitly requested to speak with a human support agent."
    if MANAGER_RE.search(text) and (not info_q or WANTS_RE.search(text)):
        return "escalation", "Customer requested to speak with a manager or supervisor."
    if COMPLAINT_RE.search(text) and not info_q:
        return "escalation", "Customer requested to file a formal complaint."
    if ANGER_STRONG_RE.search(text) or len(set(ANGER_WEAK_RE.findall(text))) >= 2:
        return "escalation", "Customer expressed strong frustration or anger regarding the service."
    if not info_q and any(p.search(text) for p in BILLING_RES):
        return "escalation", "Customer reports a billing dispute (duplicate/unauthorized charge or refund request)."

    # 2. Tickets
    if CREATE_TICKET_RE.search(text):
        return "create_ticket", "Customer explicitly asked to open a support ticket."
    if TRACK_RE.search(text) or TICKET_ID_RE.search(text):
        return "track_ticket", "Customer wants to check an existing ticket."

    # 3. Problem reports (only statements, not "how do I..." questions)
    if not info_q:
        if any(p.search(text) for p in DELIVERY_RES):
            return "create_ticket", "Customer reports a delivery problem."
        if any(p.search(text) for p in PROBLEM_RES):
            return "create_ticket", "Request appears to report a problem or create a support ticket."

    # 4. Private account details (but "how do I change my email" is a knowledge question)
    if ACCOUNT_RE.search(text) and not (procedural or CHANGE_RE.search(text)):
        return "account", "Request appears to concern private account information."

    return "knowledge", "General question routed to company knowledge search."
