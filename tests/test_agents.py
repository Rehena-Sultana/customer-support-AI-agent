"""Automated Test Suite for Multi-Agent Customer Support System.
Tests each section/agent: Coordinator, Knowledge Agent, Ticket Agent, User Agent, Escalation Agent, Audio Processor, and End-to-End LangGraph workflow.
"""
import unittest
from unittest.mock import patch, MagicMock
import tempfile
import sqlite3
from pathlib import Path

# Ensure model generation does not attempt heavy GPU/LLM loading in unit tests
import os
os.environ["SUPPORT_DISABLE_MODEL"] = "1"

from src.graph import (
    _rule_route,
    _route_after_coordinator,
    coordinator,
    knowledge_agent,
    ticket_agent,
    user_agent,
    escalation_agent,
    audio_processor,
    handle_support_request,
    SupportState,
)
import src.database as db


class BaseDatabaseTestCase(unittest.TestCase):
    """Sets up an isolated, temporary SQLite database for each test to avoid mutating production data."""

    def setUp(self):
        self.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.temp_db.close()
        self.db_path = Path(self.temp_db.name)

        # Patch db._connect to point to our temp database
        self.orig_connect = db._connect
        db._connect = lambda: self._get_test_conn()
        db.init_db()

    def _get_test_conn(self):
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def tearDown(self):
        db._connect = self.orig_connect
        try:
            self.db_path.unlink(missing_ok=True)
        except Exception:
            pass


class TestCoordinatorRouting(unittest.TestCase):
    """Section 1: Coordinator / Routing Agent Test Cases"""

    def test_route_escalation_angry_keywords(self):
        """Should route strongly frustrated customers to escalation with accurate reasons."""
        angry_cases = [
            ("I am very angry about this service!", "frustration"),
            ("This is unacceptable, I will take legal action", "legal"),
            ("I want to file a formal complaint", "complaint"),
            ("I am furious with this delay", "frustration"),
            ("Let me speak to a manager", "manager")
        ]
        for phrase, expected_substr in angry_cases:
            route, reason = _rule_route(phrase)
            self.assertEqual(route, "escalation", f"Failed route for phrase: {phrase}")
            self.assertIn(expected_substr, reason.lower(), f"Reason '{reason}' does not contain '{expected_substr}'")

    def test_route_escalation_asks_human(self):
        """Should route requests for human representatives to escalation."""
        human_phrases = [
            "I need to talk to a human",
            "Can I speak with a real person?",
            "Transfer me to a live agent",
            "Let me speak to a manager",
            "Connect me with a representative"
        ]
        for phrase in human_phrases:
            route, _ = _rule_route(phrase)
            self.assertEqual(route, "escalation", f"Failed for phrase: {phrase}")

    def test_route_account_inquiry(self):
        """Should route account / personal info requests to user agent."""
        account_phrases = [
            "What is my email?",
            "Show my account information",
            "Can I check my profile?",
            "What email is linked to my account?"
        ]
        for phrase in account_phrases:
            route, _ = _rule_route(phrase)
            self.assertEqual(route, "account", f"Failed for phrase: {phrase}")

    def test_route_track_existing_ticket(self):
        """Should route ticket status inquiries to track_ticket."""
        track_phrases = [
            "Track ticket #1",
            "What is the status of ticket 105?",
            "Show ticket status",
            "I want an update on my existing ticket",
            "Check ticket update for #42"
        ]
        for phrase in track_phrases:
            route, _ = _rule_route(phrase)
            self.assertEqual(route, "track_ticket", f"Failed for phrase: {phrase}")

    def test_route_create_support_ticket(self):
        """Should route bug reports and trouble tickets to create_ticket."""
        create_phrases = [
            "Please create a ticket for me",
            "I cannot login to the dashboard",
            "The checkout button is broken",
            "I keep getting an error message",
            "The search feature does not work",
            "I have an issue with billing"
        ]
        for phrase in create_phrases:
            route, _ = _rule_route(phrase)
            self.assertEqual(route, "create_ticket", f"Failed for phrase: {phrase}")

    def test_route_knowledge_fallback(self):
        """Should route general questions to knowledge agent (RAG)."""
        faq_phrases = [
            "How do I reset my password?",
            "What are your business hours?",
            "What is the refund policy?",
            "Tell me about subscription plans"
        ]
        for phrase in faq_phrases:
            route, _ = _rule_route(phrase)
            self.assertEqual(route, "knowledge", f"Failed for phrase: {phrase}")

    def test_coordinator_node_handles_audio_error(self):
        """Coordinator should do nothing if audio error was flagged."""
        state: SupportState = {"route": "audio_error", "message": "Failed audio"}
        res = coordinator(state)
        self.assertEqual(res, {})


class TestTicketAgent(BaseDatabaseTestCase):
    """Section 2: Ticket Agent Test Cases"""

    def test_ticket_agent_unauthorized_blocked(self):
        """Unauthenticated requests must be blocked with an authentication prompt."""
        state: SupportState = {
            "authorized": False,
            "customer_id": "",
            "message": "Create a ticket: login is broken",
            "route": "create_ticket"
        }
        result = ticket_agent(state)
        self.assertIn("sign in", result["response"].lower())
        self.assertNotIn("ticket_id", result)

    def test_create_ticket_authenticated(self):
        """Authenticated users can successfully create a ticket."""
        state: SupportState = {
            "authorized": True,
            "customer_id": "CUST-1001",
            "message": "Cannot log in, page gives 500 error",
            "route": "create_ticket"
        }
        result = ticket_agent(state)
        self.assertIn("ticket_id", result)
        self.assertIn("open", result["response"].lower())

        # Verify database record
        ticket = db.get_ticket(result["ticket_id"], "CUST-1001")
        self.assertIsNotNone(ticket)
        self.assertEqual(ticket["ticket_id"], result["ticket_id"])
        self.assertEqual(ticket["status"], "open")

    def test_track_ticket_by_id_success(self):
        """User can track their existing ticket by ID."""
        ticket_id = db.create_ticket("CUST-1001", "Billing inquiry", "Question about charge")
        state: SupportState = {
            "authorized": True,
            "customer_id": "CUST-1001",
            "message": f"Track ticket #{ticket_id}",
            "route": "track_ticket"
        }
        result = ticket_agent(state)
        self.assertEqual(result.get("ticket_id"), ticket_id)
        self.assertIn(str(ticket_id), result["response"])

    def test_track_ticket_other_user_isolation(self):
        """User cannot access tickets belonging to another customer."""
        ticket_id = db.create_ticket("CUST-1001", "Private issue", "Confidential")
        state: SupportState = {
            "authorized": True,
            "customer_id": "CUST-1002",  # Different customer
            "message": f"Track ticket #{ticket_id}",
            "route": "track_ticket"
        }
        result = ticket_agent(state)
        self.assertIn("couldn't find that ticket under your signed-in account", result["response"].lower())

    def test_track_ticket_list_all_recent(self):
        """When no ticket ID is provided, list all recent tickets for the customer."""
        state: SupportState = {
            "authorized": True,
            "customer_id": "CUST-1001",
            "message": "Show all my tickets",
            "route": "track_ticket"
        }
        result = ticket_agent(state)
        self.assertIn("recent tickets", result["response"].lower())


class TestUserAgent(unittest.TestCase):
    """Section 3: User / Account Agent Test Cases"""

    def test_user_agent_unauthorized_blocked(self):
        """Unauthorized requests for account data are blocked for privacy."""
        state: SupportState = {
            "authorized": False,
            "customer_id": "",
            "message": "What is my email?"
        }
        result = user_agent(state)
        self.assertIn("sign in", result["response"].lower())

    def test_user_agent_missing_profile_handled(self):
        """Handles edge case where customer is marked authorized but profile is missing."""
        state: SupportState = {
            "authorized": True,
            "customer_id": "CUST-1001",
            "verified_user": None,
            "message": "Show my account"
        }
        result = user_agent(state)
        self.assertIn("sign in again", result["response"].lower())

    def test_user_agent_returns_verified_details(self):
        """Returns verified name and email safely for authenticated users."""
        state: SupportState = {
            "authorized": True,
            "customer_id": "CUST-1001",
            "verified_user": {
                "customer_id": "CUST-1001",
                "display_name": "Alex Morgan",
                "email": "alex@example.com"
            },
            "message": "What is my registered email address?"
        }
        result = user_agent(state)
        self.assertIn("alex@example.com", result["response"])
        self.assertIn("Alex Morgan", result["response"])


class TestEscalationAgent(BaseDatabaseTestCase):
    """Section 4: Escalation Agent Test Cases"""

    def test_escalation_authenticated_user(self):
        """Escalation stores customer ID, reason, and context into SQLite."""
        state: SupportState = {
            "authorized": True,
            "customer_id": "CUST-1001",
            "message": "I want to speak with a human manager immediately",
            "intent_reason": "Customer requested a person",
            "conversation": "User: Hello\nBot: Hi\nUser: I need a human"
        }
        result = escalation_agent(state)
        self.assertTrue(result["escalation_required"])
        self.assertIn("escalation_id", result)
        self.assertIn(str(result["escalation_id"]), result["response"])

        # Check DB
        with self._get_test_conn() as conn:
            row = conn.execute("SELECT * FROM escalations WHERE escalation_id=?", (result["escalation_id"],)).fetchone()
            self.assertIsNotNone(row)
            self.assertEqual(row["customer_id"], "CUST-1001")
            self.assertEqual(row["status"], "pending_human")

    def test_escalation_unauthenticated_user(self):
        """Unauthenticated user can still be escalated without customer ID."""
        state: SupportState = {
            "authorized": False,
            "message": "I demand to talk to an agent right now",
            "intent_reason": "Customer expressed strong frustration."
        }
        result = escalation_agent(state)
        self.assertTrue(result["escalation_required"])
        self.assertIn("escalation_id", result)

        with self._get_test_conn() as conn:
            row = conn.execute("SELECT * FROM escalations WHERE escalation_id=?", (result["escalation_id"],)).fetchone()
            self.assertIsNone(row["customer_id"])

    def test_escalation_issue_specific_summary(self):
        """Handoff summary extracts core issue, category, urgency, and key entities."""
        state: SupportState = {
            "authorized": True,
            "customer_id": "CUST-1001",
            "verified_user": {"customer_id": "CUST-1001", "display_name": "Alex Morgan", "email": "alex@example.com"},
            "message": "I want to talk to an agent right now",
            "intent_reason": "Customer requested human agent.",
            "conversation": "User: I was charged $150 twice for Order #88219.\nAssistant: Chat cannot authorize refunds."
        }
        result = escalation_agent(state)
        with self._get_test_conn() as conn:
            row = conn.execute("SELECT * FROM escalations WHERE escalation_id=?", (result["escalation_id"],)).fetchone()
            summary = row["summary"]
            self.assertIn("Billing & Financial Dispute", summary)
            self.assertIn("HIGH", summary)
            self.assertIn("#88219", summary)
            self.assertIn("Order #88219", summary)
            self.assertIn("Recommended Action:", summary)


class TestKnowledgeAgent(unittest.TestCase):
    """Section 5: Knowledge Agent (RAG) Test Cases"""

    @patch("src.graph.retrieve")
    def test_knowledge_agent_hit(self, mock_retrieve):
        """When knowledge is found, returns context, sources, and answer."""
        mock_retrieve.return_value = (
            "To reset password, click 'Forgot Password' on the login screen.",
            ["company_faq.md"]
        )
        state: SupportState = {"message": "How do I reset my password?"}
        result = knowledge_agent(state)
        self.assertEqual(result["knowledge_sources"], ["company_faq.md"])
        self.assertIn("Forgot Password", result["response"])

    @patch("src.graph.retrieve")
    def test_knowledge_agent_miss_fallback(self, mock_retrieve):
        """When knowledge is not found, provides graceful fallback explanation."""
        mock_retrieve.return_value = ("", [])
        state: SupportState = {"message": "What is the orbital speed of Jupiter?"}
        result = knowledge_agent(state)
        self.assertEqual(result["knowledge_sources"], [])
        self.assertIn("couldn't find a reliable answer", result["response"].lower())

    def test_extract_best_paragraph_rejects_unrelated_query(self):
        """Out-of-domain queries like 'tell me 2*2?' should not return an unrelated FAQ paragraph."""
        from src.knowledge import extract_best_paragraph
        context = "20. How do I enable two-factor authentication (2FA)?\nGo to your Account Security settings and choose 'Enable Two-Factor Authentication'."
        self.assertEqual(extract_best_paragraph("tell me 2*2?", context), "")


class TestAudioProcessor(unittest.TestCase):
    """Section 6: Audio Processor Agent Test Cases"""

    @patch("src.graph.transcribe_audio")
    def test_audio_processor_success(self, mock_transcribe):
        """Valid audio is transcribed and populates message and transcript in state."""
        mock_transcribe.return_value = "I need help with my account"
        mock_file = MagicMock()
        state: SupportState = {"audio_file": mock_file}

        result = audio_processor(state)
        self.assertEqual(result["message"], "I need help with my account")
        self.assertEqual(result["transcript"], "I need help with my account")

    @patch("src.graph.transcribe_audio")
    def test_audio_processor_error_handling(self, mock_transcribe):
        """Transcription error sets audio_error route and informative message."""
        mock_transcribe.side_effect = RuntimeError("Corrupted audio data")
        mock_file = MagicMock()
        state: SupportState = {"audio_file": mock_file}

        result = audio_processor(state)
        self.assertEqual(result["route"], "audio_error")
        self.assertIn("couldn't transcribe", result["response"].lower())


class TestEndToEndWorkflow(BaseDatabaseTestCase):
    """Section 7: Full Multi-Agent LangGraph Orchestration Test Cases"""

    def test_e2e_ticket_creation_flow(self):
        """End-to-end: Authenticated user reports an issue -> coordinator -> ticket_agent -> ticket created."""
        customer = {"customer_id": "CUST-1001", "display_name": "Alex Morgan", "email": "alex@example.com"}
        result = handle_support_request(
            message="I cannot log in and keep getting an error message",
            customer=customer
        )
        self.assertEqual(result["route"], "create_ticket")
        self.assertIn("ticket_id", result)
        self.assertIn("open", result["response"].lower())

    def test_e2e_account_query_flow(self):
        """End-to-end: Authenticated user asks for email -> coordinator -> user_agent -> profile returned."""
        customer = {"customer_id": "CUST-1001", "display_name": "Alex Morgan", "email": "alex@example.com"}
        result = handle_support_request(
            message="What is my account email?",
            customer=customer
        )
        self.assertEqual(result["route"], "account")
        self.assertIn("alex@example.com", result["response"])

    def test_e2e_escalation_flow(self):
        """End-to-end: Angry customer -> coordinator -> escalation_agent -> escalation created."""
        result = handle_support_request(
            message="This is completely unacceptable! I want to talk to a human agent right now."
        )
        self.assertEqual(result["route"], "escalation")
        self.assertTrue(result["escalation_required"])
        self.assertIn("escalation_id", result)


class TestAuthenticationSecurity(BaseDatabaseTestCase):
    """Section 8: Database & Customer Authentication Security Test Cases"""

    def test_authenticate_valid_customer(self):
        """Valid customer ID and demo PIN successfully returns user profile."""
        user = db.authenticate_customer("CUST-1001", "2468")
        self.assertIsNotNone(user)
        self.assertEqual(user["customer_id"], "CUST-1001")
        self.assertEqual(user["display_name"], "Alex Morgan")

    def test_authenticate_invalid_pin(self):
        """Incorrect PIN fails authentication."""
        user = db.authenticate_customer("CUST-1001", "9999")
        self.assertIsNone(user)

    def test_authenticate_empty_input(self):
        """Empty ID or PIN safely fails."""
        self.assertIsNone(db.authenticate_customer("", "2468"))
        self.assertIsNone(db.authenticate_customer("CUST-1001", ""))

    def test_escalation_management_queries(self):
        """Test listing, filtering, and updating status of escalations."""
        esc_id = db.create_escalation("CUST-1001", "Frustrated user", "Summary 1", "Conv 1")
        all_esc = db.list_escalations()
        self.assertTrue(any(e["escalation_id"] == esc_id for e in all_esc))

        # Check pending filter
        pending = db.list_escalations(status="pending_human")
        self.assertTrue(any(e["escalation_id"] == esc_id for e in pending))

        # Update status
        success = db.update_escalation_status(esc_id, "resolved")
        self.assertTrue(success)

        # Verify updated status
        resolved = db.list_escalations(status="resolved")
        self.assertTrue(any(e["escalation_id"] == esc_id for e in resolved))

    def test_list_all_tickets(self):
        """Test listing tickets across all customers."""
        db.create_ticket("CUST-1001", "Issue A", "Desc A")
        db.create_ticket("CUST-1002", "Issue B", "Desc B")
        tickets = db.list_all_tickets()
        self.assertGreaterEqual(len(tickets), 2)
        customers = {t["customer_id"] for t in tickets}
        self.assertIn("CUST-1001", customers)
        self.assertIn("CUST-1002", customers)



if __name__ == "__main__":
    unittest.main()
