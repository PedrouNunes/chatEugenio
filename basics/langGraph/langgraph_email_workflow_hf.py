import os
from typing import Any, Dict, List, Optional, Literal

from dotenv import load_dotenv
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace


class EmailState(TypedDict):
    email: Dict[str, Any]
    email_category: Optional[str]
    spam_reason: Optional[str]
    is_spam: Optional[bool]
    email_draft: Optional[str]
    messages: List[Dict[str, Any]]


def build_model() -> ChatHuggingFace:
    load_dotenv()

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN não encontrado. Verifique seu arquivo .env.")

    llm = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-Coder-32B-Instruct",
        task="text-generation",
        huggingfacehub_api_token=hf_token,
        temperature=0.0,
        max_new_tokens=512,
    )

    return ChatHuggingFace(llm=llm)


model = build_model()


def read_email(state: EmailState) -> dict:
    email = state["email"]

    print(
        f"Alfred está processando um e-mail de {email['sender']} "
        f"com assunto: {email['subject']}"
    )

    return {}


def classify_email(state: EmailState) -> dict:
    email = state["email"]

    prompt = f"""
You are Alfred, a careful butler assistant.

Analyze the following email and return your answer in this exact format:

IS_SPAM: yes or no
CATEGORY: inquiry, complaint, thank_you, request, information, or other
SPAM_REASON: short reason or none

Email:
From: {email['sender']}
Subject: {email['subject']}
Body: {email['body']}

Rules:
- Mark as spam if it is clearly fraudulent, manipulative, mass-marketing, scam-like, or requests sensitive financial/personal data in a suspicious way.
- If it is legitimate, set IS_SPAM to no and classify the category.
- Keep the output concise and structured exactly as requested.
""".strip()

    messages = [HumanMessage(content=prompt)]
    response = model.invoke(messages)

    response_text = response.content.strip()
    response_text_lower = response_text.lower()

    is_spam = "is_spam: yes" in response_text_lower
    spam_reason = None
    email_category = None

    for line in response_text.splitlines():
        normalized = line.strip()

        if normalized.lower().startswith("category:"):
            email_category = normalized.split(":", 1)[1].strip().lower()

        if normalized.lower().startswith("spam_reason:"):
            reason = normalized.split(":", 1)[1].strip()
            if reason.lower() != "none":
                spam_reason = reason

    if is_spam and not spam_reason:
        spam_reason = "Modelo classificou o conteúdo como spam, mas não detalhou a razão."

    if not is_spam and not email_category:
        email_category = "other"

    new_messages = state.get("messages", []) + [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response.content},
    ]

    return {
        "is_spam": is_spam,
        "spam_reason": spam_reason,
        "email_category": email_category,
        "messages": new_messages,
    }


def handle_spam(state: EmailState) -> dict:
    print("\n[SPAM DETECTADO]")
    print(f"Motivo: {state['spam_reason']}")
    print("O e-mail foi movido para a pasta de spam.\n")
    return {}


def draft_response(state: EmailState) -> dict:
    email = state["email"]
    category = state["email_category"] or "other"

    prompt = f"""
You are Alfred, a polite and professional butler assistant.

Draft a short preliminary email reply for Mr. Wayne to review.

Original email:
From: {email['sender']}
Subject: {email['subject']}
Body: {email['body']}

Category: {category}

Requirements:
- Be polite and professional.
- Keep it concise.
- Do not overpromise.
- Write only the email draft body.
""".strip()

    messages = [HumanMessage(content=prompt)]
    response = model.invoke(messages)

    new_messages = state.get("messages", []) + [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response.content},
    ]

    return {
        "email_draft": response.content,
        "messages": new_messages,
    }


def notify_mr_wayne(state: EmailState) -> dict:
    email = state["email"]

    print("\n" + "=" * 60)
    print(f"Sir, you received an email from: {email['sender']}")
    print(f"Subject: {email['subject']}")
    print(f"Category: {state['email_category']}")
    print("\nI prepared a preliminary reply for your review:\n")
    print(state["email_draft"])
    print("=" * 60 + "\n")

    return {}


def route_email(state: EmailState) -> Literal["spam", "legitimate"]:
    if state["is_spam"]:
        return "spam"
    return "legitimate"


def build_graph():
    email_graph = StateGraph(EmailState)

    email_graph.add_node("read_email", read_email)
    email_graph.add_node("classify_email", classify_email)
    email_graph.add_node("handle_spam", handle_spam)
    email_graph.add_node("draft_response", draft_response)
    email_graph.add_node("notify_mr_wayne", notify_mr_wayne)

    email_graph.add_edge(START, "read_email")
    email_graph.add_edge("read_email", "classify_email")

    email_graph.add_conditional_edges(
        "classify_email",
        route_email,
        {
            "spam": "handle_spam",
            "legitimate": "draft_response",
        },
    )

    email_graph.add_edge("handle_spam", END)
    email_graph.add_edge("draft_response", "notify_mr_wayne")
    email_graph.add_edge("notify_mr_wayne", END)

    return email_graph.compile()


def get_initial_state(email: Dict[str, Any]) -> EmailState:
    return {
        "email": email,
        "is_spam": None,
        "spam_reason": None,
        "email_category": None,
        "email_draft": None,
        "messages": [],
    }


def main() -> None:
    compiled_graph = build_graph()

    legitimate_email = {
        "sender": "john.smith@example.com",
        "subject": "Question about your consulting services",
        "body": (
            "Dear Mr. Wayne, I was referred to you by a colleague and would like "
            "to learn more about your consulting services. Would it be possible "
            "to schedule a call next week? Best regards, John Smith"
        ),
    }

    spam_email = {
        "sender": "winner@lottery-intl.com",
        "subject": "YOU HAVE WON $5,000,000!!!",
        "body": (
            "CONGRATULATIONS! You have been selected as the winner of our "
            "international lottery! To claim your prize, send your bank details "
            "and a processing fee immediately."
        ),
    }

    print("\nProcessando e-mail legítimo...\n")
    legitimate_result = compiled_graph.invoke(get_initial_state(legitimate_email))

    print("=== ESTADO FINAL DO E-MAIL LEGÍTIMO ===")
    print(legitimate_result)

    print("\nProcessando e-mail spam...\n")
    spam_result = compiled_graph.invoke(get_initial_state(spam_email))

    print("=== ESTADO FINAL DO E-MAIL SPAM ===")
    print(spam_result)


if __name__ == "__main__":
    main()