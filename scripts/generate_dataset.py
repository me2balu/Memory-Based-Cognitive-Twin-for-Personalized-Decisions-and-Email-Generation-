"""
generate_dataset.py
--------------------
Generates a synthetic email dataset for the Cognitive Twin experiment.

v2 — LLM-generated responses (Groq API)
========================================
Instead of combining hardcoded phrase fragments (which produces near-identical
responses for every email), this version calls the LLM for each row to write a
genuine, topic-specific reply in the persona's voice.

Design:
  - 5 users, each with a distinct communication persona.
  - 120 emails per user (600 total) → enough data for meaningful ML training.
  - Decisions are content-driven (not random) so the ML task is meaningful.
  - User responses are LLM-generated: varied, specific, and authentically styled.
  - Falls back to template responses if the API is unavailable.

Output: data/raw/emails.csv

Usage:
    python scripts/generate_dataset.py                # full LLM generation
    python scripts/generate_dataset.py --template     # fast template fallback
    python scripts/generate_dataset.py --n 60         # 60 emails/user (quick test)
"""

import random
import csv
import pathlib
import argparse
import sys
import time
from datetime import datetime, timedelta

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import os
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_MODEL   = os.environ.get("LLM_MODEL", "llama-3.3-70b-versatile")

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

ROOT        = pathlib.Path(__file__).parent.parent
OUTPUT_PATH = ROOT / "data" / "raw" / "emails.csv"

# ---------------------------------------------------------------------------
# Persona definitions
# ---------------------------------------------------------------------------
PERSONAS = {
    "alice": {
        "style_desc": (
            "Alice writes very formal, polished professional emails. "
            "She always opens with 'Dear [Name],' and closes with 'Best regards,\\nAlice' or 'Kind regards,\\nAlice'. "
            "She never uses contractions. She uses vocabulary like 'I am pleased', 'Furthermore', 'Kindly', 'I would like to inform you'. "
            "Sentences are complete and grammatically perfect. 3-5 sentences total."
        ),
        "greetings": ["Dear {name},", "Hello {name},"],
        "sign_offs":  ["Best regards,\nAlice", "Kind regards,\nAlice", "Sincerely,\nAlice"],
        "accept_phrase": ["I am pleased to confirm", "I would be happy to", "Certainly, I can accommodate"],
        "reject_phrase": ["I regret to inform you", "Unfortunately, I must decline", "I am unable to proceed"],
        "defer_phrase":  ["Let us revisit this at a later date", "I propose we defer this matter"],
        "maybe_phrase":  ["I shall consider this", "This requires further thought"],
        "info_phrase":   ["Could you kindly provide additional details", "I would require further clarification"],
        "connectors":    ["Furthermore,", "In addition,", "Please note that", "I would like to inform you that"],
        "style": "formal",
    },
    "bob": {
        "style_desc": (
            "Bob writes casual, laid-back emails like he's texting a friend. "
            "He opens with 'Hey [Name],' or just dives in. Uses contractions everywhere: 'I'd', 'can't', 'won't'. "
            "Uses slang: 'yeah', 'nope', 'totally', 'sounds good', 'no worries', 'tbh', 'btw'. "
            "Short sentences. Sometimes incomplete. Closes with 'Cheers, Bob' or 'Later, Bob'. 2-3 sentences max."
        ),
        "greetings": ["Hey {name},", "Hi {name},", "Yo,"],
        "sign_offs":  ["Cheers,\nBob", "Later,\nBob", "Thanks,\nBob"],
        "accept_phrase": ["Sure, works for me", "Yeah, sounds good", "Totally fine with that"],
        "reject_phrase": ["Nope, can't do that", "Sorry, that doesn't work", "No can do"],
        "defer_phrase":  ["Let's table this for now", "Can we deal with this later?"],
        "maybe_phrase":  ["Maybe, let me think", "Not sure yet"],
        "info_phrase":   ["What's the deal with", "Need more info on"],
        "connectors":    ["Also,", "btw,", "FYI,"],
        "style": "casual",
    },
    "carol": {
        "style_desc": (
            "Carol writes warm, enthusiastic, friendly emails — she genuinely loves people. "
            "Opens with 'Hi [Name]! Hope you're doing well!' with exclamation marks. "
            "Uses phrases like 'That sounds wonderful!', 'I'd absolutely love to!', 'How exciting!', 'Can't wait!'. "
            "Asks a friendly follow-up question. Closes with 'Warmly, Carol' or 'Talk soon! Carol'. 3-4 sentences."
        ),
        "greetings": ["Hi {name}! Hope you're doing well!", "Hello {name}! Great to hear from you!"],
        "sign_offs":  ["Warmly,\nCarol", "Talk soon!\nCarol", "Take care!\nCarol"],
        "accept_phrase": ["I'd love to", "That sounds wonderful", "Absolutely, I'm in"],
        "reject_phrase": ["I'm so sorry, but I can't", "I really wish I could, but"],
        "defer_phrase":  ["Can we come back to this soon?", "I'd love to revisit this"],
        "maybe_phrase":  ["Hmm, I'm not sure yet!", "Let me think it over"],
        "info_phrase":   ["Could you share a bit more about", "I'd love to know more about"],
        "connectors":    ["Also,", "On top of that,", "I'm excited to say that"],
        "style": "friendly",
    },
    "dave": {
        "style_desc": (
            "Dave writes the shortest possible emails. Often NO greeting. "
            "1-3 sentences MAXIMUM, often just 1. "
            "Accept='Yes.' or 'Fine.' or 'Works for me.' Reject='No.' or 'Declined.' "
            "No filler words, no pleasantries. Closes with '-Dave' or nothing."
        ),
        "greetings": ["{name} -", ""],
        "sign_offs":  ["-Dave", "Dave", ""],
        "accept_phrase": ["Yes.", "Agreed.", "Fine."],
        "reject_phrase": ["No.", "Declined.", "Can't."],
        "defer_phrase":  ["Later.", "Not now."],
        "maybe_phrase":  ["TBD.", "Possibly."],
        "info_phrase":   ["Need details.", "More info needed."],
        "connectors":    ["Note:", "Also:"],
        "style": "concise",
    },
    "eve": {
        "style_desc": (
            "Eve writes structured, analytical emails — everything is evidence-based. "
            "Opens with 'Dear [Name],' or 'Hello [Name],'. "
            "References data/metrics: 'Based on the available data...', 'The risk-benefit analysis indicates...', 'From an analytical standpoint...'. "
            "Uses conditional language: 'Subject to...', 'Contingent on...'. "
            "May use a brief numbered list. Closes with 'Best, Eve' or 'Regards, Eve'. 3-5 sentences."
        ),
        "greetings": ["Hello {name},", "Dear {name},"],
        "sign_offs":  ["Best,\nEve", "Regards,\nEve", "Analytically yours,\nEve"],
        "accept_phrase": ["Based on the available data, I concur", "The evidence supports proceeding", "Analysis indicates this is viable"],
        "reject_phrase": ["The risk-benefit analysis does not support this", "Based on the data, this is not advisable"],
        "defer_phrase":  ["I recommend deferring pending additional data", "This decision should be postponed for analysis"],
        "maybe_phrase":  ["Further analysis is required", "The data is inconclusive at this stage"],
        "info_phrase":   ["Please provide quantitative details on", "I require the following data points:"],
        "connectors":    ["Furthermore,", "From an analytical standpoint,", "Considering the data,"],
        "style": "analytical",
    },
}

# ---------------------------------------------------------------------------
# Email templates
# ---------------------------------------------------------------------------
EMAIL_TEMPLATES = {
    "ACCEPT": [
        ("Meeting Request",        "Hi, could we schedule a {topic} meeting {time}? It should only take about {duration} minutes."),
        ("Invitation",             "You are invited to join our {topic} committee. Would you be available to participate?"),
        ("Collaboration Request",  "We'd love to collaborate with you on the {topic} project. Are you interested in partnering?"),
        ("Event Attendance",       "Please confirm your attendance at the {topic} event scheduled for {time}."),
        ("Task Assignment",        "Can you take on the {topic} task? It needs to be completed by {time}."),
        ("Webinar Registration",   "Would you like to register for the {topic} webinar happening {time}?"),
        ("Workshop Invite",        "We're hosting a {topic} workshop {time}. Would you be able to attend and present?"),
        ("Intro Call Request",     "Could we set up a 30-minute intro call about the {topic} initiative {time}?"),
    ],
    "REJECT": [
        ("Budget Request",         "We are requesting an additional budget of ${amount} for the {topic} initiative. Please approve."),
        ("Policy Change",          "We propose changing the {topic} policy to allow {change}. Do you approve this modification?"),
        ("Vendor Proposal",        "Our company would like to offer {topic} services at ${amount}/month. Please consider our proposal."),
        ("Scope Expansion",        "We want to expand the scope of {topic} to include {addition}. Can you sign off on this?"),
        ("Resource Request",       "Can we borrow {resource} for the {topic} project for the next two weeks?"),
        ("Outsourcing Proposal",   "We'd like to outsource the {topic} function to a third-party vendor. Do you agree?"),
        ("Deadline Extension Ask", "The {topic} team is requesting a 3-week deadline extension. Can you approve?"),
    ],
    "MAYBE": [
        ("Partnership Proposal",   "We'd like to explore a potential partnership around {topic}. What are your thoughts on this?"),
        ("Strategy Review",        "We're considering a new strategy for {topic}. Would you be open to discussing it next week?"),
        ("New Initiative",         "We're thinking of launching a {topic} initiative next quarter. Does this interest you?"),
        ("Optional Workshop",      "There's an optional {topic} workshop happening {time}. Would you like to join us?"),
        ("Pilot Program",          "We're piloting a new {topic} program {time}. Would you consider being an early adopter?"),
        ("Advisory Role",          "We'd love to have you as an advisor on our {topic} project. Is that something you'd consider?"),
    ],
    "DEFER": [
        ("Long-term Planning",     "We need to finalize the {topic} roadmap for next year. Can you review it and provide input?"),
        ("Contract Renewal",       "Your {topic} contract expires {time}. Shall we discuss renewal terms and conditions?"),
        ("Annual Review",          "It's time for the annual {topic} performance review. Can we schedule this for {time}?"),
        ("Policy Update",          "The {topic} policy needs updating. Can we discuss the revisions next quarter?"),
        ("Backlog Grooming",       "We have 20 items in the {topic} backlog to prioritize. Can we do this in the next sprint?"),
        ("Quarterly Planning",     "Q4 {topic} planning session needs to be scheduled. Can you block time {time}?"),
    ],
    "REQUEST_MORE_INFORMATION": [
        ("Unclear Proposal",       "We'd like to move forward with {topic}. Please confirm your availability and interest."),
        ("Vague Request",          "Can you help us with {topic}? It would be greatly appreciated if you could assist."),
        ("New Contact",            "Hi, I'm {name} from {company}. I wanted to connect regarding {topic}."),
        ("Technical Question",     "We're experiencing issues with {topic}. Could you advise on the best approach?"),
        ("Ambiguous Follow-up",    "Following up on our previous discussion about {topic}. What's the current status?"),
        ("Unclear Scope",          "The {topic} project brief mentions several deliverables. Can you clarify what's in scope?"),
        ("Missing Context",        "We received your proposal for {topic} but several key details are missing. Can you elaborate?"),
    ],
}

TOPICS = [
    "Q3 planning", "Q4 budget", "product launch", "marketing strategy", "budget allocation",
    "team building", "data analysis", "user research", "API integration",
    "performance review", "client onboarding", "security audit", "sprint planning",
    "stakeholder alignment", "roadmap prioritization", "vendor evaluation",
    "hiring pipeline", "customer feedback", "technical documentation", "compliance review",
    "cost optimization", "infrastructure migration", "brand refresh", "sales enablement",
    "employee onboarding", "data privacy", "AI governance", "platform scalability",
    "mobile app redesign", "partner integration",
]
TIMES      = ["next Monday", "this Friday", "next week", "tomorrow", "end of month",
               "next quarter", "Tuesday at 3pm", "Thursday morning", "next sprint", "end of Q4"]
AMOUNTS    = ["5,000", "10,000", "25,000", "50,000", "2,500", "15,000", "75,000"]
RESOURCES  = ["the conference room", "two developers", "the design team", "server access",
               "the testing environment", "the data team for one sprint"]
ADDITIONS  = ["mobile support", "international markets", "additional features",
               "a new reporting module", "real-time analytics", "SSO integration"]
CHANGES    = ["flexible deadlines", "remote-first policies", "optional attendance",
               "self-paced learning", "async stand-ups"]
COMPANIES  = ["TechCorp", "DataSolutions", "InnovateLtd", "NextGen Systems", "CloudBase Inc",
               "Momentum AI", "Apex Analytics", "Vertex Partners"]
NAMES_POOL = ["Alex", "Jordan", "Morgan", "Taylor", "Sam", "Riley", "Casey", "Quinn",
               "Drew", "Avery", "Blake", "Harper"]


def fill_template(s: str) -> str:
    return (s
        .replace("{topic}",    random.choice(TOPICS))
        .replace("{time}",     random.choice(TIMES))
        .replace("{duration}", str(random.choice([15, 30, 45, 60])))
        .replace("{amount}",   random.choice(AMOUNTS))
        .replace("{resource}", random.choice(RESOURCES))
        .replace("{addition}", random.choice(ADDITIONS))
        .replace("{change}",   random.choice(CHANGES))
        .replace("{name}",     random.choice(NAMES_POOL))
        .replace("{company}",  random.choice(COMPANIES))
    )


# ---------------------------------------------------------------------------
# Template fallback (fast, no API needed)
# ---------------------------------------------------------------------------
def _template_response(persona_key: str, decision: str) -> str:
    p = PERSONAS[persona_key]
    sender  = random.choice(NAMES_POOL)
    greeting = random.choice(p["greetings"]).format(name=sender)
    sign_off = random.choice(p["sign_offs"])
    connector = random.choice(p["connectors"])

    phrase_map = {
        "ACCEPT":                   ("accept_phrase",  " with your request."),
        "REJECT":                   ("reject_phrase",  " on this occasion."),
        "MAYBE":                    ("maybe_phrase",   " before giving a final answer."),
        "DEFER":                    ("defer_phrase",   " until we have more clarity."),
        "REQUEST_MORE_INFORMATION": ("info_phrase",    " before I can proceed."),
    }
    key, suffix = phrase_map.get(decision, ("accept_phrase", "."))
    core = random.choice(p[key]) + suffix

    style = p["style"]
    if style == "concise":
        body = core
    elif style == "analytical":
        body = f"{core} {connector} I will review all relevant parameters."
    elif style == "friendly":
        body = f"{core} {connector} let me know if you need anything else!"
    else:
        body = f"{core} {connector} please do not hesitate to reach out."

    parts = [x for x in [greeting, "", body, "", sign_off] if x != ""]
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# LLM-generated response (rich, varied, specific)
# ---------------------------------------------------------------------------
def _llm_response(persona_key: str, decision: str, incoming_email: str,
                  client, model: str, retries: int = 3) -> str:
    p = PERSONAS[persona_key]
    prompt = f"""{p['style_desc']}

The person ({persona_key}) just received this email:
---
{incoming_email.strip()}
---

Their decision is: {decision}

Write their reply in character. Address the specific topic in the email. 
Write ONLY the email reply — no labels, no preamble, no meta-commentary."""

    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                messages=[
                    {"role": "system", "content":
                        "You are a dataset generation assistant. Write realistic, varied email replies "
                        "that are specific to the email content. Never repeat generic phrases across emails."},
                    {"role": "user", "content": prompt},
                ],
                model=model,
                temperature=0.9 + attempt * 0.05,
                max_tokens=300,
            )
            msg = resp.choices[0].message
            content = (msg.content or "").strip()
            if not content:
                content = (getattr(msg, "reasoning_content", None) or "").strip()
            if content:
                return content
        except Exception as e:
            print(f"    [warn] LLM call failed (attempt {attempt+1}/{retries}): {e}")
            time.sleep(2 ** attempt)   # exponential back-off

    # Fallback to template if all LLM attempts fail
    print(f"    [warn] Falling back to template for {persona_key}/{decision}")
    return _template_response(persona_key, decision)


# ---------------------------------------------------------------------------
# Dataset generator
# ---------------------------------------------------------------------------
def generate_dataset(n_per_user: int = 120, use_llm: bool = True) -> list:
    client = None
    if use_llm:
        if not LLM_API_KEY:
            print("[warn] LLM_API_KEY not set — falling back to template mode.")
            use_llm = False
        else:
            try:
                from groq import Groq
                client = Groq(api_key=LLM_API_KEY)
                print(f"[OK] LLM generation enabled (model={LLM_MODEL})")
            except ImportError:
                print("[warn] groq not installed — falling back to template mode.")
                use_llm = False

    records    = []
    base_date  = datetime(2024, 1, 2)
    total      = len(PERSONAS) * n_per_user
    done       = 0

    for user_id in PERSONAS:
        current_date = base_date

        # Build a balanced decision list
        decisions_list = (
            ["ACCEPT"]                   * int(n_per_user * 0.30) +
            ["REJECT"]                   * int(n_per_user * 0.20) +
            ["MAYBE"]                    * int(n_per_user * 0.20) +
            ["DEFER"]                    * int(n_per_user * 0.15) +
            ["REQUEST_MORE_INFORMATION"] * int(n_per_user * 0.15)
        )
        while len(decisions_list) < n_per_user:
            decisions_list.append("ACCEPT")
        decisions_list = decisions_list[:n_per_user]
        random.shuffle(decisions_list)

        print(f"\n[{user_id}] Generating {n_per_user} emails …")

        for i, decision in enumerate(decisions_list):
            tmpl_subject, tmpl_body = random.choice(EMAIL_TEMPLATES[decision])
            subject        = fill_template(tmpl_subject)
            body           = fill_template(tmpl_body)
            incoming_email = f"Subject: {subject}\n\n{body}"

            if use_llm:
                user_response = _llm_response(
                    user_id, decision, incoming_email, client, LLM_MODEL
                )
            else:
                user_response = _template_response(user_id, decision)

            # Spread emails 2–5 business days apart
            current_date += timedelta(days=random.randint(2, 5))
            while current_date.weekday() >= 5:
                current_date += timedelta(days=1)

            records.append({
                "record_id":    f"{user_id}_{i:04d}",
                "user_id":      user_id,
                "timestamp":    current_date.strftime("%Y-%m-%d %H:%M:%S"),
                "email_subject": subject,
                "incoming_email": incoming_email,
                "user_response": user_response,
                "decision":     decision,
            })

            done += 1
            if use_llm:
                # Rate limit: Groq free tier ~30 req/min — add small delay
                time.sleep(0.5)
                if (i + 1) % 10 == 0:
                    print(f"  … {done}/{total} done")

        print(f"  [{user_id}] ✓ {n_per_user} emails generated")

    records.sort(key=lambda r: r["timestamp"])
    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate Cognitive Twin email dataset")
    parser.add_argument("--template", action="store_true",
                        help="Use fast template mode (no API calls)")
    parser.add_argument("--n", type=int, default=120,
                        help="Emails per user (default 120)")
    args = parser.parse_args()

    use_llm = not args.template
    mode_label = "TEMPLATE (fast)" if not use_llm else f"LLM ({LLM_MODEL})"
    print(f"=== Cognitive Twin Dataset Generator v2 ===")
    print(f"Mode        : {mode_label}")
    print(f"Emails/user : {args.n}  ×  {len(PERSONAS)} users  =  {args.n * len(PERSONAS)} total")
    print()

    records = generate_dataset(n_per_user=args.n, use_llm=use_llm)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["record_id", "user_id", "timestamp", "email_subject",
                  "incoming_email", "user_response", "decision"]

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"\n[OK] Dataset written to: {OUTPUT_PATH}")
    print(f"   Total records : {len(records)}")

    from collections import Counter
    dist = Counter(r["decision"] for r in records)
    print(f"   Decision dist : {dict(dist)}")
    print()
    print("Next step: delete cached embeddings and re-run the experiment.")
    print("   del data\\raw\\*.npy  (or remove *.npy files in project root)")
    print("   python research_experiment.py")


if __name__ == "__main__":
    main()
