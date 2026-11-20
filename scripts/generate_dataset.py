"""
generate_dataset.py
--------------------
Generates a synthetic email dataset for the Cognitive Twin experiment.

Design:
- 5 users, each with a distinct communication persona.
- ~60 emails per user, chronologically ordered from 2024-01-01.
- Decisions are content-driven (not random) so the ML task is meaningful.
- User responses reflect each persona's style (formal/casual/concise/friendly/analytical).

Output: data/raw/emails.csv
"""

import random
import csv
import pathlib
from datetime import datetime, timedelta

# -- Fixed seed for reproducibility ------------------------------------------
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# -- Paths --------------------------------------------------------------------
ROOT = pathlib.Path(__file__).parent.parent
OUTPUT_PATH = ROOT / "data" / "raw" / "emails.csv"

# -- User personas -------------------------------------------------------------
# Each persona has a name, style description, and vocabulary hints used when
# crafting synthetic user_responses.  The LLM must infer style from memories,
# NOT from user_id directly.
PERSONAS = {
    "alice": {
        "style": "formal",
        "greetings": ["Dear {name},", "Hello {name},"],
        "sign_offs": ["Best regards,\nAlice", "Sincerely,\nAlice", "Kind regards,\nAlice"],
        "connectors": ["Furthermore,", "In addition,", "Please note that", "I would like to inform you that"],
        "accept_phrase": ["I am pleased to confirm", "I would be happy to", "Certainly, I can accommodate"],
        "reject_phrase": ["I regret to inform you", "Unfortunately, I must decline", "I am unable to proceed"],
        "maybe_phrase": ["I shall consider this", "This requires further thought", "I will review and revert"],
        "defer_phrase": ["Let us revisit this at a later date", "I propose we defer this matter"],
        "info_phrase": ["Could you kindly provide additional details", "I would require further clarification"],
    },
    "bob": {
        "style": "casual",
        "greetings": ["Hey {name},", "Hi {name},", "Yo {name},"],
        "sign_offs": ["Cheers,\nBob", "Thanks,\nBob", "Later,\nBob"],
        "connectors": ["Also,", "btw,", "FYI,", "Just so you know,"],
        "accept_phrase": ["Sure, works for me", "Yeah, sounds good", "Totally fine with that"],
        "reject_phrase": ["Nope, can't do that", "Sorry, that doesn't work", "No can do"],
        "maybe_phrase": ["Maybe, let me think", "Not sure yet", "Could go either way"],
        "defer_phrase": ["Let's table this for now", "Can we deal with this later?"],
        "info_phrase": ["What's the deal with", "Can you explain", "Need more info on"],
    },
    "carol": {
        "style": "friendly",
        "greetings": ["Hi {name}! Hope you're doing well!", "Hello {name}! Great to hear from you!"],
        "sign_offs": ["Warmly,\nCarol", "Take care!\nCarol", "Talk soon!\nCarol"],
        "connectors": ["Also,", "On top of that,", "I'm excited to say that", "Great news —"],
        "accept_phrase": ["I'd love to", "That sounds wonderful", "Absolutely, I'm in"],
        "reject_phrase": ["I'm so sorry, but I can't", "I really wish I could, but", "Unfortunately I'll have to pass"],
        "maybe_phrase": ["Hmm, I'm not sure yet!", "Let me think it over", "Maybe! I'll get back to you"],
        "defer_phrase": ["Can we come back to this soon?", "I'd love to revisit this another time"],
        "info_phrase": ["Could you share a bit more about", "I'd love to know more about"],
    },
    "dave": {
        "style": "concise",
        "greetings": ["{name} -", "Hi,", ""],
        "sign_offs": ["-Dave", "Dave", "Regards, Dave"],
        "connectors": ["Note:", "Also:", "FYI:"],
        "accept_phrase": ["Yes.", "Agreed.", "Fine."],
        "reject_phrase": ["No.", "Declined.", "Can't."],
        "maybe_phrase": ["Uncertain.", "TBD.", "Possibly."],
        "defer_phrase": ["Defer.", "Later.", "Not now."],
        "info_phrase": ["Need details.", "What's the context?", "More info needed."],
    },
    "eve": {
        "style": "analytical",
        "greetings": ["Hello {name},", "Dear {name},"],
        "sign_offs": ["Best,\nEve", "Regards,\nEve", "Analytically yours,\nEve"],
        "connectors": ["Furthermore,", "It should be noted that", "From an analytical standpoint,", "Considering the data,"],
        "accept_phrase": ["Based on the available information, I concur", "The evidence supports proceeding", "Analysis indicates this is viable"],
        "reject_phrase": ["The risk-benefit analysis does not support this", "Based on the data, this is not advisable", "This does not align with the metrics"],
        "maybe_phrase": ["Further analysis is required", "The data is inconclusive at this stage", "A conditional acceptance depends on"],
        "defer_phrase": ["I recommend deferring pending additional data", "This decision should be postponed for analysis"],
        "info_phrase": ["Please provide quantitative details on", "What are the key metrics for", "I require the following data points:"],
    },
}

# -- Email templates by decision type -----------------------------------------
EMAIL_TEMPLATES = {
    "ACCEPT": [
        ("Meeting Request", "Hi, could we schedule a {topic} meeting {time}? It should only take about {duration} minutes.", "confirm"),
        ("Invitation", "You are invited to join our {topic} committee. Would you be available to participate?", "join"),
        ("Collaboration Request", "We'd love to collaborate with you on the {topic} project. Are you interested?", "collaborate"),
        ("Event Attendance", "Please confirm your attendance at the {topic} event scheduled for {time}.", "attend"),
        ("Task Assignment", "Can you take on the {topic} task? It needs to be done by {time}.", "accept_task"),
        ("Webinar Registration", "Would you like to register for the {topic} webinar on {time}?", "register"),
    ],
    "REJECT": [
        ("Budget Request", "We are requesting an additional budget of \${amount} for the {topic} initiative.", "budget"),
        ("Policy Change", "We propose changing the {topic} policy to allow {change}. Do you approve?", "policy"),
        ("Vendor Proposal", "Our company would like to offer {topic} services at \${amount}/month.", "vendor"),
        ("Scope Expansion", "We want to expand the scope of {topic} to include {addition}. Please approve.", "scope"),
        ("Resource Request", "Can we borrow {resource} for the {topic} project?", "resource"),
    ],
    "MAYBE": [
        ("Partnership Proposal", "We'd like to explore a potential partnership around {topic}. What are your thoughts?", "partnership"),
        ("Strategy Review", "We're considering a new strategy for {topic}. Would you be open to discussing it?", "strategy"),
        ("New Initiative", "We're thinking of launching a {topic} initiative. Does this interest you?", "initiative"),
        ("Optional Workshop", "There's an optional {topic} workshop happening {time}. Would you like to join?", "workshop"),
    ],
    "DEFER": [
        ("Long-term Planning", "We need to finalize the {topic} roadmap for next year. Can you review it?", "roadmap"),
        ("Contract Renewal", "Your {topic} contract expires {time}. Shall we discuss renewal terms?", "contract"),
        ("Annual Review", "It's time for the annual {topic} review. Can we schedule this?", "review"),
        ("Policy Update", "The {topic} policy needs updating. Can we discuss revisions next quarter?", "policy_update"),
    ],
    "REQUEST_MORE_INFORMATION": [
        ("Unclear Proposal", "We'd like to move forward with {topic}. Please confirm your availability.", "unclear"),
        ("Vague Request", "Can you help us with {topic}? It would be greatly appreciated.", "vague"),
        ("New Contact", "Hi, I'm {name} from {company}. I wanted to connect regarding {topic}.", "new_contact"),
        ("Technical Question", "We're experiencing issues with {topic}. Could you advise?", "tech_question"),
        ("Ambiguous Follow-up", "Following up on our previous discussion about {topic}.", "followup"),
    ],
}

TOPICS = [
    "Q3 planning", "product launch", "marketing strategy", "budget allocation",
    "team building", "data analysis", "user research", "API integration",
    "performance review", "client onboarding", "security audit", "sprint planning",
    "stakeholder alignment", "roadmap prioritization", "vendor evaluation",
    "hiring pipeline", "customer feedback", "technical documentation", "compliance review",
    "cost optimization",
]

TIMES = [
    "next Monday", "this Friday", "next week", "tomorrow", "end of month",
    "next quarter", "next Tuesday at 3pm", "this Thursday morning", "next sprint",
]

AMOUNTS = ["5,000", "10,000", "25,000", "50,000", "2,500", "15,000"]
RESOURCES = ["the conference room", "two developers", "the design team", "server access", "the testing environment"]
ADDITIONS = ["mobile support", "international markets", "additional features", "a new reporting module"]
CHANGES = ["flexible deadlines", "remote-first policies", "optional attendance", "self-paced learning"]
COMPANIES = ["TechCorp", "DataSolutions", "InnovateLtd", "NextGen Systems", "CloudBase Inc"]
NAMES_POOL = ["Alex", "Jordan", "Morgan", "Taylor", "Sam", "Riley", "Casey", "Quinn"]


def fill_template(template_str: str) -> str:
    """Replace placeholder tokens in an email template."""
    return (
        template_str
        .replace("{topic}", random.choice(TOPICS))
        .replace("{time}", random.choice(TIMES))
        .replace("{duration}", str(random.choice([15, 30, 45, 60])))
        .replace("{amount}", random.choice(AMOUNTS))
        .replace("{resource}", random.choice(RESOURCES))
        .replace("{addition}", random.choice(ADDITIONS))
        .replace("{change}", random.choice(CHANGES))
        .replace("{name}", random.choice(NAMES_POOL))
        .replace("{company}", random.choice(COMPANIES))
    )


def generate_user_response(persona_key: str, decision: str, incoming_email: str) -> str:
    """
    Craft a synthetic user response that reflects the persona's style.
    The response is deterministic given the decision and persona, making it
    a realistic proxy for how each user communicates.
    """
    p = PERSONAS[persona_key]
    sender = random.choice(NAMES_POOL)
    greeting = random.choice(p["greetings"]).format(name=sender)
    sign_off = random.choice(p["sign_offs"])
    connector = random.choice(p["connectors"])

    if decision == "ACCEPT":
        core = random.choice(p["accept_phrase"]) + " with your request."
    elif decision == "REJECT":
        core = random.choice(p["reject_phrase"]) + " on this occasion."
    elif decision == "MAYBE":
        core = random.choice(p["maybe_phrase"]) + " before giving a final answer."
    elif decision == "DEFER":
        core = random.choice(p["defer_phrase"]) + " until we have more clarity."
    else:  # REQUEST_MORE_INFORMATION
        core = random.choice(p["info_phrase"]) + " before I can proceed."

    # Concise persona keeps it very short
    if p["style"] == "concise":
        body = core
    elif p["style"] == "analytical":
        body = f"{core} {connector} I will review all relevant parameters."
    elif p["style"] == "friendly":
        body = f"{core} {connector} let me know if you need anything else!"
    else:
        body = f"{core} {connector} please do not hesitate to reach out."

    parts = [x for x in [greeting, "", body, "", sign_off] if x != ""]
    return "\n".join(parts)


def generate_dataset(n_per_user: int = 60) -> list[dict]:
    """Generate the full synthetic email dataset."""
    records = []
    base_date = datetime(2024, 1, 2)  # Tuesday

    for user_id in PERSONAS:
        current_date = base_date
        decisions_list = (
            ["ACCEPT"] * int(n_per_user * 0.30)
            + ["REJECT"] * int(n_per_user * 0.20)
            + ["MAYBE"] * int(n_per_user * 0.20)
            + ["DEFER"] * int(n_per_user * 0.15)
            + ["REQUEST_MORE_INFORMATION"] * int(n_per_user * 0.15)
        )
        # Pad to exactly n_per_user
        while len(decisions_list) < n_per_user:
            decisions_list.append("ACCEPT")
        decisions_list = decisions_list[:n_per_user]
        random.shuffle(decisions_list)

        for i, decision in enumerate(decisions_list):
            # Pick a random email template for this decision
            template_list = EMAIL_TEMPLATES[decision]
            subject_template, body_template, _ = random.choice(template_list)

            subject = fill_template(subject_template)
            body = fill_template(body_template)
            incoming_email = f"Subject: {subject}\n\n{body}"
            user_response = generate_user_response(user_id, decision, incoming_email)

            # Spread emails ~2–5 business days apart
            gap_days = random.randint(2, 5)
            current_date += timedelta(days=gap_days)
            # Skip weekends
            while current_date.weekday() >= 5:
                current_date += timedelta(days=1)

            records.append({
                "record_id": f"{user_id}_{i:04d}",
                "user_id": user_id,
                "timestamp": current_date.strftime("%Y-%m-%d %H:%M:%S"),
                "email_subject": subject,
                "incoming_email": incoming_email,
                "user_response": user_response,
                "decision": decision,
            })

    # Sort all records by timestamp
    records.sort(key=lambda r: r["timestamp"])
    return records


def main():
    print("Generating synthetic email dataset...")
    records = generate_dataset(n_per_user=60)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["record_id", "user_id", "timestamp", "email_subject",
                  "incoming_email", "user_response", "decision"]

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"[OK] Dataset written to: {OUTPUT_PATH}")
    print(f"   Total records : {len(records)}")
    print(f"   Users         : {list(PERSONAS.keys())}")

    # Decision distribution
    from collections import Counter
    dist = Counter(r["decision"] for r in records)
    print("   Decision dist :", dict(dist))


if __name__ == "__main__":
    main()
