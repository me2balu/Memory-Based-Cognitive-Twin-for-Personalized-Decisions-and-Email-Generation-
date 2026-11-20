"""
response_generator.py
----------------------
LLM wrapper for personalized email response generation.

Provider: Groq (openai-compatible, free tier)
Env vars: LLM_API_KEY  — Groq API key from console.groq.com
          LLM_MODEL    — defaults to llama-3.1-8b-instant

Design goals:
  - Single public function: generate_response(...)
  - Real mode: calls Groq API with retrieved memories as context.
  - Mock mode: returns a clearly tagged [MOCK] string without any API call.
  - Easy to swap provider: change only the _call_llm() function.

IMPORTANT:
  - API key is NEVER hardcoded.
  - Mock results must NOT be presented as research results.
  - The target email's actual response is NEVER passed to the LLM.
  - The LLM infers the user's communication style from the 20 retrieved memories.
"""

import os
import re
import pathlib
from typing import List, Dict, Any

# Load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass   # .env loading is optional; env vars may be set by the shell

# -- Configuration -------------------------------------------------------------
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_MODEL   = os.environ.get("LLM_MODEL", "llama-3.1-8b-instant")

MOCK_MODE = not bool(LLM_API_KEY)   # automatically True when no key is set


# -- Hardcoded persona style rules (overrides noisy synthetic data) ------------

PERSONA_RULES = {
    "alice": """Alice writes very formal, polished professional emails.
RULES (follow strictly):
- Always open with "Dear [Name]," (use a realistic first name)
- Never use contractions: write "do not" not "don't", "cannot" not "can't", "I would" not "I'd"
- Use formal vocabulary: "I am pleased to confirm", "Please be advised", "I would like to inform you", "I shall", "Kindly", "Furthermore", "In addition"
- Write in complete, grammatically perfect sentences — no fragments
- Tone is measured, polite, never casual
- Always close with a formal sign-off on its own line: "Best regards,\\nAlice" or "Kind regards,\\nAlice" or "Sincerely,\\nAlice"
- Emails are 3–5 sentences long""",

    "bob": """Bob writes casual, laid-back emails like he's texting a friend.
RULES (follow strictly):
- Open with "Hey [Name]," or "Hi," or just dive straight in
- Use contractions everywhere: "I'd", "can't", "won't", "that's", "it's", "we'll"
- Use informal/slang words: "yeah", "nope", "totally", "sounds good", "no worries", "tbh", "btw", "fyi", "cool", "awesome"
- Short sentences. Sometimes incomplete sentences. That's fine.
- No corporate buzzwords, no formal vocabulary at all
- Close with "Cheers, Bob" or "Later, Bob" or "Thanks, Bob" — always lowercase-feeling
- Emails are 2–3 short sentences max""",

    "carol": """Carol writes warm, enthusiastic, friendly emails — she genuinely loves people.
RULES (follow strictly):
- Open with "Hi [Name]! Hope you're doing well!" or "Hello [Name]! So great to hear from you!" — always with exclamation marks
- Use exclamation marks throughout — multiple per email
- Warm phrases: "That sounds wonderful!", "I'd absolutely love to!", "How exciting!", "Can't wait!", "So appreciate you reaching out!"
- Ask a friendly follow-up question to show genuine interest
- Never sound cold or transactional — always personal and caring
- Close with "Warmly, Carol" or "Talk soon! Carol" or "Take care! Carol"
- Emails are 3–4 cheerful sentences""",

    "dave": """Dave writes the shortest possible emails — he values time above everything.
RULES (follow strictly):
- Often NO greeting at all — or just "[Name] -" on the first line
- Responses are 1–3 sentences MAXIMUM, often just 1
- Accept = "Yes." or "Fine." or "Works for me." or "Agreed."
- Reject = "No." or "Can't." or "Declined." or "Not possible."
- Defer = "Later." or "Not now." or "Revisit next month."
- Request info = "Need more details." or "What's the context?" or "Clarify X first."
- No filler words, no pleasantries, no "I hope this email finds you well"
- Close with just "-Dave" or "Dave" or nothing at all
- If the email is 1 sentence, that is a success""",

    "eve": """Eve writes structured, analytical emails — everything is evidence-based.
RULES (follow strictly):
- Open with "Dear [Name]," or "Hello [Name],"
- Reference data, metrics, or logic: "Based on the available data...", "The risk-benefit analysis indicates...", "From an analytical standpoint...", "The evidence suggests...", "Considering the metrics..."
- Use conditional, precise language: "Subject to...", "Provided that...", "Contingent on...", "This depends on..."
- May use a brief numbered list or structure if the reply has multiple points
- Never emotional, always rational and objective
- Close with "Best, Eve" or "Regards, Eve" or "Analytically yours, Eve"
- Emails are 3–5 structured sentences""",
}


# -- Style profile extractor (fallback for unknown users) ----------------------

def _extract_style_profile(memories: List[Dict[str, Any]]) -> str:
    """
    Distil a concise style guide from memories instead of dumping
    20 near-identical template responses to the LLM.
    """
    responses = [m["user_response"] for m in memories if m.get("user_response")]

    # Greetings = first non-blank line of each response
    greetings = []
    for r in responses:
        lines = [l.strip() for l in r.strip().split("\n") if l.strip()]
        if lines:
            greetings.append(lines[0])

    # Sign-offs = last non-blank line
    signoffs = []
    for r in responses:
        lines = [l.strip() for l in r.strip().split("\n") if l.strip()]
        if len(lines) >= 2:
            signoffs.append(lines[-1])

    # Avg sentence length
    lengths = []
    for r in responses:
        for s in re.split(r"[.!?]+", r):
            words = s.split()
            if words:
                lengths.append(len(words))
    avg_len = round(sum(lengths) / len(lengths)) if lengths else 10

    # Tone: formal vs casual
    all_text = " ".join(responses).lower()
    formal = sum(1 for w in
                 ["sincerely","regards","dear","kindly","furthermore",
                  "accordingly","pleased","regret","appreciate"]
                 if w in all_text)
    casual = sum(1 for w in
                 ["hey","yeah","sure","cheers","btw","fyi","nope",
                  "totally","later","sounds good"]
                 if w in all_text)
    tone = "formal and professional" if formal > casual else \
           "casual and conversational" if casual > formal else "neutral"

    # Unique samples
    unique_greetings = list(dict.fromkeys(greetings))[:3]
    unique_signoffs  = list(dict.fromkeys(signoffs))[:3]

    return (
        f"- Tone            : {tone}\n"
        f"- Sentence length : ~{avg_len} words per sentence\n"
        f"- Greetings used  : {' | '.join(unique_greetings) or 'varies'}\n"
        f"- Sign-offs used  : {' | '.join(unique_signoffs)  or 'varies'}"
    )


# -- Prompt builder ------------------------------------------------------------

def _build_prompt(
    incoming_email: str,
    predicted_decision: str,
    memories: List[Dict[str, Any]],
    include_memories: bool = True,
    include_decision: bool = True,
    user_id: str = None,
) -> str:
    """
    Build the LLM prompt using hardcoded PERSONA_RULES (if available for this
    user) or a style profile extracted from memories as fallback.
    """
    # --- Style section ---
    if user_id and user_id in PERSONA_RULES:
        # Hardcoded vivid rules — far better than noisy synthetic data
        mem_section = (
            f"## Persona Style Rules — FOLLOW THESE EXACTLY\n\n"
            f"{PERSONA_RULES[user_id]}"
        )
    elif include_memories and memories:
        style_profile = _extract_style_profile(memories)

        # Pick 3 topically diverse examples (spread across the ranked list)
        picks = []
        step = max(1, len(memories) // 3)
        for i in range(0, min(len(memories), 3 * step), step):
            picks.append(memories[i])
        picks = picks[:3]

        example_lines = []
        for i, m in enumerate(picks, 1):
            example_lines.append(
                f"Example {i} (decision={m['decision']}):\n"
                f"  Received : {m['incoming_email'][:180].strip()}…\n"
                f"  Replied  : {m['user_response'][:250].strip()}"
            )
        examples_block = "\n\n".join(example_lines)

        mem_section = (
            f"## This Person's Writing Style\n\n"
            f"{style_profile}\n\n"
            f"## Example Replies They Have Written\n\n"
            f"{examples_block}"
        )
    else:
        mem_section = "## Style\n\nUse clear, neutral, professional English."

    decision_line = (
        f"Decision to convey: **{predicted_decision}**\n"
        if include_decision else
        "Decide naturally (accept / decline / defer / ask for info) based on the email.\n"
    )

    prompt = f"""You are simulating how a specific person replies to emails.
Study the style rules/profile below, then write a ONE-OFF reply to the new email.

{mem_section}

## New Email to Reply To

{incoming_email.strip()}

## Instructions

{decision_line}
Write a reply that:
- Directly addresses the SPECIFIC content and topic of the email above (names, dates, topics)
- Follows the persona style rules STRICTLY (tone, greeting, sign-off, sentence length, vocabulary)
- Does NOT copy any template phrases verbatim — write fresh, specific content every time
- Is complete (greeting + body + sign-off, unless the persona rules say otherwise)
- Does NOT mention that you are an AI

Write ONLY the reply. No labels, no preamble."""
    return prompt


# -- Groq API caller -----------------------------------------------------------

def _call_llm(prompt: str) -> str:
    """
    Call the Groq API (OpenAI-compatible) and return the generated text.
    Free tier: https://console.groq.com — no billing required.
    This is the only function that needs to change if you switch providers.
    """
    try:
        from groq import Groq
    except ImportError:
        raise ImportError(
            "groq is not installed.\n"
            "Run: pip install groq"
        )

    client = Groq(api_key=LLM_API_KEY)

    for attempt in range(3):
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an email assistant. Always reply with a complete, "
                        "specific email response that directly addresses the content "
                        "of the incoming email. Never return a generic or empty response."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            model=LLM_MODEL,
            temperature=0.7 + attempt * 0.1,   # raise temp slightly on retry
            max_tokens=600,
        )
        msg = chat_completion.choices[0].message

        # Reasoning models (gpt-oss-*) sometimes put the answer in
        # reasoning_content when content is empty — handle both fields
        content = msg.content or ""
        if not content.strip():
            reasoning = getattr(msg, "reasoning_content", None) or ""
            content = reasoning

        text = content.strip()
        if text:
            return text
        print(f"  [warn] Empty response (attempt {attempt+1}/3), retrying…")

    raise ValueError(
        "LLM returned empty after 3 attempts. "
        "Try a more descriptive email input, e.g. "
        "'Can we schedule a meeting on Monday to review the Q3 budget?'"
    )


# -- Mock mode -----------------------------------------------------------------

def _mock_response(
    incoming_email: str,
    predicted_decision: str,
    memories: List[Dict[str, Any]],
) -> str:
    """
    Return a clearly labelled mock response for testing without an API key.

    [!] MOCK RESULTS MUST NOT BE PRESENTED AS RESEARCH RESULTS.
    This is purely for pipeline testing and development.
    """
    style_hint = ""
    if memories:
        sample = memories[0]["user_response"][:80]
        style_hint = f'(style hint from memory: "{sample}...")'

    return (
        f"[MOCK RESPONSE — NOT A REAL LLM OUTPUT]\n\n"
        f"Decision: {predicted_decision}\n"
        f"This is a placeholder response generated in MOCK MODE because no "
        f"LLM_API_KEY was provided.\n"
        f"{style_hint}\n\n"
        f"To get real responses, set LLM_API_KEY in your .env file."
    )


# -- Public API ----------------------------------------------------------------

def generate_response(
    incoming_email: str,
    predicted_decision: str,
    memories: List[Dict[str, Any]],
    force_mock: bool = False,
    ablation_mode: str = "full",
    user_id: str = None,
) -> Dict[str, Any]:
    """
    Generate a personalized email response using retrieved historical memories.

    Parameters:
        incoming_email     : the new email the user received
        predicted_decision : ML-predicted decision label (e.g., 'ACCEPT')
        memories           : list of up to 20 memory dicts from retrieve_memories()
        force_mock         : override to True to bypass the API (for testing)
        ablation_mode      : controls which components are included in the prompt:
                               'full'        — memories + ML decision (default)
                               'memory_only' — memories only, no ML decision
                               'no_context'  — neither memories nor ML decision
        user_id            : persona name (alice/bob/carol/dave/eve) — enables
                               hardcoded PERSONA_RULES for authentic style

    Returns:
        Dict with keys:
          'response'      : generated text
          'mode'          : 'real' or 'mock'
          'model'         : model name used
          'n_memories'    : number of memories used in the prompt
          'ablation_mode' : which ablation condition was used
    """
    # Map ablation mode to prompt flags
    _mode_flags = {
        "full":        {"include_memories": True,  "include_decision": True},
        "memory_only": {"include_memories": True,  "include_decision": False},
        "no_context":  {"include_memories": False, "include_decision": False},
    }
    flags = _mode_flags.get(ablation_mode, _mode_flags["full"])

    use_mock = MOCK_MODE or force_mock

    if use_mock:
        text = _mock_response(incoming_email, predicted_decision, memories)
        mode = "mock"
    else:
        prompt = _build_prompt(
            incoming_email,
            predicted_decision,
            memories,
            **flags,
            user_id=user_id,
        )
        text = _call_llm(prompt)
        mode = "real"

    return {
        "response":      text,
        "mode":          mode,
        "model":         LLM_MODEL if not use_mock else "mock",
        "n_memories":    len(memories) if flags["include_memories"] else 0,
        "ablation_mode": ablation_mode,
        "prompt":        _build_prompt(incoming_email, predicted_decision, memories, **flags, user_id=user_id) if not use_mock else "(mock)"
    }
