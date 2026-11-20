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


# -- Style profile extractor ---------------------------------------------------

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
) -> str:
    """
    Build the LLM prompt using a style profile + a few diverse examples.
    Avoids dumping 20 near-identical template responses which causes the LLM
    to copy them verbatim instead of writing a specific reply.
    """
    if include_memories and memories:
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
Study their style profile and examples below, then write a reply to the new email.

{mem_section}

## New Email to Reply To

{incoming_email.strip()}

## Instructions

{decision_line}
Write a reply that:
- Directly addresses the SPECIFIC content and topic of the email above
- Uses the person's writing style (tone, greeting, sign-off, sentence length) from the examples
- Does NOT copy example phrases word-for-word — write fresh, specific content
- Is complete (greeting + body + sign-off)
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
        temperature=0.9,
        max_tokens=600,
    )
    content = chat_completion.choices[0].message.content
    text = content.strip() if content else ""
    if not text:
        raise ValueError(
            "LLM returned an empty response. "
            "Try a longer or more specific email input."
        )
    return text


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
        )
        text = _call_llm(prompt)
        mode = "real"

    return {
        "response":      text,
        "mode":          mode,
        "model":         LLM_MODEL if not use_mock else "mock",
        "n_memories":    len(memories) if flags["include_memories"] else 0,
        "ablation_mode": ablation_mode,
        "prompt":        _build_prompt(incoming_email, predicted_decision, memories, **flags) if not use_mock else "(mock)"
    }
