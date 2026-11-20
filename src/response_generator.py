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


# -- Prompt builder ------------------------------------------------------------

def _build_prompt(
    incoming_email: str,
    predicted_decision: str,
    memories: List[Dict[str, Any]],
    include_memories: bool = True,
    include_decision: bool = True,
) -> str:
    """
    Construct the LLM prompt.

    Parameters:
        include_memories : if False, omit historical memories (ablation: no_context)
        include_decision : if False, omit the ML-predicted decision (ablation: memory_only)

    LEAKAGE NOTE:
        The target email's 'user_response' is NEVER included here.
        The LLM must infer style from historical memories only.
    """
    # -- Memories block --------------------------------------------------------
    if include_memories and memories:
        mem_lines = []
        for i, m in enumerate(memories[:20], 1):
            mem_lines.append(
                f"[Memory {i}]"
                f"\n  Date      : {m['timestamp']}"
                f"\n  Subject   : {m['email_subject']}"
                f"\n  Email     : {m['incoming_email'][:200]}..."
                f"\n  My reply  : {m['user_response'][:300]}"
                f"\n  Decision  : {m['decision']}"
                f"\n  Similarity: {m['similarity_score']:.3f}"
            )
        memories_block = "\n\n".join(mem_lines)
        mem_section = (
            f"## Historical Memory ({len(memories)} past emails retrieved)\n\n"
            f"{memories_block}"
        )
    else:
        mem_section = (
            "## Historical Memory\n\n"
            "(No historical memories available — responding without personalization context.)"
        )

    # -- Decision block --------------------------------------------------------
    style_instruction = (
        "Adopt the writing style inferred from the historical memories above"
        if include_memories else
        "Use a clear, neutral, professional writing style"
    )

    if include_decision:
        task_section = f"""## ML-Predicted Decision

The user's decision for this email is: **{predicted_decision}**

## Your Task

Write a reply email that:
1. Directly and specifically addresses the content of the incoming email above
2. Communicates the decision ({predicted_decision}) naturally within the reply — do NOT just say "I accept your request" generically; weave it into a real, context-specific response
3. {style_instruction} — match the tone, formality, greeting, sign-off, and sentence length from the memories
4. NEVER copy a past reply word-for-word; always tailor the response to THIS specific email
5. Does NOT reveal that you are an AI

Write ONLY the email body. No preamble, no explanation, no labels.
"""
    else:
        task_section = f"""## Your Task

Write a reply email that:
1. Directly and specifically addresses the content of the incoming email above
2. Decides naturally whether to accept, decline, defer, or ask for more information — based on the email content
3. {style_instruction} — match the tone, formality, greeting, sign-off, and sentence length from the memories
4. NEVER copy a past reply word-for-word; always tailor the response to THIS specific email
5. Does NOT reveal that you are an AI

Write ONLY the email body. No preamble, no explanation, no labels.
"""

    prompt = f"""You are acting as a Cognitive Twin — a system that replies to emails \
exactly as a specific person would, based on their communication history.

{mem_section}

## Incoming Email to Reply To

{incoming_email}

{task_section}"""
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
