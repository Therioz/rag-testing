"""
hello_llm.py - Milestone 0: talk to a local LLM from Python.

Usage examples:
    python hello_llm.py What is RAG?
    python hello_llm.py "What is RAG?" --system "Answer in one sentence"
    python hello_llm.py "Write a haiku about Python" --temperature 1.5

Requirements:
    - Ollama app running in the background
    - `ollama pull llama3.2` done once
    - `pip install ollama` inside your .venv
"""

# argparse is Python's built-in tool for reading command-line arguments.
# It's a nicer version of sys.argv: it handles flags (--system), types,
# default values, and even generates a --help message for free.
import argparse

# The official Ollama Python library. It talks to the Ollama app running
# on your Mac (at http://localhost:11434) so you don't write HTTP code yourself.
import ollama

# Keeping the model name in one constant means you change it in one place
# when you want to try a different model (e.g. "qwen2.5" or "llama3.1").
MODEL = "llama3.2"


def generate(prompt: str, system: str = "", temperature: float = 0.7) -> str:
    """
    Send a prompt to the local LLM and return only the answer text.

    This function is the ONLY place in the project that knows about Ollama.
    Later, if you switch to the Claude API, you only rewrite this function
    and every other file keeps working. This pattern is called a "wrapper".

    Args:
        prompt: The user's question or instruction.
        system: Optional instructions that shape HOW the model answers
                (tone, length, persona). Empty string means "no system prompt".
        temperature: Randomness. 0 = focused and repeatable,
                     ~0.7 = balanced, 1.5+ = creative/chaotic.
    """
    # Chat models take a list of messages, each with a role:
    #   "system"    -> background instructions for the model
    #   "user"      -> what the human says
    #   "assistant" -> what the model said before (used for multi-turn chats)
    messages = []

    # An empty string is "falsy" in Python, so this only runs if a
    # system prompt was actually provided.
    if system:
        messages.append({"role": "system", "content": system})

    messages.append({"role": "user", "content": prompt})

    # The actual call to the model. `options` holds generation settings;
    # temperature is the most important one to understand.
    response = ollama.chat(
        model=MODEL,
        messages=messages,
        options={"temperature": temperature},
    )

    # The response is an object with many fields (model, timing info, etc.).
    # The answer text lives at response -> message -> content.
    return response.message.content


def main() -> None:
    # Describe what the script expects on the command line.
    parser = argparse.ArgumentParser(description="Ask a local LLM a question.")

    # A "positional" argument: the words after the script name.
    # nargs="+" means "one or more words", so quotes become optional:
    #   python hello_llm.py What is RAG   -> ["What", "is", "RAG"]
    parser.add_argument("question", nargs="+", help="The question to ask")

    # "Optional" arguments start with --. They have defaults if not given.
    parser.add_argument("--system", default="", help="Optional system prompt")
    parser.add_argument(
        "--temperature",
        type=float,  # argparse converts the text "1.5" into the number 1.5
        default=0.7,
        help="Randomness from 0 to ~2 (default 0.7)",
    )

    # Reads the command line. If the question is missing, argparse prints
    # a usage message and exits automatically - no extra code needed.
    args = parser.parse_args()

    # Join the list of words back into one sentence.
    question = " ".join(args.question)

    answer = generate(question, system=args.system, temperature=args.temperature)
    print(answer)


# This block only runs when you execute this file directly
# (python hello_llm.py). When another file does `from hello_llm import generate`,
# it is skipped - so you can reuse generate() in later milestones.
if __name__ == "__main__":
    main()