#!/usr/bin/env python3
"""LangGraph Helper Agent CLI.

Examples:
    python main.py --mode offline "How do I add persistence to a LangGraph agent?"
    python main.py --mode online "What's new in the latest LangGraph release?"

    export AGENT_MODE=online
    python main.py "What are the latest LangGraph features?"

    python main.py --mode offline   # no question -> interactive REPL
"""

from __future__ import annotations

import argparse
import sys

from langgraph_helper.config import settings
from langgraph_helper.graph import get_compiled_graph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LangGraph Helper Agent")
    parser.add_argument(
        "--mode",
        choices=["offline", "online"],
        default=None,
        help="Retrieval mode. Defaults to $AGENT_MODE, then 'offline'.",
    )
    parser.add_argument(
        "--provider",
        choices=["gemini", "openrouter", "ollama"],
        default=None,
        help="LLM provider. Defaults to $LLM_PROVIDER, then 'gemini'.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print retrieved context passages.")
    parser.add_argument("question", nargs="*", help="Question to ask. Omit for interactive mode.")
    return parser.parse_args()


def run_once(question: str, mode: str, verbose: bool) -> None:
    graph = get_compiled_graph()
    try:
        result = graph.invoke({"question": question, "mode": mode})
    except Exception as e:
        print(f"[error] LLM call failed: {e}", file=sys.stderr)
        print(
            "[hint] If this is a timeout, the API may be under heavy load -- try again, "
            "or increase LLM_TIMEOUT_SECONDS in .env.",
            file=sys.stderr,
        )
        return

    if result.get("retrieval_warning"):
        print(f"[warning] {result['retrieval_warning']}", file=sys.stderr)

    if verbose:
        print(f"\n--- retrieved via {result.get('retrieval_backend')} ---", file=sys.stderr)
        for c in result.get("context_chunks", []):
            print(f"  * {c['title']} ({c['url']})", file=sys.stderr)
        print("---\n", file=sys.stderr)

    print(result["answer"])


def main() -> None:
    args = parse_args()
    mode = settings.validate_mode(args.mode)
    if args.provider:
        settings.llm_provider = args.provider

    print(f"[mode={mode} provider={settings.llm_provider}]", file=sys.stderr)

    if args.question:
        run_once(" ".join(args.question), mode, args.verbose)
        return

    print("Interactive mode. Type a question, or 'exit' to quit.")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break
        run_once(question, mode, args.verbose)


if __name__ == "__main__":
    main()
