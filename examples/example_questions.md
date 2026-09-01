# Example questions

Try these with either mode:

```bash
python main.py --mode offline "How do I add persistence to a LangGraph agent?"
python main.py --mode offline "What's the difference between StateGraph and MessageGraph?"
python main.py --mode offline "Show me how to implement human-in-the-loop with LangGraph"
python main.py --mode offline "How do I handle errors and retries in LangGraph nodes?"
python main.py --mode offline "What are best practices for state management in LangGraph?"

python main.py --mode online "What changed in the most recent LangGraph release?"
python main.py --mode online "Are there any new LangChain v1 breaking changes I should know about?"
```

Add `--verbose` to any of these to see which document chunks / search
results were retrieved and fed to the model as context, e.g.:

```bash
python main.py --mode offline --verbose "How do I use checkpointers?"
```
