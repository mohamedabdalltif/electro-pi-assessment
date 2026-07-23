# Section 1 Notes

## Barge-in / Interruption Handling

LiveKit handles interruptions through VAD events. When the user starts speaking mid-response, the TTS stream gets cancelled, audio buffers flushed, and the agent goes back to listening. In practice you'd hook into the `will_interrupt_agent()` callback to do any cleanup before the agent stops speaking — like logging the interruption or resetting any state you were tracking.

## Adding a Second Tool Safely

Adding another `@function_tool` is straightforward, but a few things matter:

1. Keep type hints and docstrings clear — the LLM decides when to call a tool based on the docstring, so it needs to be descriptive.
2. Validate inputs before using them (especially order IDs or user-provided strings).
3. Always wrap the tool body in a try/except. If the tool hits an API or database failure, you want to return a helpful error string back to the LLM rather than crashing the whole agent.

## Bonus — Swapping the STT Provider

The nice thing about LiveKit's plugin architecture is that STT, LLM, and TTS are all just arguments to `AgentSession`. To swap from OpenAI STT to Deepgram, you change exactly one line:

```python
stt=deepgram.STT(model="nova-2")
```

Nothing else changes — the agent class, tools, and instructions are all untouched. See `agent_swapped.py` for the full example.
