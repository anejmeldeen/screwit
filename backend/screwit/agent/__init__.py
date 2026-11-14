"""The chat agent: a Claude tool-use loop plus the mock-script replayer (SPEC 6.3, 7 P4).

ChatAgent implementations yield contracts ChatEvent objects ending in a DoneEvent (interfaces.py).
Mock mode and tests use the scripted / fake-LLM paths so nothing calls out.
"""
