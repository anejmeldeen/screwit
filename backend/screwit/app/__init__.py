"""FastAPI platform shell for Screw It (SPEC 5, 7 P4).

Owns routing, loopback auth, the database, queue wiring and SSE helpers. Mock mode serves fixtures
so the UI and agent can be built with no pipeline, network or credentials (SPEC 8).
"""
