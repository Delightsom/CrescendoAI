# CrescendoAI 

**CrescendoAI** is a modular Python-based AI security tool designed to automate multi-turn red-teaming campaigns against LLM endpoints. It tests guardrail resilience by executing stateful, incremental prompt escalation (the "Crescendo" attack technique) rather than relying on static single-turn filters.

##  Architecture

The project is structured modularly for maintainability and extension:
- `core/`: Houses the main orchestration loop, state machine, and runner logic.
- `strategies/`: Contains attacker prompt templates and multi-turn escalation state management.
- `targets/`: Client wrappers for interacting with OpenAI-compatible API endpoints.
- `main.py`: CLI entry point utilizing `rich` for structured, color-coded terminal logging and run summaries.

##  Quick Start

1. **Clone the repository:**