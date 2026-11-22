# 🚀 Startup Idea Validator (AI-Powered)

## ⚡ Lean validator (v2): KILL or TEST

`validator_v2/` is a five-step pipeline that ends in a decision, not a report:

```
intake → market research (Google Search) → Hormozi → Africa lens → verdict → report
```

- **Intake**: pins the idea down to a paying buyer, problem, current alternative, offer and price.
- **Market research**: competitors and proof people already pay, Nigeria/Africa first.
- **Hormozi**: Value Equation, Grand Slam Offer, attraction offer, and which Core Four lead channel to start with.
- **Africa lens**: trust, distribution, payments, missing infrastructure, regulation.
- **Verdict**: **KILL** with the one reason, or **TEST** with a one-week experiment to get a first payment, including the exact first message to send.

The report is rendered from structured state with no extra LLM call, so it can't add claims the agents didn't make.

```bash
pip install -r requirements.txt
export GOOGLE_API_KEY=...                       # Gemini API key
export FOUNDER_CONTEXT="Solo founder, evenings only, under N50k per test"  # optional
adk web                                         # from the repo root, pick validator_v2
python -m pytest tests/                         # report tests, no API key needed
```

> The original pipeline in `app/` is kept for reference. Its `root_agent` is a single
> LLM with no sub-agents, so the multi-agent pipeline described below never runs there.

---
Validate your startup ideas using AI agents with real-time voice and video interaction. Built using [Google Agent Development Kit (ADK)](https://ai.google.dev/agents), this tool helps entrepreneurs assess originality, feasibility, and scalability—on the go.

---

## ✨ What It Does

**Pitch your idea** via voice, text, or video and get:
- ✅ Competitor analysis from real-time web search
- ✅ Market demand insights
- ✅ MVP suggestions
- ✅ Monetization models
- ✅ AI-generated scores on originality, feasibility, and scalability

All delivered **via speech, text, and optionally video**.

---

## 🧠 Multi-Agent Architecture

| Agent         | Responsibility                                 |
|--------------|-------------------------------------------------|
| `MarketAgent` | Finds similar startups & checks uniqueness     |
| `MVPAgent`    | Proposes MVP with core features & revenue ideas|
| `ScoringAgent`| Grades idea across key metrics                 |
| `InvestorAgent`| Analyzed the business idea from a VC investor POV   |
| `StreamingAgent`| Orchestrates live bidirectional sessions     |

---

## 🔥 Features

### 🗣️ Voice + Video Streaming (Bidi)
- Natural, low-latency voice-first UX
- Interruptible conversations
- Multimodal input/output (text, audio, video)
- Ideal for pitching startup ideas on the fly


## 🧩 Tech Stack

- **Backend**: Python + Google ADK
- **Voice/Video Streaming**: Gemini Live Models (e.g. `gemini-2.5-flash-live`)
- **Frontend**: Next.js or SvelteKit (WebSockets for streaming)
- **Deployment**: Cloud Run / Vertex AI Agent Engine

---

## 🗂️ Project Milestones

### ✅ Milestone 1 – Planning & Setup
- Define pipeline
- Setup ADK project & tools
- Configure memory + agents

### ✅ Milestone 2 – Core Agent Pipeline
- Build `MarketAgent`, `MVPAgent`, `ScoringAgent`
- Chain via `SequentialAgent`
- Test with text-only input/output

### ✅ Milestone 3 – Voice & Video Streaming
- Implement `LiveRequestQueue`
- Use `StreamingMode.BIDI`
- Integrate speech-to-text & TTS

### ✅ Milestone 4 – WebUI
- Audio/video input form
- WebSocket client-server stream
- Live transcripts & feedback

### ✅ Milestone 5 – Testing & Evaluation
- Validate scores
- Tune prompts
- Session history checks

### ✅ Milestone 6 – Deployment
- Containerize backend
- Deploy to Cloud Run
- User test and polish UX

---

## 💡 Sample Use Case

> 🎙️ *"I want to build an app that connects remote software developers with African startups needing affordable talent."*

🧠 The agent will:
- Search existing platforms (e.g. Andela, Turing)
- Check search trends for "hire African developers"
- Suggest MVP features (profile matching, escrow, video chat)
- Score originality: 7.8, feasibility: 8.5, scalability: 9.1

---

## 📦 Installation

```bash
git clone https://github.com/yourusername/startup-idea-validator.git
cd startup-idea-validator
pip install -r requirements.txt
````

---

## 🚀 Run (Development)

```bash
python main.py
# Or launch Web UI
npm run dev  # for frontend (if using Next.js/SvelteKit)
```

---

## 📜 License

MIT License © 2025 Joseph

