# TODO — PharmaRA Demo

Last updated: 2026-05-03

---

## Must-Have Before Wednesday (May 6, 2026) — ALL COMPLETE ✅

### Session 2026-05-01 (Today)
Implemented from scratch:
- [x] Per-user chat history isolation (namespaced session state)
- [x] Redis-backed chat persistence (`chat/history.py`)
- [x] Admin PDF upload page with 2-pass semantic chunking
- [x] Clear chat history button
- [x] Fixed auth bugs: session-only cookies + proper login gate flow
- [x] Removed UI flicker (`st.rerun()` removal)
- [x] Full visual smoke test — all features working ✓

### Remaining Must-Haves

- [x] Copy `.env.example` → `.env` and set `OPENAI_API_KEY` *(done — Google Gemini API configured)*
- [x] Build Redis vector index (`python scripts/build_index.py`) — 33 chunks in `pharma_ra` index *(done — verified 2026-04-30)*
- [x] Programmatic end-to-end pipeline test — all 5 scenarios verified *(done 2026-04-30)*:
  - [x] RAG: returns answer with `[Source: ...]` citations
  - [x] CSV: returns markdown table sorted by IC50
  - [x] Guardrail: fires on "What dosage should I take for COPD?" — no LLM call
  - [x] GitHub tool: fetches `deepmind/alphafold` README correctly
  - [x] Scholar tool: returns paper metadata from Semantic Scholar
- [x] Open http://localhost:8501 in browser and do a visual smoke test: *(done 2026-05-01)*
  - [x] Verify citations appear in expandable Sources panel
  - [x] Verify guardrail shows red `st.error` banner (not just text)
  - [x] Click all 7 sidebar example queries — confirm each works without errors
  - [x] Check no crashes or unhandled exceptions (browser console clean)
- [ ] Prepare 3–4 demo talking points matching each tool to a Sanofi R&D use case — **see DEMO_NOTES.md** *(optional; script ready, can improvise from DEMO_NOTES.md during interview)*

---

## Nice-to-Have (if time permits before Wednesday)

- [x] Add a "How it works" diagram or flow description in the sidebar (text only, no image) *(done 2026-05-03)*
- [x] Show tool call trace in the UI (which tool was invoked, with what args) — `🔧 Tool calls` expander *(done 2026-05-03)*
- [x] Add 1–2 more mock papers to improve RAG coverage — PROTAC degraders + immuno-oncology *(done 2026-05-03)*
- [x] Add a "Clear chat" button to the sidebar *(done 2026-05-01)*
- [x] Improve CSV tool: add support for date-range filters and researcher-name queries *(done 2026-05-03)*
- [x] Add `st.toast` confirmation when guardrail fires (in addition to red banner) *(done 2026-05-03)*
- [x] Pin exact package versions in `requirements.txt` for reproducibility *(done 2026-05-03)*

---

## Future Production Extensions

- [ ] MCP integration: wrap assay tool as an MCP server for use in Cursor / Claude Code
- [ ] PDF upload: replace `.txt` files with `PyMuPDF` loader for real documents
- [ ] Persistent chat memory: add LangGraph `MemorySaver` checkpointer with session IDs
- [ ] Authentication: add Streamlit auth or an API key gate for multi-user deployment
- [ ] Molecule viewer: render RDKit 2D structures for compound IDs found in CSV queries
- [ ] Batch embedding refresh: watch `data/docs/` for new files and auto-rebuild index
- [ ] Evaluation harness: ragas or ARES for RAG quality scoring on the 5 mock papers
- [ ] Real Sanofi document corpus: replace mock papers with internal R&D documents (behind auth)
- [ ] Streaming responses: use LangChain streaming + `st.write_stream` for faster perceived latency
- [ ] Cost tracking: log token usage per query for budget awareness in production
