# TODO — PharmaRA Demo

Last updated: 2026-04-30

---

## Must-Have Before Wednesday (May 6, 2026)

- [x] Copy `.env.example` → `.env` and set `OPENAI_API_KEY` *(done — Google Gemini API configured)*
- [x] Build Redis vector index (`python scripts/build_index.py`) — 33 chunks in `pharma_ra` index *(done — verified 2026-04-30)*
- [x] Programmatic end-to-end pipeline test — all 5 scenarios verified *(done 2026-04-30)*:
  - [x] RAG: returns answer with `[Source: ...]` citations
  - [x] CSV: returns markdown table sorted by IC50
  - [x] Guardrail: fires on "What dosage should I take for COPD?" — no LLM call
  - [x] GitHub tool: fetches `deepmind/alphafold` README correctly
  - [x] Scholar tool: returns paper metadata from Semantic Scholar
- [ ] Open http://localhost:8501 in browser and do a visual smoke test:
  - [ ] Verify citations appear in expandable Sources panel
  - [ ] Verify guardrail shows red `st.error` banner (not just text)
  - [ ] Click all 7 sidebar example queries — confirm each works without errors
  - [ ] Check no crashes or unhandled exceptions (browser console clean)
- [ ] Prepare 3–4 demo talking points matching each tool to a Sanofi R&D use case — **see DEMO_NOTES.md**

---

## Nice-to-Have (if time permits before Wednesday)

- [ ] Add a "How it works" diagram or flow description in the sidebar (text only, no image)
- [ ] Show tool call trace in the UI (which tool was invoked, with what args) — use `st.expander` or `st.caption`
- [ ] Add 1–2 more mock papers to improve RAG coverage (e.g., PROTAC degraders, immuno-oncology)
- [ ] Add a "Clear chat" button to the sidebar
- [ ] Improve CSV tool: add support for date-range filters and researcher-name queries
- [ ] Add `st.toast` confirmation when guardrail fires (in addition to red banner)
- [ ] Pin exact package versions in `requirements.txt` for reproducibility

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
