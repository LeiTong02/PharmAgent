# PharmaRA — Demo Talking Points

Target audience: PhD-level researchers and/or AI/ML engineers at Sanofi.

---

## Talking Point 1 — RAG over research papers
**Demo query:** "Summarize the key findings on EGFR T790M resistance mutations."
**What to show:** The agent calls `rag_search`, retrieves 4 relevant chunks from the internal paper library, and responds with inline `[Source: Title, Year]` citations that appear in an expandable Sources panel.

**Sanofi angle:**
> "At Sanofi, this vector index would sit over internal clinical study reports, preclinical write-ups, and published literature. Instead of a researcher spending an hour searching SharePoint and PubMed, they get a synthesized answer with traceable citations in seconds — and the citations link to the exact documents, not just a list of search results."

---

## Talking Point 2 — Structured assay data query
**Demo query:** "What are the IC50 values for EGFR inhibitors in the database?" or "Show me lead compounds for CDK4/Cyclin D1."
**What to show:** The agent calls `query_assay_data`, runs a pandas filter on `assay_results.csv`, and returns a clean markdown table sorted by potency.

**Sanofi angle:**
> "Medicinal chemists at Sanofi work with HTS databases of tens of thousands of compounds. Today they write SQL or open Excel. This lets them ask 'which BRAF inhibitors have selectivity ratio above 100 and are still active?' in plain English and get a table immediately — the same workflow, but accessible to anyone on the team."

---

## Talking Point 3 — Safety guardrail
**Demo query:** "What dosage should I take for COPD?" (then immediately follow with a legitimate research question to show it doesn't over-block)
**What to show:** The red error banner fires instantly — no spinner, no LLM call, no cost.

**Sanofi angle:**
> "A system like this wouldn't just serve internal researchers — it could be a front door to external partners or physicians. The guardrail fires before the LLM is ever invoked: zero added latency, reliable even if the API is down, and fully auditable. That matters in a regulated industry where you need to show exactly what the system does and doesn't do."

---

## Talking Point 4 — External tool calling (GitHub + Semantic Scholar)
**Demo query:** "Fetch the README from https://github.com/deepmind/alphafold" then "Look up papers on protein structure prediction with AlphaFold."
**What to show:** The agent correctly routes to two different tools, handles fallback gracefully, and synthesizes the GitHub README content and paper metadata into a coherent response.

**Sanofi angle:**
> "Sanofi's computational biology team needs to track fast-moving external research — new AlphaFold versions, open-source PROTAC tools, published benchmarks. This agent bridges internal and external knowledge in one interface, without the researcher having to switch contexts between GitHub, PubMed, and Slack."

---

## Architecture one-liner (if asked to describe the system)
> "It's a LangGraph state machine: guardrail fires first, then the LLM routes to whichever tool fits the query, and the result flows back for synthesis. Redis Stack persists the vector index, so re-embedding only happens if we add new documents — not on every request."

---

## Demo flow (suggested order, ~10 min)
1. Show the sidebar — explain the 4 tools briefly
2. Click "What are the IC50 values for EGFR inhibitors?" → CSV table with citations
3. Click "Summarize the key findings on EGFR T790M resistance mutations." → RAG answer with Sources panel
4. Type "What dosage should I take for COPD?" → guardrail fires instantly
5. Click "Fetch the README from https://github.com/deepmind/alphafold" → GitHub tool
6. Click "Look up papers on protein structure prediction with AlphaFold." → Scholar tool
7. Optional: show architecture diagram from PROJECT_CONTEXT.md in response to questions
