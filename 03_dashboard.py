import streamlit as st
import json
import os
from groq import Groq
from config import GROQ_API_KEY
from pyvis.network import Network
import streamlit.components.v1 as components
import tempfile

st.set_page_config(page_title="ResearchMind", page_icon="🧠", layout="wide")

st.markdown("""
<style>
  .main { background-color: #0f1117; }
  .metric-card {
    background: #1a1d27; border: 1px solid #2a2d3a;
    border-radius: 10px; padding: 1rem 1.25rem; margin-bottom: 0.5rem;
  }
  .metric-label { font-size: 12px; color: #888; text-transform: uppercase; }
  .metric-value { font-size: 26px; font-weight: 600; color: #eee; }
  .finding-box {
    background: #1a1d27; border-left: 3px solid #7f77dd;
    border-radius: 0 8px 8px 0; padding: 0.75rem 1rem;
    margin-bottom: 0.5rem; font-size: 13px; color: #ccc;
  }
  h1, h2, h3 { color: #eee !important; }
</style>
""", unsafe_allow_html=True)

# ── Load data ──────────────────────────────────────────────────
@st.cache_data
def load_data():
    with open("data/papers_analyzed.json", "r") as f:
        papers = json.load(f)
    with open("outputs/graph_data.json", "r") as f:
        graph  = json.load(f)
    with open("outputs/contradictions.json", "r") as f:
        contradictions = json.load(f)
    return papers, graph, contradictions

papers, graph_data, contradictions = load_data()
paper_map = {p["id"]: p for p in papers}

# ── Header ─────────────────────────────────────────────────────
st.title("🧠 ResearchMind")
st.markdown("*AI-powered research paper knowledge graph*")
st.divider()

# ── Metrics ────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
paper_nodes = [n for n in graph_data["nodes"] if n.get("node_type") == "paper"]
concept_nodes = [n for n in graph_data["nodes"] if n.get("node_type") == "concept"]
sim_edges = [e for e in graph_data["edges"] if e.get("edge_type") == "similar"]

with col1:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Papers analyzed</div><div class="metric-value">{len(papers)}</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Shared concepts</div><div class="metric-value">{len(concept_nodes)}</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Connections found</div><div class="metric-value">{len(sim_edges)}</div></div>', unsafe_allow_html=True)
with col4:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Contradictions</div><div class="metric-value">{len(contradictions)}</div></div>', unsafe_allow_html=True)

st.divider()

# ── Tabs ───────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["🕸️ Knowledge Graph", "📄 Papers", "💬 Ask AI", "⚠️ Contradictions"])

# ════════════════════════════════════════════════════
# Tab 1 — Knowledge Graph
# ════════════════════════════════════════════════════
with tab1:
    st.subheader("Research Knowledge Graph")
    st.markdown("Purple = papers · Teal = shared concepts · Edges = similarity")

    net = Network(height="600px", width="100%", bgcolor="#0f1117", font_color="#eee")
    net.barnes_hut(gravity=-8000, central_gravity=0.3, spring_length=150)

    for node in graph_data["nodes"]:
        if node.get("node_type") == "paper":
            label = node.get("title", "")[:35] + "..."
            title = f"{node.get('title','')}\n{node.get('year','')}\n{node.get('contribution','')[:100]}"
            net.add_node(node["id"], label=label, title=title,
                        color="#7f77dd", size=20, font={"size": 11})
        else:
            net.add_node(node["id"], label=node["id"], title=node["id"],
                        color="#1d9e75", size=12, font={"size": 10})

    for edge in graph_data["edges"]:
        if edge.get("edge_type") == "similar":
            net.add_edge(edge["source"], edge["target"],
                        value=edge.get("weight", 0.5), color="#444")
        else:
            net.add_edge(edge["source"], edge["target"], color="#2a5a40")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode="w") as f:
        net.save_graph(f.name)
        html_content = open(f.name).read()

    components.html(html_content, height=620)

# ════════════════════════════════════════════════════
# Tab 2 — Papers
# ════════════════════════════════════════════════════
with tab2:
    st.subheader("Analyzed Papers")

    for paper in papers:
        with st.expander(f"📄 {paper['title']} ({paper['published']})"):
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown("**Key claims:**")
                for claim in paper["analysis"]["key_claims"]:
                    st.markdown(f'<div class="finding-box">• {claim}</div>',
                               unsafe_allow_html=True)
                st.markdown(f"**Methodology:** {paper['analysis']['methodology']}")
                st.markdown(f"**Contribution:** {paper['analysis']['contribution']}")
            with col2:
                st.markdown("**Concepts:**")
                for concept in paper["analysis"]["concepts"]:
                    st.markdown(f"`{concept}`")
                st.markdown(f"**Authors:** {', '.join(paper['authors'])}")
                st.markdown(f"[View paper →]({paper['url']})")

# ════════════════════════════════════════════════════
# Tab 3 — Ask AI
# ════════════════════════════════════════════════════
with tab3:
    st.subheader("Ask questions across all papers")
    st.markdown("The AI will answer using only the papers in the knowledge base.")

    question = st.text_input("Your question",
                             placeholder="What are the main approaches to improving LLM reasoning?")

    if question:
        with st.spinner("Thinking across papers..."):
            context = ""
            for p in papers:
                context += f"\nPaper: {p['title']} ({p['published']})\n"
                context += f"Abstract: {p['abstract'][:300]}\n"
                context += f"Claims: {'; '.join(p['analysis']['key_claims'])}\n"

            prompt = f"""You are a research assistant with access to these papers:

{context}

Question: {question}

Answer the question using ONLY information from the papers above.
For each point you make, cite the paper title in brackets like [Paper Title].
Be concise and precise."""

            client = Groq(api_key=GROQ_API_KEY)
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )

            answer = response.choices[0].message.content
            st.markdown("### Answer")
            st.markdown(answer)

# ════════════════════════════════════════════════════
# Tab 4 — Contradictions
# ════════════════════════════════════════════════════
with tab4:
    st.subheader("Potentially contradicting claims")

    if contradictions:
        for c in contradictions:
            st.markdown(f'<div class="finding-box">⚠️ <strong>{c["paper"][:60]}</strong><br/>{c["claim"]}</div>',
                       unsafe_allow_html=True)
    else:
        st.info("No contradictions detected in the current paper set.")
        st.markdown("The papers in this set are largely aligned. Load papers from competing research areas to see contradictions.")