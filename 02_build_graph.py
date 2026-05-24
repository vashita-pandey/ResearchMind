import json
import numpy as np
import networkx as nx
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os

os.makedirs("outputs", exist_ok=True)

# ── 1. Load analyzed papers ────────────────────────────────────
with open("data/papers_analyzed.json", "r", encoding="utf-8") as f:
    papers = json.load(f)
print(f"Loaded {len(papers)} papers")

# ── 2. Generate embeddings ─────────────────────────────────────
print("Generating embeddings (first run downloads ~90MB model)...")
model = SentenceTransformer("all-MiniLM-L6-v2")

texts = [f"{p['title']}. {p['abstract']}" for p in papers]
embeddings = model.encode(texts, show_progress_bar=True)
print(f"✓ Generated {len(embeddings)} embeddings")

# ── 3. Build similarity matrix ─────────────────────────────────
sim_matrix = cosine_similarity(embeddings)
print("✓ Similarity matrix computed")

# ── 4. Build NetworkX graph ────────────────────────────────────
G = nx.Graph()

# Add paper nodes
for i, paper in enumerate(papers):
    G.add_node(
        paper["id"],
        title=paper["title"],
        year=paper["published"],
        authors=", ".join(paper["authors"]),
        contribution=paper["analysis"]["contribution"],
        concepts=", ".join(paper["analysis"]["concepts"]),
        node_type="paper"
    )

# Add concept nodes and edges
all_concepts = {}
for paper in papers:
    for concept in paper["analysis"]["concepts"]:
        concept = concept.lower().strip()
        if concept not in all_concepts:
            all_concepts[concept] = []
        all_concepts[concept].append(paper["id"])

for concept, paper_ids in all_concepts.items():
    if len(paper_ids) >= 2:  # only shared concepts
        G.add_node(concept, node_type="concept", title=concept)
        for pid in paper_ids:
            G.add_edge(pid, concept, edge_type="has_concept", weight=0.5)

# Add similarity edges between papers
SIMILARITY_THRESHOLD = 0.5
for i in range(len(papers)):
    for j in range(i+1, len(papers)):
        sim = float(sim_matrix[i][j])
        if sim >= SIMILARITY_THRESHOLD:
            G.add_edge(
                papers[i]["id"],
                papers[j]["id"],
                edge_type="similar",
                weight=round(sim, 3),
                similarity=round(sim, 3)
            )

print(f"✓ Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# ── 5. Detect contradictions ───────────────────────────────────
print("\nChecking for contradictions...")
contradictions = []

keywords_pos = ["outperforms", "better", "improves", "superior", "exceeds"]
keywords_neg = ["fails", "limitation", "mirage", "illusion", "misleading"]

for paper in papers:
    claims = paper["analysis"]["key_claims"]
    for claim in claims:
        claim_lower = claim.lower()
        if any(k in claim_lower for k in keywords_neg):
            contradictions.append({
                "paper": paper["title"],
                "claim": claim,
                "type":  "challenges_mainstream"
            })

if contradictions:
    print(f"⚠ Found {len(contradictions)} potentially contradicting claims:")
    for c in contradictions[:3]:
        print(f"  [{c['paper'][:40]}] {c['claim'][:80]}")
else:
    print("No contradictions detected")

# ── 6. Save graph data ─────────────────────────────────────────
graph_data = {
    "nodes": [],
    "edges": []
}

for node_id, attrs in G.nodes(data=True):
    graph_data["nodes"].append({"id": node_id, **attrs})

for u, v, attrs in G.edges(data=True):
    graph_data["edges"].append({"source": u, "target": v, **attrs})

with open("outputs/graph_data.json", "w") as f:
    json.dump(graph_data, f, indent=2)

with open("outputs/contradictions.json", "w") as f:
    json.dump(contradictions, f, indent=2)

print(f"\n✓ Graph saved to outputs/graph_data.json")
print(f"✓ {len(graph_data['nodes'])} nodes, {len(graph_data['edges'])} edges")

# ── 7. Print most connected papers ────────────────────────────
print("\n── Most connected papers ──")
paper_nodes = [(n, d) for n, d in G.degree() 
               if G.nodes[n].get("node_type") == "paper"]
paper_nodes.sort(key=lambda x: x[1], reverse=True)
for node_id, degree in paper_nodes[:5]:
    title = G.nodes[node_id]["title"][:55]
    print(f"  {degree} connections — {title}")