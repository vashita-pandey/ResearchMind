import json
import os
import time
from groq import Groq
from config import GROQ_API_KEY

os.makedirs("data", exist_ok=True)

def extract_claims(paper, groq_client):
    prompt = f"""Analyze this research paper abstract and extract structured information.

Title: {paper['title']}
Abstract: {paper['abstract']}

Return a JSON object with exactly these fields:
{{
  "key_claims": ["claim 1", "claim 2", "claim 3"],
  "methodology": "one sentence describing the method used",
  "limitations": "one sentence describing limitations or gaps",
  "concepts": ["concept1", "concept2", "concept3"],
  "contribution": "one sentence on what this paper contributes"
}}

Return ONLY the JSON object, no other text."""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    
    raw = response.choices[0].message.content.strip()
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        return json.loads(raw)
    except:
        return {
            "key_claims":  ["Could not parse"],
            "methodology": "Unknown",
            "limitations": "Unknown",
            "concepts":    [],
            "contribution":"Unknown"
        }

if __name__ == "__main__":
    # Load local papers
    with open("data/sample_papers.json", "r", encoding="utf-8") as f:
        papers = json.load(f)
    print(f"Loaded {len(papers)} papers")

    groq_client = Groq(api_key=GROQ_API_KEY)
    print("Extracting claims using LLM...")

    for i, paper in enumerate(papers):
        print(f"  Processing {i+1}/{len(papers)}: {paper['title'][:60]}...")
        paper["analysis"] = extract_claims(paper, groq_client)
        time.sleep(1)

    output_path = "data/papers_analyzed.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Saved to {output_path}")
    print("\n── Sample Analysis ──")
    sample = papers[0]
    print(f"Title: {sample['title'][:70]}")
    print(f"Claims: {sample['analysis']['key_claims']}")
    print(f"Concepts: {sample['analysis']['concepts']}")
    print(f"Contribution: {sample['analysis']['contribution']}")