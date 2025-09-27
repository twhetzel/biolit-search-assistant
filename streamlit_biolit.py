import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import streamlit as st
import torch, os

# pick CUDA on NVIDIA boxes; otherwise stick to CPU to avoid MPS edge cases
device = "cuda" if torch.cuda.is_available() else "cpu"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

st.set_page_config(page_title="BioLit Search Assistant", page_icon="🔎", layout="wide")

st.title("🔎 BioLit Search Assistant — NeMo-Ready")

pm = pd.read_csv("sample_pubmed_abstracts.tsv", sep="\t")
on = pd.read_csv("sample_mondo_terms.tsv", sep="\t")

def fuse_pubmed(r):
    text = f"{r['title']}. {r['abstract']}"
    return {"doc_id": r["pmid"], "title": r["title"], "text": text, "source": "pubmed",
            "link": f"https://pubmed.ncbi.nlm.nih.gov/{r['pmid']}/"}

def fuse_ontology(r):
    text = f"{r['label']}. {r['definition']}"
    return {"doc_id": r["id"], "title": r["label"], "text": text, "source": "mondo",
            "link": f"https://monarchinitiative.org/disease/{r['id'].replace(':','_')}"}

corpus = pd.DataFrame([fuse_pubmed(r) for _, r in pm.iterrows()] +
                      [fuse_ontology(r) for _, r in on.iterrows()])

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
vecs = model.encode(corpus["text"].tolist(), normalize_embeddings=True)
nn = NearestNeighbors(metric="cosine").fit(vecs)

query = st.text_input("Enter a biomedical query", "novel Alzheimer’s drug target")

topk = st.number_input("Number of results", min_value=3, max_value=20, value=5, step=1)
topk = int(topk) 

if query:
    qv = model.encode([query], normalize_embeddings=True)
    distances, indices = nn.kneighbors(qv, n_neighbors=topk)
    sims = 1.0 - distances[0]
    out = []
    for rank, (i, sim) in enumerate(zip(indices[0], sims), start=1):
        out.append({
            "rank": rank,
            "source": corpus.iloc[i]["source"],
            "id/pmid": corpus.iloc[i]["doc_id"],
            "title": f"[{corpus.iloc[i]['title']}]({corpus.iloc[i]['link']})",
            "similarity": float(sim),
        })
    st.dataframe(pd.DataFrame(out), use_container_width=True)

st.caption("Runs locally on CPU/Apple Silicon. Swap to NVIDIA NeMo embeddings on GPUs for large-scale corpora.")
