# BioLit Search Assistant — NeMo-Ready

Semantic search across **PubMed abstracts** and **MONDO disease ontology**.

- Runs on Mac M-series (CPU) with Sentence-Transformers. Note, M-series Macs run CPU/Metal, but NeMo needs NVIDIA GPUs.
- Swap to **NVIDIA NeMo** embeddings on GPU for scale.
- Real-time search via Streamlit UI.

## Prerequisites
- Conda

## Quickstart
- Create the conda environment:
```
conda env create -f environment.yml
```

- Activate the environment:
```
conda activate biolit-nemo
```

- Start the web app as:
```
streamlit run streamlit_biolit.py
```

## Data
- `sample_pubmed_abstracts.tsv`
- `sample_mondo_terms.tsv`


## 🚀 Future Directions
This demo is intentionally lightweight for quick reproducibility. Potential next steps include:
- NeMo backend integration — swap from Sentence-Transformers to NVIDIA NeMo embeddings on GPU for large-scale corpora.
- FAISS or cuML indexes — accelerate search across millions of documents using GPU-optimized nearest-neighbor libraries.
- Ontology enrichment — include synonyms, cross-references, and hierarchical structure to improve recall.
- Real-world data connectors — add PubMed API, OBO/OWL ontology parsers, or BigQuery pipelines for larger datasets.
- PyPI packaging — finalize pyproject.toml and implement CLI (biolit-search) and Streamlit app launcher (biolit-search-app) for community use.
- Community demos — polish notebooks, blog tutorials, and add starter kits for hackathons or workshops.