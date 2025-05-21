# Legal OCR QA pipeline

Pipeline for running question-answering over OCR-parsed legal documents using semantic search, vector embeddings, reranking, and heuristics. 

Legal cases often involve hundreds of scanned pages containing requests, orders, and discovery documents. This system enables users (lawyers, analysts, researchers) to **ask free-form legal questions** (e.g. “Which Virginia rule governs authenticity of documents?”) and get **high-confidence answers with page references** in real-time.

It supports:

- OCR ingestion and confidence filtering
- Sentence-level embedding (Sentence-Transformers)
- Page block indexing in Qdrant
- Question answering using CrossEncoder reranking + spaCy heuristics

### The structure:

```markdown
mle-take-home
│
├── app/
│   ├── main.py              ← FastAPI app entry
│   └── legal_qa_pipeline.py ← retriever pipeline
├── src/                     ← ML/NLP modules (loader, retriever, etc.)
├── data/                    ← OCR input data
├── tests/                   ← unit tests
├── generate_eval.py         ← evaluation gen script
├── eval.py                  ← evaluation script
├── requirements.txt 
└── Dockerfile               ← (for container deployment)
```

## Engine design insights

### 1. OCR-Aware Semantic Structuring

Instead of flattening all text, the engine leverages OCR JSON structure:

- Parses page-level blocks with line-level confidence.
- Filters out noisy lines with low confidence scores.
- Preserves natural document layout—vital for legal workflows (e.g., “14th Request for Admission”).

### 2. Page-Level Embedding Granularity

Pages are treated as semantic units:

- Dense vector embeddings are computed at the **page** level.
- Enables fast indexing and citation back to exact page numbers.
- Improves recall and speed compared to sentence- or document-level approaches.

### 3. Multi-Level Retrieval Logic

The engine fuses:
- **Dense retrieval**: Using SentenceTransformer.
- **Neural reranking**: CrossEncoder compares (question, span) pairs.
- **Heuristic biasing**: Legal entity presence, dates, token overlap, and brevity are all scored.

Final answers are selected by combining neural scores with rule-based bonuses.

### 4. Legal-Specific Pattern Matching (via spaCy Matcher)

Uses rule-based span boosting for legal patterns

Patterns are **generalised**, not hardcoded to a single case.

### 5. Span Fusion for Contextual QA

OCR lines often break mid-sentence. 

This engine:
- Merges consecutive sentences up to a token limit (default 50).
- Balances context and conciseness for QA.
- Allows the model to attend across multiple legal clauses.

### 6. Hybrid Boosting Strategy

Each candidate span is scored as:
```markdown
final_score = cross_encoder_score + heuristic_bonus + token_overlap
```

Boost factors include:
- Presence of `DATE`, `PERSON`, `ORG`, `LAW`.
- Rule matches (`Matcher` hit).
- Token overlap with the question.
- Shorter factual spans get preference.

## Requirements

- Python 3.9+
- Models:
  - `sentence-transformers` (e.g., `all-MiniLM-L6-v2`)
  - `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Qdrant (local or remote instance)

Install dependencies:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_trf
```

## Qdrant setup

Qdrant is a high-performance vector search engine used to store and retrieve embeddings efficiently.

This pipeline uses Qdrant to:

- Store sentence embeddings of OCR-parsed legal page blocks
- Perform fast approximate nearest neighbor (ANN) search to retrieve relevant pages given a question


```shell
docker stop qdrant
```

```shell
docker rm qdrant
```

```shell
docker ps -a | grep qdrant
```

```shell
docker run -d \
  --name qdrant \
  -p 6333:6333 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```
---
### Overview of the QA pipeline

```
                      ┌──────────────────────────┐
                      │   OCR JSON Input File    │
                      └────────────┬─────────────┘
                                   │
                                   ▼
                      ┌──────────────────────────┐
                      │   JSONOCRReader          │
                      │ (filter lines by conf.)  │
                      └────────────┬─────────────┘
                                   │
                                   ▼
                      ┌──────────────────────────┐
                      │  OCRPageBlock List       │
                      └───────────┬──────────────┘
                                  │
             ┌────────────────────┴────────────────────┐
             │                                         │
             ▼                                         ▼
 ┌────────────────────────────┐           ┌────────────────────────────┐
 │ SentenceTransformerEmbedder│           │ CrossEncoder Reranker      │
 └────────────────────────────┘           └────────────────────────────┘
             │                                         ▲
             ▼                                         │
 ┌────────────────────────────┐           ┌────────────────────────────┐
 │  PageBlockIndexer (Qdrant) │◄──────────┤ AnswerRetriever            │
 └────────────────────────────┘           │ (fuse, rerank, score spans)│
             │                            └────────────────────────────┘
             ▼                                         ▲
   ┌────────────────────────────┐         ┌────────────────────────────┐
   │ Qdrant Vector Search Engine│         │ spaCy Matcher + Heuristics │
   └────────────────────────────┘         └────────────────────────────┘
             │                                         ▲
             ▼                                         │
     ┌────────────────────────┐       ┌──────────────────────────────────┐
     │Top-k Pages with Vectors│─────► │ Sentence Fusion + Span Selection │
     └────────────────────────┘       └──────────────────────────────────┘
                                                 │
                                                 ▼
                                       ┌────────────────────────┐
                                       │ Final Answer Selection │
                                       └────────────────────────┘
                                                 │
                                                 ▼
                                          Answer + Pages + Score

```


### PageBlockIndexer 

PageBlockIndexer allows semantic search and QA pipelines to query this vector by similarity to a question embedding and retrieve this passage as a match.

```
┌────────────────────────────┐
│   PageBlockIndexer Class   │
└────────────┬───────────────┘
             │
             │
             ▼
  Input: List[OCRPageBlock]
  ┌───────────────────────────────────────────────┐
  │ OCRPageBlock (one per page)                   │
  │  ┌──────────────┐                             │
  │  │ page: 42     │                             │
  │  │ id: 42       │                             │
  │  │ text:        │◄──────────────────────────┐ │
  │  │ "The court finds that..."                │ │
  │  │ lines: [line1, line2, ...]               │ │
  │  └──────────────┘                             │
  └───────────────────────────────────────────────┘
             │
             ▼
┌───────────────────────────────────────────────┐
│ SentenceTransformer: embedder.encode(text)    │
│   → produces dense vector: [0.12, 0.87, ...]  │
└───────────────────────────────────────────────┘
             │
             ▼
┌───────────────────────────────────────────────┐
│ Check if Qdrant collection exists             │
│  └─ If not → create with VectorParams         │
└───────────────────────────────────────────────┘
             │
             ▼
For each OCRPageBlock:
 ┌──────────────────────────────────────────────┐
 │ Construct Qdrant PointStruct:                │
 │   id: page.id                                │
 │   vector: embedded vector                    │
 │   payload: {                                 │
 │     "text": page.text,                       │
 │     "page": page.page                        │
 │   }                                          │
 └──────────────────────────────────────────────┘
             │
             ▼
┌───────────────────────────────────────────────┐
│ Upsert into Qdrant:                           │
│   client.upsert(collection_name, points)      │
└───────────────────────────────────────────────┘
             │
             ▼
Qdrant Collection: `"page_index"`
┌───────────────────────────────────────────────┐
│ id: 42                                        │
│ vector: [0.12, 0.87, ...]                     │
│ payload: {                                    │
│   "text": "The court finds that...",          │
│   "page": 42                                  │
│ }                                             │
└───────────────────────────────────────────────┘

```

### AnswerRetriever

AnswerRetriever retrieves answers from embedded legal OCR pages using dense vector retrieval, sentence fusion, cross-encoder reranking, and entity/matcher-based heuristics.

```
User Query
──────────────
"Which Virginia rule allows unlimited RFAs on document authenticity?"

        │
        ▼
┌───────────────────────────────────────┐
│ SentenceTransformer (embedder.encode) │
│ Embeds the question into a vector     │
└───────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────┐
│ Qdrant Vector Search (client.query)  │
│  → Top-5 most relevant page blocks   │
└──────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────┐
│ Merge top page block texts into 1    │
│ string for span fusion               │
└──────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│ Sentence Fusion:                            │
│  Example fused spans (max_tokens=50):       │
│  ┌ Span 1 ────────────────────────────────┐ │
│  │ "Ms. Heard’s motion was denied..."     │ │
│  └────────────────────────────────────────┘ │
│  ┌ Span 2 ────────────────────────────────┐ │
│  │ "Rule 4:11(e)(2) imposes no limit..."  │ │
│  └────────────────────────────────────────┘ │
│  ...                                        │
└─────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────┐
│ CrossEncoder Reranker                         │
│  → Assigns base relevance score to (Q, Span)  │
└───────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────┐
│ Heuristic Bonus Scorer                        │
│  +0.2 if:                                     │
│    • Span contains DATE entity                │
│    • Span contains PERSON/ORG/GPE/LAW         │
│    • Span matches Matcher rule (e.g. LAWSUIT) │
│    • Span ≤ 30 tokens                         │
│  + Lexical token_overlap(Q, Span)             │
└───────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────┐
│ Score Aggregator                              │
│  final_score = rerank_score + bonus + overlap │
└───────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────┐
│ Select Span with Highest Score                │
│ Return:                                       │
│  • answer text                                │
│  • source pages                               │
│  • confidence score                           │
└───────────────────────────────────────────────┘

Final Output:
─────────────
Answer: Rule 4:11(e)(2) imposes no limit of RFAs related to document authenticity.
Pages: [3, 166, 368]
Score: 4.37
```

---

## Key features

- **Semantic Search**: Retrieves top-matching document pages using sentence embeddings.
- **OCR-Aware Indexing**: Embeds and indexes per-page text blocks from OCR.
- **Span Fusion**: Dynamically fuses sentences to create multi-sentence answer spans.
- **Cross-Encoder Reranking**: Re-ranks candidate spans with BERT-level deep relevance scoring.
- **Legal Entity Matching**: Uses `spaCy` + `Matcher` to bias answers with legal dates, rules, and lawsuit patterns.
- **Qdrant Vector DB**: Efficient retrieval using Qdrant’s high-performance vector engine.

## Synthetic question generation with OpenAI

This script helps generate high-quality, factual legal question/answer (QA) pairs from OCR-parsed documents using OpenAI's `gpt-4o` model.

### What does it do?

- Loads OCR-parsed pages from a JSON file.
- Samples a subset of pages (default = 10).
- For each page, sends the text to OpenAI with a prompt that asks for 3–5 answerable, factual legal questions.
- Parses and saves the output in a structured JSON format.

### Model Used

By default, the script uses:
- `gpt-4o` for contextual question generation.

You can modify the `model` parameter inside the code to use `gpt-3.5-turbo` or others if needed.

---

### Requirements

- Python 3.9+
- OpenAI Python SDK (`pip install openai`)
- set your `OPENAI_API_KEY` as an environment variable.

---

### Running the script

```bash
python generate_openai_qa.py \
  --input "data/ocr/Amber Heard's memorandum.json" \
  --output "data/synthetic/generated_questions_openai.json" \
  --sample 10
```

## Evaluation script for legal QA pipeline

### `eval.py`

This script evaluates the performance of a semantic question answering system built over OCR-parsed legal documents. It uses sentence-transformer embeddings to assess the semantic similarity between predicted and synthetic answers.

- Loads precomputed synthetic QA datasets.
- Compares system predictions to expected answers using **Sentence-BERT** similarity.
- Checks if the correct **page(s)** were retrieved.

#### Evaluation metrics

- Semantic match 
  - cosine similarity ≥ 0.6 using all-MiniLM-L6-v2.

- Page hit 
  - at least one predicted page overlaps with synthetic pages.


The retriever is built in-memory using a Qdrant instance and SentenceTransformer/CrossEncoder models. You can modify the model names or the similarity threshold in the semantic_match() function.

# Productionalisation

```bash
uvicorn main:app --host 0.0.0.0 --port 8080 --workers 4
```

```bash
docker build -t legal-qa-app .
docker run -p 8000:8000 legal-qa-app --json "data/ocr/Amber Heard's memorandum.json"
```

## Examples:

Run some queries:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Which Virginia discovery rule places no numerical cap on Requests for Admission about document authenticity?"}'
```