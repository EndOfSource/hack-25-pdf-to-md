# PEAT Knowledge Graph System

An advanced document analysis system for MOD projects that automatically extracts evidence from Project Management Plans (PMPs) and other project documents, builds a knowledge graph, and performs gap analysis against PEAT (Programme Evidence and Assurance Tailoring) requirements.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Usage Guide](#usage-guide)
- [LLM Integration](#llm-integration)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

## Overview

The PEAT Knowledge Graph System addresses the challenge of manually locating evidence in MOD project documentation. It uses advanced NLP techniques to:

- Automatically identify and extract evidence from project documents
- Build a searchable knowledge graph of relationships
- Assess compliance with MOD standards (JSP892, GovS 002, DCMA-14, JSP 507)
- Provide gap analysis and recommendations
- Enable LLM-powered querying for complex compliance questions

## Features

### Core Capabilities

- **Semantic Document Processing**: Extracts evidence using transformer models
- **Chunk-Based Analysis**: Processes documents by category for higher accuracy
- **Entity Recognition**: Identifies MOD-specific entities (DLOD, KUR, committees, etc.)
- **Knowledge Graph Construction**: Builds relationships between documents, evidence, and entities
- **Document Clustering**: Automatically groups similar documents
- **Gap Analysis**: Assesses coverage against PEAT requirements
- **Interactive Visualization**: Creates HTML-based graph visualizations

### Advanced Features

- **New Document Integration**: Add documents to existing graphs
- **Similarity Analysis**: Find related evidence across documents
- **LLM Integration**: Export structured data for AI-powered querying
- **Neo4j Support**: Optional persistent graph storage
- **Confidence Scoring**: Rates evidence relevance and quality

## Installation

### Prerequisites

```bash
# Python 3.8 or higher
python --version

# Install required packages
pip install -r requirements.txt

# If you see errors about missing 'en_core_web_sm', install it with:
python -m spacy download en_core_web_sm
```

### Required Dependencies

```txt
numpy>=1.21.0
scikit-learn>=1.0.0
sentence-transformers>=2.2.0
transformers>=4.25.0
torch>=1.13.0
spacy>=3.4.0
networkx>=2.8
plotly>=5.11.0
neo4j>=5.0.0  # Optional for Neo4j support
langchain>=0.0.200  # For LLM integration
openai>=0.27.0  # For OpenAI integration
```

### Setup Ollama

https://ollama.com/download

### Setup

```bash
# Clone the repository
git clone [repository-url]
cd peat-knowledge-graph

# Install SpaCy language model
python -m spacy download en_core_web_sm

# Create required directories
mkdir -p documents_processed analysis_results
```

## Quick Start

### 1. Prepare Your Documents

Organize your MOD project documents in the following structure:

```
documents_processed/
├── project_name_001/
│   ├── risk_management.md
│   ├── governance.md
│   ├── schedule.md
│   ├── cost_management.md
│   └── full.md (optional)
├── project_name_002/
│   └── ...
```

### 2. Run Basic Analysis

```python
from peat_knowledge_graph import PEATKnowledgeGraph

# Initialize the system
kg = PEATKnowledgeGraph()

# Process all documents
summary = kg.process_documents('documents_processed')

# Generate gap analysis
gap_report = kg.get_gap_analysis_report()
print(f"Overall Coverage: {gap_report['summary']['overall_coverage']:.1%}")

# Save results
kg.save_analysis_results('analysis_results')

# Create visualization
kg.visualize_knowledge_graph('knowledge_graph.html')
```

### 3. Quick Test

```bash
# Run quick test with sample data
python test_peat_system.py --quick

# Run full test suite
python test_peat_system.py --full
```

## Architecture

### System Components

```
┌─────────────────────┐     ┌─────────────────────┐
│  Semantic Analyzer  │────▶│  Knowledge Graph    │
│  - Text Processing  │     │  - Node Creation    │
│  - Entity Extract.  │     │  - Relationship     │
│  - Classification   │     │  - Gap Analysis     │
└─────────────────────┘     └─────────────────────┘
         │                           │
         ▼                           ▼
┌─────────────────────┐     ┌─────────────────────┐
│  Evidence Objects   │     │  Graph Database     │
│  - Content          │     │  - NetworkX         │
│  - Embeddings       │     │  - Neo4j (optional) │
│  - Categories       │     │  - Visualization    │
└─────────────────────┘     └─────────────────────┘
```

### Data Flow

1. Documents are processed into semantic chunks
2. Evidence is extracted with confidence scores
3. Entities are identified and linked
4. Knowledge graph is constructed
5. Gap analysis performed against PEAT questions
6. Results exported for visualization/LLM use

## Usage Guide

### Processing Documents

#### Option 1: Process Document Chunks (Recommended)

```python
from semantic_analyzer import SemanticProcessor

processor = SemanticProcessor()
chunk_evidence = processor.process_document_chunks(doc_dir)

# Returns dict with evidence by chunk type
for chunk_type, evidence_list in chunk_evidence.items():
    print(f"{chunk_type}: {len(evidence_list)} evidence items")
```

#### Option 2: Process Full Documents

```python
evidence_list = processor.process_document(
    text,
    document_id,
    metadata={'source': 'full_document'}
)
```

### Adding New Documents

```python
# Add a new document to existing graph
result = kg.add_new_document('documents_processed/new_project')

print(f"Evidence extracted: {result['evidence_extracted']}")
print(f"Similar documents: {result['most_similar_documents']}")
print(f"Coverage impact: {result['peat_coverage_impact']}")
```

### Analyzing Document Relationships

```python
# Get document clusters
for cluster_id, cluster in kg.document_clusters.items():
    print(f"Cluster {cluster_id} - Theme: {cluster.theme}")
    print(f"Documents: {cluster.document_ids}")
    print(f"Keywords: {cluster.keywords}")
```

## LLM Integration

The system provides multiple ways to integrate with Large Language Models for advanced querying and analysis.

### Method 1: Direct Export for LLM Context

```python
# Export structured data for LLM
export_data = kg.export_for_llm_integration()

# Save for external LLM use
import json
with open('llm_context.json', 'w') as f:
    json.dump(export_data, f, indent=2)

# The export contains:
# - Documents with their evidence
# - Evidence grouped by category
# - Entity index with mentions
# - PEAT assessments
# - Document relationships
```

### Method 2: LangChain Integration

```python
from langchain.llms import OpenAI
from langchain.chains import QAGenerationChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# Prepare evidence for LangChain
documents = []
for evidence in kg.evidence_store.values():
    documents.append({
        'page_content': evidence.content,
        'metadata': {
            'document_id': evidence.document_id,
            'categories': evidence.peat_categories,
            'confidence': evidence.confidence_score
        }
    })

# Create vector store
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents, embeddings)

# Create QA chain
llm = OpenAI(temperature=0)
qa_chain = QAGenerationChain.from_llm(llm, vectorstore)

# Query examples
response = qa_chain.run("""
Based on JSP892 and GovS 002, does the PMP:
- Identify specific roles for managing strategic risks?
- Define escalation thresholds?
Provide evidence and recommendations.
""")
```

### Method 3: Custom LLM Integration with Graph Context

```python
import openai

class PEATQueryEngine:
    def __init__(self, kg, api_key):
        self.kg = kg
        openai.api_key = api_key

    def query_with_context(self, question):
        # Find relevant evidence
        context = self.kg.query_with_llm(question)

        # Build prompt with graph context
        prompt = self._build_prompt(question, context)

        # Query LLM
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        return response.choices[0].message['content']

    def _build_prompt(self, question, context):
        prompt = f"Question: {question}\n\n"
        prompt += "Relevant Evidence:\n"

        for item in context['relevant_evidence'][:5]:
            evidence = item['evidence']
            prompt += f"\n- Document: {evidence.document_id}"
            prompt += f"\n  Content: {evidence.content}"
            prompt += f"\n  Categories: {', '.join(evidence.peat_categories)}"
            prompt += f"\n  Confidence: {evidence.confidence_score:.2f}\n"

        prompt += "\nBased on this evidence, provide a detailed assessment."
        return prompt

    def _get_system_prompt(self):
        return """You are a MOD compliance expert analyzing project documents
        against PEAT requirements. Provide specific, evidence-based assessments
        with clear recommendations. Reference relevant standards (JSP892, GovS 002,
        DCMA-14, JSP 507) where applicable."""

# Usage
engine = PEATQueryEngine(kg, "your-api-key")
result = engine.query_with_context(
    "Assess the risk governance structure against JSP892 requirements"
)
```

### Method 4: Neo4j with LLM Graph Builder

If using Neo4j, integrate with the LLM Graph Builder:

```python
# First, ensure Neo4j is configured
neo4j_config = {
    'uri': 'bolt://localhost:7687',
    'username': 'neo4j',
    'password': 'your-password'
}

kg = PEATKnowledgeGraph(neo4j_config=neo4j_config)
kg.process_documents('documents_processed')

# Then use Neo4j's LLM integration
# See: https://github.com/neo4j-labs/llm-graph-builder

from neo4j_llm import GraphQAChain

# Query using natural language
chain = GraphQAChain(
    driver=kg.driver,
    llm_model="gpt-4"
)

response = chain.query("""
    Find all evidence related to risk escalation thresholds
    and their compliance with JSP892
""")
```

### Example Compliance Queries

The system is optimized for MOD compliance queries like:

```python
# Risk Management Assessment
risk_query = """
According to JSP892 and GovS 002, assess whether the PMP:
- Identifies specific roles for managing strategic risks
- Defines escalation thresholds (cost, time, impact)
- Outlines risk reporting to governance boards
Provide evidence and improvement recommendations.
"""

# Schedule Compliance Check
schedule_query = """
Review this PMP for DCMA-14 compliance:
- Is schedule methodology described with logic links?
- Are baseline and performance tracking defined?
- Is there governance-level schedule monitoring?
Provide specific gaps and remediation steps.
"""

# Cost Management Audit
cost_query = """
Assess financial management against JSP 507:
- Is there a clear Financial Case with funding sources?
- Are cost control structures and owners defined?
- Are cost overrun provisions included?
Rate maturity as Red/Amber/Green with evidence.
"""
```

## API Reference

### SemanticProcessor

```python
processor = SemanticProcessor(config={
    'chunk_size': 256,
    'min_confidence_threshold': 0.7,
    'embedding_model': 'all-MiniLM-L6-v2'
})

# Process document chunks
evidence_dict = processor.process_document_chunks(doc_directory)

# Find similar evidence
similar = processor.find_similar_evidence(evidence, evidence_list, threshold=0.8)

# Cluster evidence
clusters = processor.cluster_evidence(evidence_list, n_clusters=5)
```

### PEATKnowledgeGraph

```python
kg = PEATKnowledgeGraph(
    neo4j_config={'uri': 'bolt://localhost:7687', ...},
    semantic_config={'chunk_size': 512, ...}
)

# Core methods
summary = kg.process_documents(path)
new_doc_result = kg.add_new_document(doc_path)
gap_report = kg.get_gap_analysis_report()
export_data = kg.export_for_llm_integration()

# Analysis methods
similarity_matrix, clusters = kg.analyze_document_relationships(doc_evidence_map)
context = kg.query_with_llm(query_text)

# Visualization
fig = kg.visualize_knowledge_graph(output_file)
kg.save_analysis_results(output_dir)
```

## Examples

### Complete Workflow Example

```python
# 1. Initialize system
from peat_knowledge_graph import PEATKnowledgeGraph
kg = PEATKnowledgeGraph()

# 2. Process documents
summary = kg.process_documents('documents_processed')
print(f"Processed {summary['documents_processed']} documents")
print(f"Created {summary['total_nodes']} nodes with {summary['total_edges']} relationships")

# 3. Analyze gaps
gap_report = kg.get_gap_analysis_report()
for category, data in gap_report['by_category'].items():
    if data['avg_coverage'] < 0.5:
        print(f"⚠️  Low coverage in {category}: {data['avg_coverage']:.1%}")

# 4. Find similar documents
for cluster_id, cluster in kg.document_clusters.items():
    print(f"\nCluster {cluster_id} - {cluster.theme}:")
    for doc_id in cluster.document_ids:
        print(f"  - {doc_id}")

# 5. Export for LLM
export = kg.export_for_llm_integration()
print(f"\nExported {len(export['documents'])} documents for LLM integration")

# 6. Save everything
kg.save_analysis_results('results')
kg.visualize_knowledge_graph('graph.html')
```

## Troubleshooting

### Common Issues

1. **No evidence extracted**

   - Check document format (should be markdown)
   - Ensure documents contain MOD-relevant content
   - Verify chunk files exist in document directories

2. **Low confidence scores**

   - Documents may lack specific PEAT-related content
   - Try adjusting confidence thresholds in config
   - Ensure proper document categorization

3. **Memory issues with large documents**

   - Use chunk-based processing instead of full documents
   - Reduce embedding model size
   - Process documents in batches

4. **LLM integration errors**

   - Verify API keys are set correctly
   - Check rate limits for your LLM provider
   - Ensure export data size is within context limits

5. **SpaCy model not found**
   - If you see an error like `[E050] Can't find model 'en_core_web_sm'`, run:
     ```
     python -m spacy download en_core_web_sm
     ```

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# This will show detailed processing information
kg = PEATKnowledgeGraph()
```

## Contributing

Contributions are welcome! Areas for enhancement:

- Additional MOD-specific entity patterns
- More PEAT assessment questions
- Enhanced visualization options
- Additional LLM integration examples
- Performance optimizations

## License

MIT

## Contact

For questions or support, contact: TEAM AWESOME
