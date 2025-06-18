if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")

    from modules.peat_knowledge_graph import PEATKnowledgeGraph
    from src.import_knowledge_graph import import_knowledge_graph
    from src.server import runServer
    from src.pdf_to_markdown import pdf_to_markdown

    import json
    import numpy as np
    from pathlib import Path

    class NumpyEncoder(json.JSONEncoder):
        """Custom JSON encoder for NumPy types."""

        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NumpyEncoder, self).default(obj)

    print(f"❓ Did you run docker compose up?")

    pdf_to_markdown("documents/")

    complete_semantic_config = {
        "embedding_model": "all-MiniLM-L6-v2",
        "spacy_model": "en_core_web_sm",
        "ner_model": "dbmdz/bert-large-cased-finetuned-conll03-english",
        "classification_model": "facebook/bart-large-mnli",
        "chunk_size": 256,
        "chunk_overlap": 50,
        # Lowered thresholds for comprehensive output
        "min_confidence_threshold": 0.2,
        "entity_confidence_threshold": 0.3,
        "similarity_threshold": 0.5,
        "peat_relevance_threshold": 0.3,
        "evidence_inclusion_threshold": 0.2,
    }

    kg = PEATKnowledgeGraph(
        semantic_config=complete_semantic_config,
        use_full_documents=True,
        enable_simple_fallback=False,
    )

    # Process documents

    summary = kg.process_documents("documents_processed")
    print(f"Processed {summary['documents_processed']} documents")
    print(
        f"Created {summary['total_nodes']} nodes with {summary['total_edges']} relationships"
    )

    export = kg.export_for_llm_integration()

    # Save with custom encoder
    output_dir = Path("peat_analysis")
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / "llm_integration_export.json", "w", encoding="utf-8") as f:
        json.dump(export, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

    print(f"✅ LLM export saved successfully! For basic Webapp")

    print(f"Processed {summary['documents_processed']} documents")
    print(
        f"Created {summary['total_nodes']} nodes with {summary['total_edges']} relationships"
    )

    graph_data = {
        "nodes": [{"id": n, **kg.nx_graph.nodes[n]} for n in kg.nx_graph.nodes()],
        "edges": [
            {"source": u, "target": v, **kg.nx_graph.edges[u, v, k]}
            for u, v, k in kg.nx_graph.edges(keys=True)
        ],
    }

    with open("peat_analysis/knowledge_graph.json", "w") as f:
        json.dump(graph_data, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

    print("✅ Knowledge graph data saved successfully! For Neo4j import")

    import_knowledge_graph(
        "peat_analysis/knowledge_graph.json",
        "bolt://localhost:7687",
        "neo4j",
        "hack25gg",
    )

    print("✅ Knowledge graph data imported into Neo4j successfully!")

    print("✅ started server!")

    runServer()
