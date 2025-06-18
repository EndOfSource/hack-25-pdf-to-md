from peat_knowledge_graph import PEATKnowledgeGraph

kg = PEATKnowledgeGraph()

# 2. Process documents
summary = kg.process_documents("documents_processed")
print(f"Processed {summary['documents_processed']} documents")
print(
    f"Created {summary['total_nodes']} nodes with {summary['total_edges']} relationships"
)

# 3. Analyze gaps
gap_report = kg.get_gap_analysis_report()
for category, data in gap_report["by_category"].items():
    if data["avg_coverage"] < 0.5:
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
kg.save_analysis_results("results")
kg.visualize_knowledge_graph("graph.html")
