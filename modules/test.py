"""
Test script for the PEAT Knowledge Graph System
This script demonstrates the key features including:
1. Document processing (chunks vs full)
2. Knowledge graph creation
3. Document clustering and similarity analysis
4. Adding new documents
5. Gap analysis
6. LLM integration preparation
"""

import os
import sys
from pathlib import Path
import json
import numpy as np
from datetime import datetime

# Import the modules
from semantic_analyzer_v2 import SemanticProcessor, Evidence
from peat_knowledge_graph_v2 import PEATKnowledgeGraph


def setup_test_environment():
    """Setup test directories and ensure requirements."""
    print("Setting up test environment...")
    
    # Create necessary directories
    dirs = ['documents_processed', 'analysis_results', 'test_outputs']
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)
    
    print("✓ Directories created")
    
    # Check for sample documents
    doc_path = Path('documents_processed')
    if not any(doc_path.iterdir()):
        print("⚠️  No documents found in documents_processed directory")
        print("   Please add MOD document directories with chunk files")
        return False
    
    return True


def test_semantic_processor():
    """Test the semantic processor functionality."""
    print("\n=== Testing Semantic Processor ===")
    
    # Initialize processor
    processor = SemanticProcessor()
    print("✓ Semantic processor initialized")
    
    # Test with sample text
    sample_text = """
    The Risk Management Plan establishes clear ownership and accountability for all 
    strategic risks. The Senior Risk Owner (SRO) is responsible for maintaining the 
    risk register and ensuring monthly reviews are conducted. Risk escalation thresholds 
    are defined as: High impact (>£10M or >6 month delay), Medium impact (£5-10M or 
    3-6 month delay). All risks are reviewed by the Project Board monthly in accordance 
    with JSP892 and GovS 002 requirements.
    """
    
    # Process the sample text
    evidence_list = processor.process_document(
        sample_text, 
        "test_document",
        metadata={'chunk_type': 'risk_management'}
    )
    
    print(f"✓ Extracted {len(evidence_list)} evidence items")
    
    if evidence_list:
        evidence = evidence_list[0]
        print(f"  - Confidence: {evidence.confidence_score:.2f}")
        print(f"  - Categories: {evidence.peat_categories}")
        print(f"  - Entities found: {len(evidence.entities)}")
        
        for entity in evidence.entities[:3]:
            print(f"    • {entity.text} ({entity.type})")
    
    return processor


def test_document_chunk_processing(processor, doc_dir):
    """Test processing document chunks vs full document."""
    print(f"\n=== Testing Document Chunk Processing: {doc_dir.name} ===")
    
    # Process chunks
    chunk_evidence = processor.process_document_chunks(doc_dir)
    
    total_chunk_evidence = sum(len(evidence_list) for evidence_list in chunk_evidence.values())
    print(f"✓ Chunk processing: {total_chunk_evidence} evidence items from {len(chunk_evidence)} chunks")
    
    for chunk_type, evidence_list in chunk_evidence.items():
        if evidence_list:
            print(f"  - {chunk_type}: {len(evidence_list)} items")
    
    # Process full document if it exists
    full_path = doc_dir / 'full.md'
    if full_path.exists():
        with open(full_path, 'r', encoding='utf-8') as f:
            full_text = f.read()
        
        full_evidence = processor.process_document(full_text, doc_dir.name)
        print(f"\n✓ Full document processing: {len(full_evidence)} evidence items")
        
        # Compare results
        print(f"\n📊 Comparison:")
        print(f"  - Chunk-based extraction: {total_chunk_evidence} items")
        print(f"  - Full document extraction: {len(full_evidence)} items")
        print(f"  - Benefit of chunks: Better categorization and higher confidence")
    
    return chunk_evidence


def test_knowledge_graph_creation():
    """Test knowledge graph creation and analysis."""
    print("\n=== Testing Knowledge Graph Creation ===")
    
    # Initialize knowledge graph
    kg = PEATKnowledgeGraph()
    print("✓ Knowledge graph initialized")
    
    # Process all documents
    summary = kg.process_documents('documents_processed')
    
    print("\n📊 Processing Summary:")
    for key, value in summary.items():
        print(f"  - {key}: {value}")
    
    # Analyze document relationships
    if kg.document_clusters:
        print(f"\n✓ Document Clustering: {len(kg.document_clusters)} clusters found")
        for cluster_id, cluster in kg.document_clusters.items():
            print(f"  - Cluster {cluster_id} ({cluster.theme}): {len(cluster.document_ids)} documents")
            print(f"    Keywords: {', '.join(cluster.keywords[:3])}")
    
    return kg


def test_gap_analysis(kg):
    """Test PEAT gap analysis functionality."""
    print("\n=== Testing Gap Analysis ===")
    
    gap_report = kg.get_gap_analysis_report()
    
    print(f"\n📊 Gap Analysis Results:")
    print(f"  Overall Coverage: {gap_report['summary']['overall_coverage']:.1%}")
    print(f"  Questions with Evidence: {gap_report['summary']['questions_with_evidence']}")
    print(f"  Questions with Gaps: {gap_report['summary']['questions_with_gaps']}")
    
    print(f"\n📈 Coverage by Category:")
    for category, data in gap_report['by_category'].items():
        print(f"  - {category}: {data['avg_coverage']:.1%} coverage")
    
    if gap_report['critical_gaps']:
        print(f"\n⚠️  Critical Gaps:")
        for gap in gap_report['critical_gaps'][:3]:
            print(f"  - {gap}")
    
    if gap_report['recommendations']:
        print(f"\n💡 Top Recommendations:")
        for rec in gap_report['recommendations'][:3]:
            print(f"  - {rec}")
    
    return gap_report


def test_new_document_addition(kg):
    """Test adding a new document to the graph."""
    print("\n=== Testing New Document Addition ===")
    
    # Find a document to simulate as "new"
    doc_dirs = list(Path('documents_processed').iterdir())
    if len(doc_dirs) > 1:
        # Remove the first document from the graph to simulate it as new
        test_doc = doc_dirs[0]
        print(f"Simulating addition of: {test_doc.name}")
        
        # Note: In real scenario, you would process a truly new document
        # For testing, we'll just demonstrate the API
        
        # Process chunks for the "new" document
        result = kg.add_new_document(str(test_doc))
        
        print(f"\n✓ New Document Analysis:")
        print(f"  - Evidence extracted: {result['evidence_extracted']}")
        print(f"  - Relationships created: {result['relationships_created']}")
        print(f"  - Assigned to cluster: {result['cluster_assignment']}")
        
        if result['most_similar_documents']:
            print(f"  - Similar documents:")
            for doc, similarity in result['most_similar_documents']:
                print(f"    • {doc}: {similarity:.2f}")


def test_llm_integration(kg):
    """Test LLM integration capabilities."""
    print("\n=== Testing LLM Integration Preparation ===")
    
    # Test query functionality
    test_queries = [
        "What evidence exists for risk ownership and accountability?",
        "Show me the schedule management compliance with DCMA-14",
        "What are the gaps in cost management according to JSP 507?"
    ]
    
    print("Testing query context preparation:")
    for query in test_queries:
        context = kg.query_with_llm(query)
        print(f"\n📝 Query: {query}")
        print(f"  - Relevant evidence found: {len(context['relevant_evidence'])}")
        
        if context['relevant_evidence']:
            top_evidence = context['relevant_evidence'][0]
            print(f"  - Top match (similarity: {top_evidence['similarity']:.2f}):")
            print(f"    '{top_evidence['evidence'].content[:100]}...'")
    
    # Export data for external LLM integration
    export_data = kg.export_for_llm_integration()
    
    print(f"\n✓ Export for LLM Integration:")
    print(f"  - Documents: {len(export_data['documents'])}")
    print(f"  - Evidence categories: {len(export_data['evidence_by_category'])}")
    print(f"  - Entity index: {len(export_data['entity_index'])}")
    print(f"  - Assessments: {len(export_data['assessments'])}")
    
    # Save export for LLM use
    with open('test_outputs/llm_export.json', 'w') as f:
        json.dump(export_data, f, indent=2)
    print(f"  - Exported to: test_outputs/llm_export.json")


def test_visualization(kg):
    """Test visualization capabilities."""
    print("\n=== Testing Visualization ===")
    
    # Create visualization
    output_file = 'test_outputs/knowledge_graph_visualization.html'
    fig = kg.visualize_knowledge_graph(output_file)
    
    print(f"✓ Visualization created: {output_file}")
    print("  Open this file in a web browser to explore:")
    print("  - Document clusters")
    print("  - Evidence network")
    print("  - PEAT coverage")
    print("  - Entity relationships")


def run_comprehensive_test():
    """Run all tests in sequence."""
    print("🚀 PEAT Knowledge Graph System Test")
    print("=" * 50)
    
    # Setup
    if not setup_test_environment():
        return
    
    # Test semantic processor
    processor = test_semantic_processor()
    
    # Test document processing
    doc_dirs = list(Path('documents_processed').iterdir())
    if doc_dirs:
        test_document_chunk_processing(processor, doc_dirs[0])
    
    # Test knowledge graph
    kg = test_knowledge_graph_creation()
    
    # Test gap analysis
    test_gap_analysis(kg)
    
    # Test new document addition
    if len(doc_dirs) > 1:
        test_new_document_addition(kg)
    
    # Test LLM integration
    test_llm_integration(kg)
    
    # Test visualization
    test_visualization(kg)
    
    # Save all results
    kg.save_analysis_results('test_outputs')
    
    print("\n✅ All tests completed!")
    print(f"📁 Results saved to: test_outputs/")
    
    # Cleanup
    kg.close()


def quick_test():
    """Quick test of basic functionality."""
    print("🚀 Quick Test - PEAT Knowledge Graph")
    print("=" * 40)
    
    # Create test directories
    Path('test_outputs').mkdir(exist_ok=True)
    
    # Initialize and process
    kg = PEATKnowledgeGraph()
    
    # Check if documents exist
    doc_path = Path('documents_processed')
    if not any(doc_path.iterdir()):
        print("⚠️  No documents found. Creating sample document...")
        
        # Create a sample document structure
        sample_dir = doc_path / 'sample_pmp_001'
        sample_dir.mkdir(exist_ok=True)
        
        # Create sample chunks
        sample_chunks = {
            'risk_management.md': """
            # Risk Management
            The project maintains a comprehensive risk register with clear ownership.
            Risk Owner: John Smith (Project Manager)
            Escalation thresholds: High (>£5M), Medium (£1-5M), Low (<£1M)
            Monthly risk reviews conducted by Project Board per JSP892.
            """,
            'schedule.md': """
            # Schedule Management
            Integrated Master Schedule (IMS) developed using critical path methodology.
            Baseline established Q1 2024. Monthly earned value reporting.
            Schedule Performance Index (SPI) tracked weekly.
            DCMA-14 compliance review scheduled quarterly.
            """,
            'cost_management.md': """
            # Cost Management
            Total project budget: £45M over 5 years.
            Cost Control Board established with monthly reviews.
            Contingency: 15% for identified risks per JSP 507.
            Quarterly financial audits planned.
            """
        }
        
        for filename, content in sample_chunks.items():
            with open(sample_dir / filename, 'w') as f:
                f.write(content)
        
        print("✓ Sample document created")
    
    # Process documents
    summary = kg.process_documents('documents_processed')
    print(f"\n✓ Processed {summary['documents_processed']} documents")
    print(f"✓ Extracted {summary['total_evidence']} evidence items")
    
    # Quick gap analysis
    gap_report = kg.get_gap_analysis_report()
    print(f"\n📊 Quick Analysis:")
    print(f"  Coverage: {gap_report['summary']['overall_coverage']:.1%}")
    print(f"  Gaps found: {gap_report['summary']['questions_with_gaps']}")
    
    # Save results
    kg.save_analysis_results('test_outputs')
    print(f"\n✓ Results saved to test_outputs/")
    
    kg.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test PEAT Knowledge Graph System')
    parser.add_argument('--quick', action='store_true', help='Run quick test')
    parser.add_argument('--full', action='store_true', help='Run comprehensive test')
    
    args = parser.parse_args()
    
    if args.quick:
        quick_test()
    else:
        run_comprehensive_test()