
    
"""
Enhanced PEAT Knowledge Graph Module
Builds and manages a knowledge graph from MOD project documents,
performs gap analysis, and supports LLM integration.
"""

import logging
from typing import List, Dict, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import networkx as nx
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Optional Neo4j support
try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    
# Import the semantic analyzer
from semantic_analyzer import SemanticProcessor, Evidence, Entity

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GraphNode:
    """Represents a node in the knowledge graph."""
    id: str
    type: str  # document, evidence, entity, peat_question
    label: str
    properties: Dict[str, Any]
    embedding: Optional[np.ndarray] = None


@dataclass
class GraphEdge:
    """Represents an edge in the knowledge graph."""
    source_id: str
    target_id: str
    relationship: str
    weight: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AssessmentResult:
    """Results of PEAT assessment for evidence."""
    peat_question_id: str
    evidence_ids: List[str]
    coverage_score: float
    confidence_score: float
    gaps: List[str]
    recommendations: List[str]


@dataclass
class DocumentCluster:
    """Represents a cluster of related documents."""
    cluster_id: int
    document_ids: List[str]
    centroid_embedding: np.ndarray
    theme: str
    keywords: List[str]


class PEATKnowledgeGraph:
    """
    Enhanced knowledge graph system for PEAT evidence assessment.
    """
    
    def __init__(self, neo4j_config: Optional[Dict[str, str]] = None,
                 semantic_config: Optional[Dict[str, Any]] = None):
        """Initialize the knowledge graph system."""
        self.semantic_processor = SemanticProcessor(config=semantic_config)
        
        # Initialize graph storage
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: List[GraphEdge] = []
        self.evidence_store: Dict[str, Evidence] = {}
        self.assessments: Dict[str, AssessmentResult] = {}
        self.document_clusters: Dict[int, DocumentCluster] = {}
        
        # Initialize NetworkX graph
        self.nx_graph = nx.DiGraph()
        
        # Neo4j connection (optional)
        self.driver = None
        self.use_neo4j = False
        if NEO4J_AVAILABLE and neo4j_config:
            self._initialize_neo4j(neo4j_config)
        
        # Load PEAT questions
        self.peat_questions = self._load_peat_questions()
        
        logger.info("PEAT Knowledge Graph initialized")
    
    def _initialize_neo4j(self, config: Dict[str, str]):
        """Initialize Neo4j connection."""
        try:
            self.driver = GraphDatabase.driver(
                config['uri'],
                auth=(config['username'], config['password'])
            )
            self.use_neo4j = True
            logger.info("Connected to Neo4j database")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            self.use_neo4j = False
    
    def _load_peat_questions(self) -> Dict[str, Dict[str, Any]]:
        """Load PEAT assessment questions."""
        # Sample PEAT questions based on your prompts
        return {
            'risk_ownership': {
                'category': 'risk_management',
                'question': 'Does the document identify specific individual roles responsible for managing strategic risks?',
                'keywords': ['risk owner', 'risk manager', 'responsible', 'accountability'],
                'standards': ['JSP892', 'GovS 002']
            },
            'risk_escalation': {
                'category': 'risk_management',
                'question': 'Are escalation thresholds (cost, time, impact) defined?',
                'keywords': ['escalation', 'threshold', 'trigger', 'cost impact', 'time impact'],
                'standards': ['JSP892', 'GovS 002']
            },
            'risk_governance': {
                'category': 'risk_management',
                'question': 'Is there a description of how risks are reported and reviewed by governance boards?',
                'keywords': ['risk reporting', 'governance board', 'risk review', 'oversight'],
                'standards': ['JSP892', 'GovS 002']
            },
            'schedule_methodology': {
                'category': 'schedule_management',
                'question': 'Does the PMP describe how the project schedule is built, including logic links and milestones?',
                'keywords': ['schedule', 'logic links', 'milestone', 'critical path', 'dependencies'],
                'standards': ['DCMA-14', 'GovS 002']
            },
            'schedule_tracking': {
                'category': 'schedule_management',
                'question': 'Are baseline schedules or performance tracking methods defined?',
                'keywords': ['baseline schedule', 'earned value', 'critical path analysis', 'SPI'],
                'standards': ['DCMA-14', 'GovS 002']
            },
            'schedule_monitoring': {
                'category': 'schedule_management',
                'question': 'Is there evidence of governance-level monitoring of schedule risk?',
                'keywords': ['schedule risk', 'recovery plan', 'schedule monitoring', 'governance'],
                'standards': ['DCMA-14', 'GovS 002']
            },
            'financial_case': {
                'category': 'financial_management',
                'question': 'Does the PMP present a clear Financial Case with affordability and funding sources?',
                'keywords': ['financial case', 'affordability', 'funding', 'budget', 'contingency'],
                'standards': ['JSP 507', 'GovS 002']
            },
            'cost_control': {
                'category': 'financial_management',
                'question': 'Is there a defined Cost Control and Reporting structure?',
                'keywords': ['cost control', 'budget owner', 'approval threshold', 'cost reporting'],
                'standards': ['JSP 507', 'GovS 002']
            },
            'cost_overrun': {
                'category': 'financial_management',
                'question': 'Does the plan include provisions for managing cost overruns and financial risk?',
                'keywords': ['cost overrun', 'liability', 'financial risk', 'contingency'],
                'standards': ['JSP 507', 'GovS 002']
            }
        }
    
    def process_documents(self, documents_path: str) -> Dict[str, Any]:
        """Process documents using chunk-based approach."""
        logger.info(f"Processing documents from: {documents_path}")
        
        processed_docs = 0
        total_evidence = 0
        doc_evidence_map = {}
        
        # Process each document directory
        for doc_dir in Path(documents_path).iterdir():
            if doc_dir.is_dir():
                # Process document chunks
                chunk_evidence = self.semantic_processor.process_document_chunks(doc_dir)
                
                # Create document node
                doc_node = GraphNode(
                    id=doc_dir.name,
                    type='document',
                    label=doc_dir.name,
                    properties={'path': str(doc_dir), 'processed_date': datetime.now().isoformat()}
                )
                self._add_node(doc_node)
                
                # Process evidence from all chunks
                all_doc_evidence = []
                for chunk_type, evidence_list in chunk_evidence.items():
                    for evidence in evidence_list:
                        # Add evidence to graph
                        self._add_evidence_to_graph(evidence, doc_node.id)
                        self.evidence_store[evidence.id] = evidence
                        all_doc_evidence.append(evidence)
                        total_evidence += 1
                
                doc_evidence_map[doc_dir.name] = all_doc_evidence
                processed_docs += 1
                logger.info(f"Processed {doc_dir.name}: {len(all_doc_evidence)} total evidence items")
        
        # Build relationships between nodes
        self._build_graph_relationships()
        
        # Analyze document relationships and create clusters
        self.analyze_document_relationships(doc_evidence_map)
        
        # Perform PEAT assessment
        self._assess_evidence_against_peat()
        
        summary = {
            'documents_processed': processed_docs,
            'total_evidence': total_evidence,
            'total_nodes': len(self.nodes),
            'total_edges': len(self.edges),
            'document_clusters': len(self.document_clusters),
            'peat_questions': len(self.peat_questions),
            'assessments_completed': len(self.assessments)
        }
        
        logger.info(f"Knowledge graph built: {summary}")
        return summary
    
    def add_new_document(self, document_path: str) -> Dict[str, Any]:
        """Add a new document to the existing knowledge graph."""
        logger.info(f"Adding new document: {document_path}")
        
        doc_path = Path(document_path)
        if not doc_path.is_dir():
            raise ValueError(f"Document path must be a directory: {document_path}")
        
        # Process document chunks
        chunk_evidence = self.semantic_processor.process_document_chunks(doc_path)
        
        # Create document node
        doc_node = GraphNode(
            id=doc_path.name,
            type='document',
            label=doc_path.name,
            properties={'path': str(doc_path), 'processed_date': datetime.now().isoformat()}
        )
        self._add_node(doc_node)
        
        # Process evidence
        new_evidence = []
        relationships_created = 0
        
        for chunk_type, evidence_list in chunk_evidence.items():
            for evidence in evidence_list:
                # Add evidence to graph
                self._add_evidence_to_graph(evidence, doc_node.id)
                self.evidence_store[evidence.id] = evidence
                new_evidence.append(evidence)
                
                # Find similar existing evidence
                similar_evidence = self._find_similar_evidence_in_graph(evidence, threshold=0.8)
                
                # Create relationships with similar evidence
                for similar, similarity in similar_evidence:
                    edge = GraphEdge(
                        source_id=evidence.id,
                        target_id=similar.id,
                        relationship='similar_to',
                        weight=similarity,
                        properties={'similarity_score': similarity}
                    )
                    self._add_edge(edge)
                    relationships_created += 1
        
        # Update document clusters
        self._update_document_clusters_with_new_doc(doc_path.name, new_evidence)
        
        # Re-run PEAT assessment for new evidence
        self._assess_new_evidence_against_peat(new_evidence)
        
        # Generate insights about the new document
        insights = self._analyze_new_document_position(doc_path.name, new_evidence)
        
        summary = {
            'document_id': doc_path.name,
            'evidence_extracted': len(new_evidence),
            'relationships_created': relationships_created,
            'cluster_assignment': insights['cluster_id'],
            'most_similar_documents': insights['similar_docs'][:3],
            'peat_coverage_impact': insights['coverage_impact']
        }
        
        logger.info(f"New document added: {summary}")
        return summary
    
    def analyze_document_relationships(self, doc_evidence_map: Dict[str, List[Evidence]]) -> Tuple[np.ndarray, Dict[int, List[str]]]:
        """Analyze relationships and clusters between documents."""
        logger.info("Analyzing document relationships and creating clusters")
        
        # Calculate document embeddings from their evidence
        doc_embeddings = {}
        doc_ids = []
        
        for doc_id, evidence_list in doc_evidence_map.items():
            if evidence_list:
                # Calculate mean embedding of all evidence
                embeddings = [e.embedding for e in evidence_list if e.embedding is not None]
                if embeddings:
                    doc_embeddings[doc_id] = np.mean(embeddings, axis=0)
                    doc_ids.append(doc_id)
                    
                    # Update document node with embedding
                    if doc_id in self.nodes:
                        self.nodes[doc_id].embedding = doc_embeddings[doc_id]
        
        if len(doc_embeddings) < 2:
            logger.warning("Not enough documents for clustering")
            return np.array([]), {}
        
        # Create similarity matrix
        embeddings_matrix = np.array([doc_embeddings[doc_id] for doc_id in doc_ids])
        similarity_matrix = cosine_similarity(embeddings_matrix)
        
        # Perform clustering
        n_clusters = min(5, len(doc_ids) // 2)  # Adaptive cluster count
        if n_clusters > 0:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(embeddings_matrix)
            
            # Create document clusters
            for cluster_id in range(n_clusters):
                cluster_doc_ids = [doc_ids[i] for i, c in enumerate(clusters) if c == cluster_id]
                
                # Extract cluster theme from evidence
                cluster_evidence = []
                for doc_id in cluster_doc_ids:
                    cluster_evidence.extend(doc_evidence_map[doc_id])
                
                theme, keywords = self._extract_cluster_theme(cluster_evidence)
                
                self.document_clusters[cluster_id] = DocumentCluster(
                    cluster_id=cluster_id,
                    document_ids=cluster_doc_ids,
                    centroid_embedding=kmeans.cluster_centers_[cluster_id],
                    theme=theme,
                    keywords=keywords
                )
            
            # Create similarity edges between documents
            for i in range(len(doc_ids)):
                for j in range(i + 1, len(doc_ids)):
                    if similarity_matrix[i, j] > 0.7:  # Threshold for significant similarity
                        edge = GraphEdge(
                            source_id=doc_ids[i],
                            target_id=doc_ids[j],
                            relationship='similar_to',
                            weight=similarity_matrix[i, j],
                            properties={'similarity_type': 'document_level'}
                        )
                        self._add_edge(edge)
        
        return similarity_matrix, self.document_clusters
    
    def _add_node(self, node: GraphNode):
        """Add a node to the graph."""
        self.nodes[node.id] = node
        self.nx_graph.add_node(node.id, **node.properties)
        
        if self.use_neo4j:
            self._add_node_to_neo4j(node)
    
    def _add_edge(self, edge: GraphEdge):
        """Add an edge to the graph."""
        self.edges.append(edge)
        self.nx_graph.add_edge(edge.source_id, edge.target_id, 
                              relationship=edge.relationship, weight=edge.weight)
        
        if self.use_neo4j:
            self._add_edge_to_neo4j(edge)
    
    def _add_evidence_to_graph(self, evidence: Evidence, document_id: str):
        """Add evidence and related entities to the knowledge graph."""
        # Create evidence node
        evidence_node = GraphNode(
            id=evidence.id,
            type='evidence',
            label=f"Evidence: {evidence.content[:50]}...",
            properties={
                'content': evidence.content,
                'confidence': evidence.confidence_score,
                'categories': evidence.peat_categories,
                'chunk_type': evidence.chunk_type
            },
            embedding=evidence.embedding
        )
        self._add_node(evidence_node)
        
        # Create relationship to document
        doc_edge = GraphEdge(
            source_id=document_id,
            target_id=evidence.id,
            relationship='contains_evidence',
            weight=evidence.confidence_score
        )
        self._add_edge(doc_edge)
        
        # Add entity nodes and relationships
        for entity in evidence.entities:
            entity_id = f"entity_{entity.type}_{entity.text}"
            
            # Check if entity already exists
            if entity_id not in self.nodes:
                entity_node = GraphNode(
                    id=entity_id,
                    type='entity',
                    label=entity.text,
                    properties={
                        'entity_type': entity.type,
                        'confidence': entity.confidence
                    }
                )
                self._add_node(entity_node)
            
            # Create relationship from evidence to entity
            entity_edge = GraphEdge(
                source_id=evidence.id,
                target_id=entity_id,
                relationship='mentions_entity',
                weight=entity.confidence
            )
            self._add_edge(entity_edge)
    
    def _build_graph_relationships(self):
        """Build relationships between evidence nodes based on similarity."""
        logger.info("Building graph relationships based on semantic similarity")
        
        evidence_list = list(self.evidence_store.values())
        
        for i, evidence1 in enumerate(evidence_list):
            if evidence1.embedding is None:
                continue
                
            # Find similar evidence
            for j in range(i + 1, len(evidence_list)):
                evidence2 = evidence_list[j]
                if evidence2.embedding is None:
                    continue
                
                similarity = cosine_similarity(
                    evidence1.embedding.reshape(1, -1),
                    evidence2.embedding.reshape(1, -1)
                )[0, 0]
                
                if similarity > 0.85:  # High similarity threshold
                    edge = GraphEdge(
                        source_id=evidence1.id,
                        target_id=evidence2.id,
                        relationship='semantically_similar',
                        weight=similarity
                    )
                    self._add_edge(edge)
    
    def _assess_evidence_against_peat(self):
        """Assess collected evidence against PEAT questions."""
        logger.info("Assessing evidence against PEAT requirements")
        
        for question_id, question_data in self.peat_questions.items():
            relevant_evidence = []
            
            # Find evidence relevant to this question
            for evidence in self.evidence_store.values():
                # Check category match
                if question_data['category'] in evidence.peat_categories:
                    relevance_score = self._calculate_evidence_relevance(evidence, question_data)
                    if relevance_score > 0.6:
                        relevant_evidence.append((evidence.id, relevance_score))
            
            # Sort by relevance
            relevant_evidence.sort(key=lambda x: x[1], reverse=True)
            
            # Create assessment result
            if relevant_evidence:
                coverage_score = min(1.0, len(relevant_evidence) * 0.2)  # More evidence = better coverage
                confidence_score = np.mean([score for _, score in relevant_evidence[:5]])  # Top 5 evidence
                gaps = []
                recommendations = []
            else:
                coverage_score = 0.0
                confidence_score = 0.0
                gaps = [f"No evidence found for: {question_data['question']}"]
                recommendations = [f"Need to add content covering: {', '.join(question_data['keywords'][:3])}"]
            
            assessment = AssessmentResult(
                peat_question_id=question_id,
                evidence_ids=[eid for eid, _ in relevant_evidence],
                coverage_score=coverage_score,
                confidence_score=confidence_score,
                gaps=gaps,
                recommendations=recommendations
            )
            
            self.assessments[question_id] = assessment
    
    def _calculate_evidence_relevance(self, evidence: Evidence, question_data: Dict[str, Any]) -> float:
        """Calculate relevance of evidence to a PEAT question."""
        score = 0.0
        
        # Check for keyword matches
        evidence_lower = evidence.content.lower()
        keyword_matches = sum(1 for kw in question_data['keywords'] if kw.lower() in evidence_lower)
        score += min(0.5, keyword_matches * 0.1)
        
        # Check for standard references
        for standard in question_data.get('standards', []):
            if standard in evidence.content:
                score += 0.2
        
        # Category match boost
        if question_data['category'] in evidence.peat_categories:
            score += 0.3
        
        # Entity relevance
        relevant_entities = [e for e in evidence.entities if e.type in ['MOD_DOCUMENT', 'MOD_STANDARD']]
        if relevant_entities:
            score += min(0.2, len(relevant_entities) * 0.05)
        
        return min(1.0, score)
    
    def _find_similar_evidence_in_graph(self, new_evidence: Evidence, threshold: float = 0.8) -> List[Tuple[Evidence, float]]:
        """Find similar evidence already in the graph."""
        similar_evidence = []
        
        if new_evidence.embedding is None:
            return similar_evidence
        
        for existing_evidence in self.evidence_store.values():
            if existing_evidence.id == new_evidence.id or existing_evidence.embedding is None:
                continue
            
            similarity = cosine_similarity(
                new_evidence.embedding.reshape(1, -1),
                existing_evidence.embedding.reshape(1, -1)
            )[0, 0]
            
            if similarity >= threshold:
                similar_evidence.append((existing_evidence, similarity))
        
        # Sort by similarity
        similar_evidence.sort(key=lambda x: x[1], reverse=True)
        return similar_evidence[:10]  # Return top 10
    
    def _update_document_clusters_with_new_doc(self, doc_id: str, evidence_list: List[Evidence]):
        """Update document clusters with a new document."""
        if not self.document_clusters or not evidence_list:
            return
        
        # Calculate document embedding
        embeddings = [e.embedding for e in evidence_list if e.embedding is not None]
        if not embeddings:
            return
        
        doc_embedding = np.mean(embeddings, axis=0)
        
        # Find closest cluster
        min_distance = float('inf')
        closest_cluster = 0
        
        for cluster_id, cluster in self.document_clusters.items():
            distance = np.linalg.norm(doc_embedding - cluster.centroid_embedding)
            if distance < min_distance:
                min_distance = distance
                closest_cluster = cluster_id
        
        # Add document to closest cluster
        self.document_clusters[closest_cluster].document_ids.append(doc_id)
    
    def _assess_new_evidence_against_peat(self, new_evidence: List[Evidence]):
        """Assess new evidence against PEAT questions and update assessments."""
        for question_id, question_data in self.peat_questions.items():
            updated = False
            
            for evidence in new_evidence:
                if question_data['category'] in evidence.peat_categories:
                    relevance_score = self._calculate_evidence_relevance(evidence, question_data)
                    if relevance_score > 0.6:
                        # Update existing assessment
                        if question_id in self.assessments:
                            self.assessments[question_id].evidence_ids.append(evidence.id)
                            updated = True
            
            if updated:
                # Recalculate scores
                assessment = self.assessments[question_id]
                assessment.coverage_score = min(1.0, len(assessment.evidence_ids) * 0.2)
                if assessment.gaps and assessment.coverage_score > 0.5:
                    assessment.gaps = []  # Clear gaps if coverage improved
    
    def _analyze_new_document_position(self, doc_id: str, evidence_list: List[Evidence]) -> Dict[str, Any]:
        """Analyze how a new document fits within the existing graph."""
        insights = {
            'cluster_id': None,
            'similar_docs': [],
            'coverage_impact': {},
            'unique_contributions': []
        }
        
        # Find cluster assignment
        for cluster_id, cluster in self.document_clusters.items():
            if doc_id in cluster.document_ids:
                insights['cluster_id'] = cluster_id
                break
        
        # Find similar documents
        if doc_id in self.nodes and self.nodes[doc_id].embedding is not None:
            doc_embedding = self.nodes[doc_id].embedding
            
            similarities = []
            for other_doc_id, node in self.nodes.items():
                if node.type == 'document' and node.id != doc_id and node.embedding is not None:
                    sim = cosine_similarity(
                        doc_embedding.reshape(1, -1),
                        node.embedding.reshape(1, -1)
                    )[0, 0]
                    similarities.append((other_doc_id, sim))
            
            similarities.sort(key=lambda x: x[1], reverse=True)
            insights['similar_docs'] = [(doc, score) for doc, score in similarities[:5]]
        
        # Calculate coverage impact
        for category in self.peat_questions:
            category_evidence = [e for e in evidence_list if self.peat_questions[category]['category'] in e.peat_categories]
            if category_evidence:
                insights['coverage_impact'][category] = len(category_evidence)
        
        return insights
    
    def _extract_cluster_theme(self, evidence_list: List[Evidence]) -> Tuple[str, List[str]]:
        """Extract theme and keywords for a document cluster."""
        # Count category frequencies
        category_counts = {}
        all_entities = []
        
        for evidence in evidence_list:
            for category in evidence.peat_categories:
                category_counts[category] = category_counts.get(category, 0) + 1
            all_entities.extend([e.text for e in evidence.entities if e.type.startswith('MOD_')])
        
        # Determine primary theme
        if category_counts:
            theme = max(category_counts, key=category_counts.get)
        else:
            theme = "General"
        
        # Extract top keywords (most common entities)
        from collections import Counter
        entity_counts = Counter(all_entities)
        keywords = [entity for entity, _ in entity_counts.most_common(5)]
        
        return theme, keywords
    
    def query_with_llm(self, query: str, llm_interface=None) -> Dict[str, Any]:
        """Query the knowledge graph using an LLM interface."""
        # This is a placeholder for LLM integration
        # In practice, you would use LangChain or similar
        
        # Find relevant evidence based on query
        query_embedding = self.semantic_processor._generate_embedding(query)
        relevant_evidence = []
        
        for evidence in self.evidence_store.values():
            if evidence.embedding is not None:
                similarity = cosine_similarity(
                    query_embedding.reshape(1, -1),
                    evidence.embedding.reshape(1, -1)
                )[0, 0]
                
                if similarity > 0.7:
                    relevant_evidence.append({
                        'evidence': evidence,
                        'similarity': similarity
                    })
        
        # Sort by relevance
        relevant_evidence.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Prepare context for LLM
        context = {
            'query': query,
            'relevant_evidence': relevant_evidence[:10],
            'document_clusters': self.document_clusters,
            'assessments': self.assessments
        }
        
        return context
    
    def get_gap_analysis_report(self) -> Dict[str, Any]:
        """Generate a comprehensive gap analysis report."""
        report = {
            'summary': {
                'total_questions': len(self.peat_questions),
                'questions_with_evidence': 0,
                'questions_with_gaps': 0,
                'overall_coverage': 0.0,
                'overall_confidence': 0.0
            },
            'by_category': {},
            'critical_gaps': [],
            'recommendations': []
        }
        
        # Analyze by category
        category_scores = {}
        
        for question_id, assessment in self.assessments.items():
            category = self.peat_questions[question_id]['category']
            
            if category not in category_scores:
                category_scores[category] = {
                    'coverage_scores': [],
                    'confidence_scores': [],
                    'gaps': []
                }
            
            category_scores[category]['coverage_scores'].append(assessment.coverage_score)
            category_scores[category]['confidence_scores'].append(assessment.confidence_score)
            
            if assessment.coverage_score > 0:
                report['summary']['questions_with_evidence'] += 1
            else:
                report['summary']['questions_with_gaps'] += 1
                report['critical_gaps'].extend(assessment.gaps)
            
            report['recommendations'].extend(assessment.recommendations)
        
        # Calculate category summaries
        for category, scores in category_scores.items():
            report['by_category'][category] = {
                'avg_coverage': np.mean(scores['coverage_scores']) if scores['coverage_scores'] else 0,
                'avg_confidence': np.mean(scores['confidence_scores']) if scores['confidence_scores'] else 0,
                'num_gaps': len(scores['gaps'])
            }
        
        # Calculate overall scores
        all_coverage = [a.coverage_score for a in self.assessments.values()]
        all_confidence = [a.confidence_score for a in self.assessments.values()]
        
        report['summary']['overall_coverage'] = np.mean(all_coverage) if all_coverage else 0
        report['summary']['overall_confidence'] = np.mean(all_confidence) if all_confidence else 0
        
        # Remove duplicate recommendations
        report['recommendations'] = list(set(report['recommendations']))[:10]
        
        return report
    
    def visualize_knowledge_graph(self, output_file: str = 'knowledge_graph.html') -> go.Figure:
        """Create an interactive visualization of the knowledge graph."""
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Document Clusters', 'Evidence Network', 
                          'PEAT Coverage', 'Entity Relationships'),
            specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
                   [{'type': 'bar'}, {'type': 'scatter'}]]
        )
        
        # 1. Document Clusters Visualization
        if self.document_clusters:
            cluster_x = []
            cluster_y = []
            cluster_text = []
            cluster_colors = []
            
            # Use MDS for 2D projection of document embeddings
            from sklearn.manifold import MDS
            doc_embeddings = []
            doc_labels = []
            
            for node_id, node in self.nodes.items():
                if node.type == 'document' and node.embedding is not None:
                    doc_embeddings.append(node.embedding)
                    doc_labels.append(node_id)
            
            if len(doc_embeddings) > 1:
                mds = MDS(n_components=2, random_state=42)
                coords = mds.fit_transform(doc_embeddings)
                
                for i, label in enumerate(doc_labels):
                    # Find cluster
                    cluster_id = None
                    for cid, cluster in self.document_clusters.items():
                        if label in cluster.document_ids:
                            cluster_id = cid
                            break
                    
                    cluster_x.append(coords[i, 0])
                    cluster_y.append(coords[i, 1])
                    cluster_text.append(label)
                    cluster_colors.append(cluster_id if cluster_id is not None else -1)
                
                fig.add_trace(
                    go.Scatter(
                        x=cluster_x, y=cluster_y,
                        mode='markers+text',
                        text=cluster_text,
                        textposition="top center",
                        marker=dict(size=15, color=cluster_colors, colorscale='Viridis'),
                        name='Documents'
                    ),
                    row=1, col=1
                )
        
        # 2. Evidence Network (simplified)
        edge_trace = go.Scatter(
            x=[], y=[],
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        node_trace = go.Scatter(
            x=[], y=[],
            mode='markers',
            hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                size=10,
                colorbar=dict(
                    thickness=15,
                    title='Node Connections',
                    xanchor='left',
                    titleside='right'
                )
            )
        )
        
        # Use spring layout for evidence network
        evidence_graph = nx.Graph()
        for edge in self.edges[:100]:  # Limit to first 100 edges for visualization
            if 'evidence' in edge.source_id and 'evidence' in edge.target_id:
                evidence_graph.add_edge(edge.source_id, edge.target_id, weight=edge.weight)
        
        if len(evidence_graph.nodes()) > 0:
            pos = nx.spring_layout(evidence_graph)
            
            for edge in evidence_graph.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_trace['x'] += tuple([x0, x1, None])
                edge_trace['y'] += tuple([y0, y1, None])
            
            for node in evidence_graph.nodes():
                x, y = pos[node]
                node_trace['x'] += tuple([x])
                node_trace['y'] += tuple([y])
        
        fig.add_trace(edge_trace, row=1, col=2)
        fig.add_trace(node_trace, row=1, col=2)
        
        # 3. PEAT Coverage Bar Chart
        categories = []
        coverage_scores = []
        
        gap_report = self.get_gap_analysis_report()
        for category, data in gap_report['by_category'].items():
            categories.append(category.replace('_', ' ').title())
            coverage_scores.append(data['avg_coverage'] * 100)
        
        fig.add_trace(
            go.Bar(
                x=categories,
                y=coverage_scores,
                marker_color='lightblue',
                name='Coverage %'
            ),
            row=2, col=1
        )
        
        # 4. Entity Relationships (top entities)
        entity_counts = {}
        for node_id, node in self.nodes.items():
            if node.type == 'entity':
                entity_counts[node.label] = len([e for e in self.edges if e.target_id == node_id])
        
        # Get top 20 entities
        top_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)[:20]
        
        if top_entities:
            entity_names = [e[0] for e in top_entities]
            entity_frequencies = [e[1] for e in top_entities]
            
            fig.add_trace(
                go.Scatter(
                    x=entity_frequencies,
                    y=entity_names,
                    mode='markers',
                    marker=dict(size=entity_frequencies, sizemode='diameter', 
                              sizeref=max(entity_frequencies)/50),
                    name='Entity Frequency'
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title_text="PEAT Knowledge Graph Analysis",
            showlegend=False,
            height=1000,
            width=1400
        )
        
        # Save to file
        fig.write_html(output_file)
        logger.info(f"Visualization saved to {output_file}")
        
        return fig
    
    def save_analysis_results(self, output_dir: str):
        """Save analysis results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save gap analysis report
        gap_report = self.get_gap_analysis_report()
        with open(output_path / 'gap_analysis_report.json', 'w') as f:
            json.dump(gap_report, f, indent=2)
        
        # Save assessments
        assessments_data = {}
        for question_id, assessment in self.assessments.items():
            assessments_data[question_id] = {
                'question': self.peat_questions[question_id]['question'],
                'category': self.peat_questions[question_id]['category'],
                'evidence_count': len(assessment.evidence_ids),
                'coverage_score': assessment.coverage_score,
                'confidence_score': assessment.confidence_score,
                'gaps': assessment.gaps,
                'recommendations': assessment.recommendations
            }
        
        with open(output_path / 'peat_assessments.json', 'w') as f:
            json.dump(assessments_data, f, indent=2)
        
        # Save document clusters
        clusters_data = {}
        for cluster_id, cluster in self.document_clusters.items():
            clusters_data[f"cluster_{cluster_id}"] = {
                'theme': cluster.theme,
                'documents': cluster.document_ids,
                'keywords': cluster.keywords
            }
        
        with open(output_path / 'document_clusters.json', 'w') as f:
            json.dump(clusters_data, f, indent=2)
        
        # Save graph statistics
        stats = {
            'total_nodes': len(self.nodes),
            'total_edges': len(self.edges),
            'documents': len([n for n in self.nodes.values() if n.type == 'document']),
            'evidence_items': len([n for n in self.nodes.values() if n.type == 'evidence']),
            'entities': len([n for n in self.nodes.values() if n.type == 'entity']),
            'peat_questions': len(self.peat_questions),
            'assessments': len(self.assessments)
        }
        
        with open(output_path / 'graph_statistics.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Save the graph structure (for later loading)
        graph_data = {
            'nodes': {node_id: {
                'type': node.type,
                'label': node.label,
                'properties': node.properties
            } for node_id, node in self.nodes.items()},
            'edges': [{
                'source': edge.source_id,
                'target': edge.target_id,
                'relationship': edge.relationship,
                'weight': edge.weight
            } for edge in self.edges]
        }
        
        with open(output_path / 'graph_structure.json', 'w') as f:
            json.dump(graph_data, f, indent=2)
        
        logger.info(f"Analysis results saved to {output_path}")
    
    def _add_node_to_neo4j(self, node: GraphNode):
        """Add a node to Neo4j database."""
        if not self.use_neo4j:
            return
        
        with self.driver.session() as session:
            query = """
            MERGE (n:{type} {{id: $id}})
            SET n.label = $label
            SET n += $properties
            """.format(type=node.type.upper())
            
            session.run(query, id=node.id, label=node.label, properties=node.properties)
    
    def _add_edge_to_neo4j(self, edge: GraphEdge):
        """Add an edge to Neo4j database."""
        if not self.use_neo4j:
            return
        
        with self.driver.session() as session:
            query = """
            MATCH (a {{id: $source_id}})
            MATCH (b {{id: $target_id}})
            MERGE (a)-[r:{relationship}]->(b)
            SET r.weight = $weight
            SET r += $properties
            """.format(relationship=edge.relationship.upper())
            
            session.run(query,
                       source_id=edge.source_id,
                       target_id=edge.target_id,
                       weight=edge.weight,
                       properties=edge.properties)
    
    def export_for_llm_integration(self) -> Dict[str, Any]:
        """Export graph data in a format suitable for LLM integration."""
        export_data = {
            'documents': {},
            'evidence_by_category': {},
            'entity_index': {},
            'assessments': {},
            'document_relationships': []
        }
        
        # Export documents with their evidence
        for node_id, node in self.nodes.items():
            if node.type == 'document':
                doc_evidence = []
                for edge in self.edges:
                    if edge.source_id == node_id and edge.relationship == 'contains_evidence':
                        if edge.target_id in self.evidence_store:
                            evidence = self.evidence_store[edge.target_id]
                            doc_evidence.append({
                                'content': evidence.content,
                                'categories': evidence.peat_categories,
                                'confidence': evidence.confidence_score,
                                'chunk_type': evidence.chunk_type
                            })
                
                export_data['documents'][node_id] = {
                    'label': node.label,
                    'evidence': doc_evidence,
                    'properties': node.properties
                }
        
        # Group evidence by category
        for evidence in self.evidence_store.values():
            for category in evidence.peat_categories:
                if category not in export_data['evidence_by_category']:
                    export_data['evidence_by_category'][category] = []
                
                export_data['evidence_by_category'][category].append({
                    'document_id': evidence.document_id,
                    'content': evidence.content,
                    'confidence': evidence.confidence_score
                })
        
        # Create entity index
        for node_id, node in self.nodes.items():
            if node.type == 'entity':
                mentions = []
                for edge in self.edges:
                    if edge.target_id == node_id and edge.relationship == 'mentions_entity':
                        mentions.append(edge.source_id)
                
                export_data['entity_index'][node.label] = {
                    'type': node.properties.get('entity_type', 'UNKNOWN'),
                    'mentions': mentions
                }
        
        # Export assessments
        for question_id, assessment in self.assessments.items():
            export_data['assessments'][question_id] = {
                'question': self.peat_questions[question_id]['question'],
                'category': self.peat_questions[question_id]['category'],
                'standards': self.peat_questions[question_id].get('standards', []),
                'coverage': assessment.coverage_score,
                'confidence': assessment.confidence_score,
                'evidence_count': len(assessment.evidence_ids),
                'has_gaps': len(assessment.gaps) > 0
            }
        
        # Export document relationships
        for edge in self.edges:
            if (edge.relationship == 'similar_to' and 
                edge.source_id in self.nodes and 
                edge.target_id in self.nodes and
                self.nodes[edge.source_id].type == 'document' and
                self.nodes[edge.target_id].type == 'document'):
                
                export_data['document_relationships'].append({
                    'source': edge.source_id,
                    'target': edge.target_id,
                    'similarity': edge.weight
                })
        
        return export_data
    
    def close(self):
        """Close connections and cleanup."""
        if self.use_neo4j and self.driver:
            self.driver.close()
            logger.info("Closed Neo4j connection")


def main():
    """Main function to demonstrate the knowledge graph system."""
    # Initialize the knowledge graph
    kg = PEATKnowledgeGraph()
    
    # Process documents
    documents_path = 'documents_processed'
    summary = kg.process_documents(documents_path)
    
    print("Processing Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Generate gap analysis report
    gap_report = kg.get_gap_analysis_report()
    print(f"\nGap Analysis Summary:")
    print(f"  Overall Coverage: {gap_report['summary']['overall_coverage']:.1%}")
    print(f"  Questions with Evidence: {gap_report['summary']['questions_with_evidence']}")
    print(f"  Questions with Gaps: {gap_report['summary']['questions_with_gaps']}")
    
    # Save results
    kg.save_analysis_results('analysis_results')
    
    # Create visualization
    fig = kg.visualize_knowledge_graph('knowledge_graph.html')
    
    # Close connections
    kg.close()
    
    return kg


if __name__ == "__main__":
    main()