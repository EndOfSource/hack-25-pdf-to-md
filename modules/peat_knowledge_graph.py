"""
PEAT Knowledge Graph System
A comprehensive semantic knowledge graph for MOD project evidence assessment
"""

import os
import json
import logging
import networkx as nx
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any, Set
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np
from pathlib import Path

# Import the existing semantic analyzer
import sys

from semantic_analyzer import SemanticProcessor, Evidence, Entity

# Knowledge graph specific imports
from neo4j import GraphDatabase
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class KnowledgeNode:
    """Represents a node in the knowledge graph."""
    id: str
    type: str  # Evidence, Entity, Category, Document, Question
    properties: Dict[str, Any]
    embeddings: Optional[np.ndarray] = None


@dataclass
class KnowledgeEdge:
    """Represents an edge in the knowledge graph."""
    source: str
    target: str
    relationship: str
    properties: Dict[str, Any]
    weight: float = 1.0


@dataclass
class PEATQuestion:
    """Represents a PEAT assessment question."""
    id: str
    category: str
    text: str
    keywords: List[str]
    weight: float
    required_evidence_types: List[str]


@dataclass
class EvidenceAssessment:
    """Assessment result for evidence against PEAT questions."""
    question_id: str
    evidence_id: str
    relevance_score: float
    confidence_score: float
    explanation: str
    supporting_entities: List[str]
    gaps_identified: List[str]


class PEATKnowledgeGraph:
    """
    Main knowledge graph system for PEAT evidence assessment.
    Integrates semantic analysis with graph-based reasoning.
    """
    
    def __init__(self, neo4j_uri: Optional[str] = None, 
                 neo4j_user: Optional[str] = None,
                 neo4j_password: Optional[str] = None):
        """
        Initialize the PEAT Knowledge Graph system.
        
        Args:
            neo4j_uri: Neo4j database URI (optional, uses NetworkX if not provided)
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
        """
        # Initialize semantic processor
        self.semantic_processor = SemanticProcessor(
            categories_file='/modules/embedding_sematic_2/peat_categories_config.json',
            weights_file='/modules/embedding_sematic_2/peat_weights_config.json'
        )
        
        # Initialize graph storage
        self.use_neo4j = neo4j_uri is not None
        if self.use_neo4j:
            self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
            logger.info("Connected to Neo4j database")
        else:
            self.graph = nx.MultiDiGraph()
            logger.info("Using NetworkX in-memory graph")
        
        # Storage for processed data
        self.nodes: Dict[str, KnowledgeNode] = {}
        self.edges: List[KnowledgeEdge] = []
        self.evidence_store: Dict[str, Evidence] = {}
        self.peat_questions: Dict[str, PEATQuestion] = {}
        
        # Load PEAT questions
        self._load_peat_questions()
        
        # Analytics storage
        self.assessments: List[EvidenceAssessment] = []
        self.gap_analysis: Dict[str, List[str]] = {}
        
    def _load_peat_questions(self):
        """Load PEAT assessment questions from configuration."""
        # Define standard PEAT questions for each category
        peat_questions_config = {
            'governance': [
                {
                    'id': 'GOV_001',
                    'text': 'Is there evidence of a properly constituted project board with clear terms of reference?',
                    'keywords': ['project board', 'terms of reference', 'TOR', 'governance structure'],
                    'weight': 2.0,
                    'required_evidence_types': ['governance_document', 'meeting_minutes', 'charter']
                },
                {
                    'id': 'GOV_002', 
                    'text': 'Has a Senior Responsible Owner (SRO) been appointed with clear accountability?',
                    'keywords': ['SRO', 'senior responsible owner', 'accountability', 'authority'],
                    'weight': 2.0,
                    'required_evidence_types': ['appointment_letter', 'role_description', 'delegation']
                },
                {
                    'id': 'GOV_003',
                    'text': 'Are approval gates and decision points clearly defined and documented?',
                    'keywords': ['approval gates', 'decision points', 'gate reviews', 'approval process'],
                    'weight': 1.8,
                    'required_evidence_types': ['process_document', 'gate_criteria', 'approval_matrix']
                }
            ],
            'requirements': [
                {
                    'id': 'REQ_001',
                    'text': 'Are Key User Requirements (KUR) clearly documented and validated?',
                    'keywords': ['KUR', 'key user requirements', 'user requirements document', 'URD'],
                    'weight': 2.0,
                    'required_evidence_types': ['requirements_document', 'validation_report', 'stakeholder_sign_off']
                },
                {
                    'id': 'REQ_002',
                    'text': 'Is there evidence of requirements traceability throughout the project lifecycle?',
                    'keywords': ['requirements traceability', 'traceability matrix', 'requirement ID'],
                    'weight': 1.8,
                    'required_evidence_types': ['traceability_matrix', 'requirements_database', 'change_log']
                }
            ],
            'risk_management': [
                {
                    'id': 'RISK_001',
                    'text': 'Is there a comprehensive risk register with identified mitigation strategies?',
                    'keywords': ['risk register', 'risk assessment', 'mitigation', 'risk management'],
                    'weight': 2.0,
                    'required_evidence_types': ['risk_register', 'risk_assessment', 'mitigation_plan']
                },
                {
                    'id': 'RISK_002',
                    'text': 'Has a safety case been developed and approved where required?',
                    'keywords': ['safety case', 'HAZARD log', 'safety assessment', 'SEMP'],
                    'weight': 1.8,
                    'required_evidence_types': ['safety_case', 'hazard_log', 'safety_plan']
                }
            ],
            'schedule': [
                {
                    'id': 'SCH_001',
                    'text': 'Is there an Integrated Master Schedule (IMS) with critical path analysis?',
                    'keywords': ['IMS', 'integrated master schedule', 'critical path', 'project plan'],
                    'weight': 2.0,
                    'required_evidence_types': ['master_schedule', 'critical_path_analysis', 'milestone_plan']
                }
            ],
            'resources': [
                {
                    'id': 'RES_001',
                    'text': 'Are Suitably Qualified and Experienced Personnel (SQEP) identified and allocated?',
                    'keywords': ['SQEP', 'suitably qualified experienced personnel', 'resource plan', 'skills matrix'],
                    'weight': 2.0,
                    'required_evidence_types': ['resource_plan', 'skills_matrix', 'cv_evidence']
                }
            ]
        }
        
        # Convert to PEATQuestion objects
        for category, questions in peat_questions_config.items():
            for q_config in questions:
                question = PEATQuestion(
                    id=q_config['id'],
                    category=category,
                    text=q_config['text'],
                    keywords=q_config['keywords'],
                    weight=q_config['weight'],
                    required_evidence_types=q_config['required_evidence_types']
                )
                self.peat_questions[question.id] = question
        
        logger.info(f"Loaded {len(self.peat_questions)} PEAT questions")
    
    def process_documents(self, documents_path: str) -> Dict[str, Any]:
        """
        Process all documents in the given path and build the knowledge graph.
        
        Args:
            documents_path: Path to directory containing processed documents
            
        Returns:
            Processing summary with statistics
        """
        logger.info(f"Processing documents from: {documents_path}")
        
        processed_docs = 0
        total_evidence = 0
        
        # Process each document directory
        for doc_dir in Path(documents_path).iterdir():
            if doc_dir.is_dir():
                # Process the full document
                full_doc_path = doc_dir / 'full.md'
                if full_doc_path.exists():
                    with open(full_doc_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Extract evidence using semantic processor
                    evidence_list = self.semantic_processor.process_document(
                        content, 
                        doc_dir.name,
                        metadata={'source_path': str(doc_dir)}
                    )
                    
                    # Add evidence to knowledge graph
                    for evidence in evidence_list:
                        self._add_evidence_to_graph(evidence)
                        self.evidence_store[evidence.document_id + '_' + str(hash(evidence.content))] = evidence
                    
                    processed_docs += 1
                    total_evidence += len(evidence_list)
                    
                    logger.info(f"Processed {doc_dir.name}: {len(evidence_list)} evidence items")
        
        # Build relationships between nodes
        self._build_graph_relationships()
        
        # Perform PEAT assessment
        self._assess_evidence_against_peat()
        
        summary = {
            'documents_processed': processed_docs,
            'total_evidence': total_evidence,
            'total_nodes': len(self.nodes),
            'total_edges': len(self.edges),
            'peat_questions': len(self.peat_questions),
            'assessments_completed': len(self.assessments)
        }
        
        logger.info(f"Knowledge graph built: {summary}")
        return summary
    
    def _add_evidence_to_graph(self, evidence: Evidence):
        """Add evidence and related entities to the knowledge graph."""
        # Create evidence node
        evidence_id = f"evidence_{evidence.document_id}_{hash(evidence.content)}"
        evidence_node = KnowledgeNode(
            id=evidence_id,
            type='Evidence',
            properties={
                'content': evidence.content,
                'document_id': evidence.document_id,
                'confidence_score': evidence.confidence_score,
                'categories': evidence.peat_categories,
                'page_number': evidence.page_number,
                'timestamp': datetime.now().isoformat()
            },
            embeddings=evidence.embedding
        )
        self.nodes[evidence_id] = evidence_node
        
        # Create document node if not exists
        doc_id = f"document_{evidence.document_id}"
        if doc_id not in self.nodes:
            doc_node = KnowledgeNode(
                id=doc_id,
                type='Document',
                properties={
                    'document_id': evidence.document_id,
                    'source_type': self._infer_document_type(evidence.document_id)
                }
            )
            self.nodes[doc_id] = doc_node
        
        # Create edge from document to evidence
        doc_evidence_edge = KnowledgeEdge(
            source=doc_id,
            target=evidence_id,
            relationship='CONTAINS',
            properties={'extraction_confidence': evidence.confidence_score}
        )
        self.edges.append(doc_evidence_edge)
        
        # Create entity nodes and relationships
        for entity in evidence.entities:
            entity_id = f"entity_{hash(entity.text + entity.type)}"
            if entity_id not in self.nodes:
                entity_node = KnowledgeNode(
                    id=entity_id,
                    type='Entity',
                    properties={
                        'text': entity.text,
                        'entity_type': entity.type,
                        'confidence': entity.confidence
                    }
                )
                self.nodes[entity_id] = entity_node
            
            # Create edge from evidence to entity
            entity_edge = KnowledgeEdge(
                source=evidence_id,
                target=entity_id,
                relationship='MENTIONS',
                properties={
                    'position': f"{entity.start_pos}-{entity.end_pos}",
                    'confidence': entity.confidence
                }
            )
            self.edges.append(entity_edge)
        
        # Create category nodes and relationships
        for category in evidence.peat_categories:
            category_id = f"category_{category}"
            if category_id not in self.nodes:
                category_node = KnowledgeNode(
                    id=category_id,
                    type='Category',
                    properties={
                        'name': category,
                        'keywords': self.semantic_processor.peat_categories.get(category, [])
                    }
                )
                self.nodes[category_id] = category_node
            
            # Create edge from evidence to category
            category_edge = KnowledgeEdge(
                source=evidence_id,
                target=category_id,
                relationship='BELONGS_TO',
                properties={'relevance_score': evidence.confidence_score}
            )
            self.edges.append(category_edge)
    
    def _infer_document_type(self, document_id: str) -> str:
        """Infer document type from document ID."""
        doc_id_lower = document_id.lower()
        if 'obc' in doc_id_lower:
            return 'Outline Business Case'
        elif 'fbc' in doc_id_lower:
            return 'Full Business Case'
        elif 'pmp' in doc_id_lower:
            return 'Project Management Plan'
        elif 'outletter' in doc_id_lower:
            return 'Outletter'
        elif 'rn' in doc_id_lower:
            return 'Review Note'
        else:
            return 'Unknown'
    
    def _build_graph_relationships(self):
        """Build additional relationships between nodes based on semantic similarity."""
        logger.info("Building semantic relationships...")
        
        # Find similar evidence items
        evidence_nodes = [node for node in self.nodes.values() if node.type == 'Evidence']
        
        for i, node1 in enumerate(evidence_nodes):
            for node2 in evidence_nodes[i+1:]:
                if node1.embeddings is not None and node2.embeddings is not None:
                    # Calculate cosine similarity
                    similarity = np.dot(node1.embeddings, node2.embeddings) / (
                        np.linalg.norm(node1.embeddings) * np.linalg.norm(node2.embeddings)
                    )
                    
                    # Create relationship if similarity is high enough
                    if similarity > 0.7:
                        similarity_edge = KnowledgeEdge(
                            source=node1.id,
                            target=node2.id,
                            relationship='SIMILAR_TO',
                            properties={'similarity_score': float(similarity)},
                            weight=similarity
                        )
                        self.edges.append(similarity_edge)
        
        # Build entity co-occurrence relationships
        self._build_entity_cooccurrence()
        
        logger.info(f"Built {len(self.edges)} total relationships")
    
    def _build_entity_cooccurrence(self):
        """Build relationships between entities that co-occur in evidence."""
        entity_cooccurrence = {}
        
        # Track which entities appear together
        for evidence_id, evidence in self.evidence_store.items():
            entities_in_evidence = [f"entity_{hash(e.text + e.type)}" for e in evidence.entities]
            
            for i, entity1 in enumerate(entities_in_evidence):
                for entity2 in entities_in_evidence[i+1:]:
                    pair = tuple(sorted([entity1, entity2]))
                    entity_cooccurrence[pair] = entity_cooccurrence.get(pair, 0) + 1
        
        # Create co-occurrence edges
        for (entity1, entity2), count in entity_cooccurrence.items():
            if count >= 2:  # Threshold for co-occurrence
                cooccurrence_edge = KnowledgeEdge(
                    source=entity1,
                    target=entity2,
                    relationship='CO_OCCURS_WITH',
                    properties={'cooccurrence_count': count},
                    weight=min(count / 5.0, 1.0)  # Normalize weight
                )
                self.edges.append(cooccurrence_edge)
    
    def _assess_evidence_against_peat(self):
        """Assess all evidence against PEAT questions."""
        logger.info("Performing PEAT assessment...")
        
        for question_id, question in self.peat_questions.items():
            # Find relevant evidence for this question
            relevant_evidence = self.semantic_processor.find_evidence_for_peat_question(
                question.text,
                list(self.evidence_store.values())
            )
            
            for evidence, relevance_score in relevant_evidence[:5]:  # Top 5 most relevant
                # Calculate confidence based on multiple factors
                confidence = self._calculate_assessment_confidence(evidence, question)
                
                # Generate explanation
                explanation = self._generate_assessment_explanation(evidence, question, relevance_score)
                
                # Identify supporting entities
                supporting_entities = [
                    entity.text for entity in evidence.entities 
                    if any(keyword.lower() in entity.text.lower() for keyword in question.keywords)
                ]
                
                # Create assessment
                assessment = EvidenceAssessment(
                    question_id=question_id,
                    evidence_id=f"evidence_{evidence.document_id}_{hash(evidence.content)}",
                    relevance_score=relevance_score,
                    confidence_score=confidence,
                    explanation=explanation,
                    supporting_entities=supporting_entities,
                    gaps_identified=[]
                )
                
                self.assessments.append(assessment)
        
        # Perform gap analysis
        self._perform_gap_analysis()
        
        logger.info(f"Completed {len(self.assessments)} evidence assessments")
    
    def _calculate_assessment_confidence(self, evidence: Evidence, question: PEATQuestion) -> float:
        """Calculate confidence score for evidence assessment."""
        factors = []
        
        # Base evidence confidence
        factors.append(evidence.confidence_score * 0.4)
        
        # Keyword match score
        keyword_matches = sum(
            1 for keyword in question.keywords 
            if keyword.lower() in evidence.content.lower()
        )
        keyword_score = min(keyword_matches / len(question.keywords), 1.0)
        factors.append(keyword_score * 0.3)
        
        # Category alignment
        category_match = 1.0 if question.category in evidence.peat_categories else 0.0
        factors.append(category_match * 0.2)
        
        # Entity relevance
        relevant_entities = [
            entity for entity in evidence.entities
            if any(keyword.lower() in entity.text.lower() for keyword in question.keywords)
        ]
        entity_score = min(len(relevant_entities) / 3.0, 1.0)
        factors.append(entity_score * 0.1)
        
        return sum(factors)
    
    def _generate_assessment_explanation(self, evidence: Evidence, question: PEATQuestion, 
                                       relevance_score: float) -> str:
        """Generate human-readable explanation for assessment."""
        explanation_parts = []
        
        # Relevance explanation
        if relevance_score > 0.8:
            explanation_parts.append("High semantic similarity to question")
        elif relevance_score > 0.6:
            explanation_parts.append("Moderate semantic similarity to question")
        else:
            explanation_parts.append("Low semantic similarity to question")
        
        # Keyword matches
        keyword_matches = [
            keyword for keyword in question.keywords 
            if keyword.lower() in evidence.content.lower()
        ]
        if keyword_matches:
            explanation_parts.append(f"Contains key terms: {', '.join(keyword_matches)}")
        
        # Category alignment
        if question.category in evidence.peat_categories:
            explanation_parts.append(f"Classified under {question.category} category")
        
        # Entity mentions
        relevant_entities = [
            entity.text for entity in evidence.entities
            if any(keyword.lower() in entity.text.lower() for keyword in question.keywords)
        ]
        if relevant_entities:
            explanation_parts.append(f"Mentions relevant entities: {', '.join(relevant_entities)}")
        
        return ". ".join(explanation_parts)
    
    def _perform_gap_analysis(self):
        """Identify gaps in evidence coverage for PEAT questions."""
        logger.info("Performing gap analysis...")
        
        for question_id, question in self.peat_questions.items():
            # Find assessments for this question
            question_assessments = [
                a for a in self.assessments 
                if a.question_id == question_id and a.confidence_score > 0.6
            ]
            
            gaps = []
            
            # Check if we have high-confidence evidence
            if not question_assessments:
                gaps.append("No relevant evidence found")
            elif max(a.confidence_score for a in question_assessments) < 0.7:
                gaps.append("Low confidence evidence only")
            
            # Check for required evidence types
            evidence_types_found = set()
            for assessment in question_assessments:
                evidence = self.evidence_store.get(assessment.evidence_id)
                if evidence:
                    # Infer evidence type from content/metadata
                    evidence_type = self._infer_evidence_type(evidence)
                    evidence_types_found.add(evidence_type)
            
            missing_types = set(question.required_evidence_types) - evidence_types_found
            if missing_types:
                gaps.append(f"Missing evidence types: {', '.join(missing_types)}")
            
            self.gap_analysis[question_id] = gaps
        
        logger.info("Gap analysis completed")
    
    def _infer_evidence_type(self, evidence: Evidence) -> str:
        """Infer the type of evidence from its content."""
        content_lower = evidence.content.lower()
        
        if any(term in content_lower for term in ['minutes', 'meeting', 'agenda']):
            return 'meeting_minutes'
        elif any(term in content_lower for term in ['plan', 'strategy', 'approach']):
            return 'planning_document'
        elif any(term in content_lower for term in ['register', 'log', 'list']):
            return 'register_document'
        elif any(term in content_lower for term in ['assessment', 'analysis', 'evaluation']):
            return 'assessment_document'
        elif any(term in content_lower for term in ['approval', 'decision', 'authorization']):
            return 'approval_document'
        else:
            return 'general_document'
    
    def get_evidence_for_question(self, question_id: str, 
                                min_confidence: float = 0.6) -> List[Dict[str, Any]]:
        """
        Get evidence items relevant to a specific PEAT question.
        
        Args:
            question_id: PEAT question identifier
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of evidence items with assessment details
        """
        question_assessments = [
            a for a in self.assessments 
            if a.question_id == question_id and a.confidence_score >= min_confidence
        ]
        
        results = []
        for assessment in sorted(question_assessments, key=lambda x: x.confidence_score, reverse=True):
            evidence = self.evidence_store.get(assessment.evidence_id)
            if evidence:
                result = {
                    'evidence_content': evidence.content[:500] + '...' if len(evidence.content) > 500 else evidence.content,
                    'document_id': evidence.document_id,
                    'confidence_score': assessment.confidence_score,
                    'relevance_score': assessment.relevance_score,
                    'explanation': assessment.explanation,
                    'supporting_entities': assessment.supporting_entities,
                    'categories': evidence.peat_categories
                }
                results.append(result)
        
        return results
    
    def get_gap_analysis_report(self) -> Dict[str, Any]:
        """Generate comprehensive gap analysis report."""
        report = {
            'summary': {
                'total_questions': len(self.peat_questions),
                'questions_with_evidence': 0,
                'questions_with_gaps': 0,
                'overall_coverage': 0.0
            },
            'by_category': {},
            'detailed_gaps': self.gap_analysis,
            'recommendations': []
        }
        
        # Calculate summary statistics
        questions_with_evidence = 0
        for question_id in self.peat_questions:
            question_assessments = [
                a for a in self.assessments 
                if a.question_id == question_id and a.confidence_score > 0.6
            ]
            if question_assessments:
                questions_with_evidence += 1
        
        report['summary']['questions_with_evidence'] = questions_with_evidence
        report['summary']['questions_with_gaps'] = len(self.peat_questions) - questions_with_evidence
        report['summary']['overall_coverage'] = questions_with_evidence / len(self.peat_questions)
        
        # Analyze by category
        for category in set(q.category for q in self.peat_questions.values()):
            category_questions = [q for q in self.peat_questions.values() if q.category == category]
            category_covered = 0
            
            for question in category_questions:
                question_assessments = [
                    a for a in self.assessments 
                    if a.question_id == question.id and a.confidence_score > 0.6
                ]
                if question_assessments:
                    category_covered += 1
            
            report['by_category'][category] = {
                'total_questions': len(category_questions),
                'covered_questions': category_covered,
                'coverage_percentage': category_covered / len(category_questions) if category_questions else 0
            }
        
        # Generate recommendations
        for category, stats in report['by_category'].items():
            if stats['coverage_percentage'] < 0.5:
                report['recommendations'].append(
                    f"Priority: Improve {category} evidence collection - only {stats['coverage_percentage']:.1%} coverage"
                )
        
        return report
    
    def visualize_knowledge_graph(self, output_path: str = None) -> go.Figure:
        """
        Create interactive visualization of the knowledge graph.
        
        Args:
            output_path: Optional path to save the visualization
            
        Returns:
            Plotly figure object
        """
        # Create NetworkX graph for layout
        G = nx.Graph()
        
        # Add nodes
        for node in self.nodes.values():
            G.add_node(node.id, type=node.type, **node.properties)
        
        # Add edges
        for edge in self.edges:
            G.add_edge(edge.source, edge.target, relationship=edge.relationship, weight=edge.weight)
        
        # Calculate layout
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # Prepare data for Plotly
        node_trace = go.Scatter(
            x=[pos[node][0] for node in G.nodes()],
            y=[pos[node][1] for node in G.nodes()],
            mode='markers+text',
            text=[self.nodes[node].type for node in G.nodes()],
            textposition="middle center",
            hovertemplate='<b>%{text}</b><br>ID: %{customdata}<extra></extra>',
            customdata=[node for node in G.nodes()],
            marker=dict(
                size=20,
                color=[hash(self.nodes[node].type) for node in G.nodes()],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Node Type")
            )
        )
        
        edge_trace = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace.append(go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=1, color='gray'),
                hoverinfo='none'
            ))
        
        # Create figure
        fig = go.Figure(data=[node_trace] + edge_trace)
        fig.update_layout(
            title="PEAT Knowledge Graph",
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[ dict(
                text="Interactive Knowledge Graph - Hover over nodes for details",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002,
                xanchor='left', yanchor='bottom',
                font=dict(color="gray", size=12)
            )],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        
        if output_path:
            fig.write_html(output_path)
            logger.info(f"Knowledge graph visualization saved to {output_path}")
        
        return fig
    
    def export_to_neo4j(self, clear_existing: bool = True):
        """Export the knowledge graph to Neo4j database."""
        if not self.use_neo4j:
            logger.warning("Neo4j not configured - cannot export")
            return
        
        with self.driver.session() as session:
            if clear_existing:
                session.run("MATCH (n) DETACH DELETE n")
                logger.info("Cleared existing Neo4j data")
            
            # Create nodes
            for node in self.nodes.values():
                query = f"""
                CREATE (n:{node.type} {{id: $id}})
                SET n += $properties
                """
                session.run(query, id=node.id, properties=node.properties)
            
            # Create relationships
            for edge in self.edges:
                query = f"""
                MATCH (a {{id: $source}}), (b {{id: $target}})
                CREATE (a)-[r:{edge.relationship}]->(b)
                SET r += $properties
                SET r.weight = $weight
                """
                session.run(query, 
                           source=edge.source, 
                           target=edge.target,
                           properties=edge.properties,
                           weight=edge.weight)
            
            logger.info(f"Exported {len(self.nodes)} nodes and {len(self.edges)} relationships to Neo4j")
    
    def save_analysis_results(self, output_dir: str):
        """Save analysis results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save assessments
        assessments_data = [asdict(assessment) for assessment in self.assessments]
        with open(output_path / 'evidence_assessments.json', 'w') as f:
            json.dump(assessments_data, f, indent=2)
        
        # Save gap analysis
        with open(output_path / 'gap_analysis.json', 'w') as f:
            json.dump(self.gap_analysis, f, indent=2)
        
        # Save gap analysis report
        gap_report = self.get_gap_analysis_report()
        with open(output_path / 'gap_analysis_report.json', 'w') as f:
            json.dump(gap_report, f, indent=2)
        
        # Save PEAT questions
        questions_data = {qid: asdict(q) for qid, q in self.peat_questions.items()}
        with open(output_path / 'peat_questions.json', 'w') as f:
            json.dump(questions_data, f, indent=2)
        
        logger.info(f"Analysis results saved to {output_dir}")
    
    def close(self):
        """Close database connections."""
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