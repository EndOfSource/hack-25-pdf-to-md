"""
Semantic Analysis Module for PEAT Evidence Assessment
This module provides semantic processing capabilities for MOD project documents
to automatically identify and assess evidence against PEAT requirements.
"""

import logging
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer
import torch
import spacy
from sklearn.metrics.pairwise import cosine_similarity
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """Represents an extracted entity from document text."""
    text: str
    type: str
    start_pos: int
    end_pos: int
    confidence: float


@dataclass
class Evidence:
    """Represents a piece of evidence extracted from documents."""
    content: str
    document_id: str
    page_number: Optional[int]
    entities: List[Entity]
    embedding: Optional[np.ndarray]
    peat_categories: List[str]
    confidence_score: float
    metadata: Dict[str, Any]


@dataclass
class SemanticChunk:
    """Represents a semantically meaningful chunk of text."""
    text: str
    start_index: int
    end_index: int
    embedding: Optional[np.ndarray]
    topic: Optional[str]
    entities: List[Entity]


class SemanticProcessor:
    """
    Main class for semantic processing of MOD project documents.
    Handles embeddings, entity extraction, and PEAT category mapping.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 categories_file: Optional[str] = None,
                 weights_file: Optional[str] = None):
        """
        Initialize the semantic processor with required models.
        
        Args:
            config: Configuration dictionary for model parameters
            categories_file: Path to PEAT categories JSON file
            weights_file: Path to category keyword weights JSON file
        """
        self.config = config or self._get_default_config()
        
        # Store file paths for configuration
        self.categories_file = categories_file or 'peat_categories.json'
        self.weights_file = weights_file or 'peat_weights.json'
        
        # Initialize models
        logger.info("Initializing semantic models...")
        self._initialize_models()
        
        # Load PEAT categories and weights
        self.peat_categories = self._load_peat_categories()
        self.category_weights = self._load_category_weights()
        
        # Initialize entity types relevant to MOD projects
        self.relevant_entity_types = {
            'ORG': 'Organization',
            'DATE': 'Date',
            'PERSON': 'Person',
            'GPE': 'Geo-political entity',
            'MONEY': 'Monetary value',
            'PERCENT': 'Percentage',
            'ORDINAL': 'Ordinal number',
            'CARDINAL': 'Cardinal number',
            'FAC': 'Facility',
            'PRODUCT': 'Product',
            'EVENT': 'Event',
            'LAW': 'Law or regulation'
        }
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration for the processor."""
        return {
            'embedding_model': 'sentence-transformers/all-mpnet-base-v2',
            'ner_model': 'dslim/bert-base-NER',
            'qa_model': 'deepset/roberta-base-squad2',
            'classifier_model': 'facebook/bart-large-mnli',
            'similarity_model': 'microsoft/deberta-v3-base',
            'chunk_size': 512,
            'chunk_overlap': 50,
            'min_confidence_threshold': 0.6,
            'entity_confidence_threshold': 0.8
        }
    
    def _initialize_models(self):
        """Initialize all required NLP models."""
        try:
            # Sentence embeddings model
            self.embedder = SentenceTransformer(self.config['embedding_model'])
            
            # Named Entity Recognition
            self.ner_pipeline = pipeline(
                "ner", 
                model=self.config['ner_model'],
                aggregation_strategy="simple"
            )
            
            # Question Answering
            self.qa_pipeline = pipeline(
                "question-answering",
                model=self.config['qa_model']
            )
            
            # Zero-shot classification
            self.classifier = pipeline(
                "zero-shot-classification",
                model=self.config['classifier_model']
            )
            
            # SpaCy for additional NLP tasks
            self.nlp = spacy.load("en_core_web_sm")
            
            logger.info("All models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
            raise
    
    def _load_peat_categories(self) -> Dict[str, List[str]]:
        """Load PEAT categories from configuration file or use defaults."""
        try:
            with open(self.categories_file, 'r') as f:
                categories = json.load(f)
                logger.info(f"Loaded PEAT categories from {self.categories_file}")
                return categories
        except FileNotFoundError:
            logger.warning(f"Categories file {self.categories_file} not found, using defaults")
            return self._get_default_categories()
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing categories file: {e}")
            return self._get_default_categories()
    
    def _load_category_weights(self) -> Dict[str, Dict[str, float]]:
        """Load category keyword weights from configuration file."""
        try:
            with open(self.weights_file, 'r') as f:
                weights = json.load(f)
                logger.info(f"Loaded category weights from {self.weights_file}")
                return weights
        except FileNotFoundError:
            logger.warning(f"Weights file {self.weights_file} not found, using default weight 1.0")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing weights file: {e}")
            return {}
    
    def _get_default_categories(self) -> Dict[str, List[str]]:
        """Return default PEAT categories if config file not found."""
        return {
            'governance': [
                'project board', 'steering committee', 'governance structure',
                'decision making', 'accountability', 'RACI matrix', 'terms of reference',
                'investment approval committee', 'IAC', 'project control', 'PC',
                'senior responsible owner', 'SRO', 'project director', 'PD',
                'project manager', 'PM', 'PSEC', 'project safety environmental committee',
                'CIAWG', 'capability integration acceptance working group',
                'authority', 'delegation', 'oversight', 'assurance', 'scrutiny',
                'approval gates', 'review board', 'programme board', 'portfolio board',
                'escalation', 'reporting structure', 'command structure', 'TOR'
            ],
            'requirements': [
                'KUR', 'key user requirements', 'functional requirements',
                'non-functional requirements', 'specifications', 'acceptance criteria',
                'user requirements document', 'URD', 'system requirements', 'SRD',
                'KPP', 'key performance parameters', 'measures of effectiveness',
                'threshold', 'objective', 'requirement justification', 'requirement ID',
                'operational requirements', 'technical requirements', 'interface requirements',
                'performance requirements', 'JROC', 'joint requirements oversight committee',
                'requirement validation', 'requirement verification', 'requirement traceability'
            ],
            'risk_management': [
                'risk register', 'risk assessment', 'mitigation', 'contingency',
                'risk appetite', 'risk matrix', 'probability', 'impact',
                'risk owner', 'risk category', 'technical risk', 'schedule risk',
                'cost risk', 'operational risk', 'safety risk', 'environmental risk',
                'HAZARD log', 'hazard analysis', 'safety case', 'SEMP',
                'safety environmental management plan', 'risk reduction', 'risk acceptance',
                'residual risk', 'risk treatment', 'risk monitoring', 'risk review'
            ],
            'stakeholder_management': [
                'stakeholder register', 'stakeholder analysis', 'engagement plan',
                'communication plan', 'RACI', 'power interest grid', 'stakeholder matrix',
                'front line command', 'FLC', 'user community', 'delivery partner',
                'industry partner', 'supplier engagement', 'customer engagement',
                'stakeholder communication', 'consultation', 'information requirements',
                'stakeholder categories', 'influence', 'interest', 'engagement strategy',
                'communication frequency', 'reporting requirements', 'stakeholder review'
            ],
            'benefits_management': [
                'benefits realization', 'benefits register', 'benefits tracking',
                'value proposition', 'outcomes', 'KPIs', 'success criteria',
                'operational benefits', 'capability benefits', 'efficiency gains',
                'cost avoidance', 'benefit owner', 'benefit profile', 'dis-benefits',
                'benefit dependencies', 'benefit measurement', 'benefit baseline',
                'FOC', 'full operating capability', 'IOC', 'initial operating capability',
                'capability delivery', 'force elements at readiness', 'operational advantage'
            ],
            'resources': [
                'resource plan', 'staffing', 'budget', 'funding', 'allocation',
                'capacity planning', 'skills matrix', 'SQEP', 'suitably qualified experienced personnel',
                'manpower', 'FTE', 'full time equivalent', 'contractor resources',
                'cost estimate', 'ABC', 'annual budget cycle', 'financial authority',
                'cost cap', 'cost baseline', 'resource profile', 'team structure',
                'organization chart', 'organogram', 'roles and responsibilities',
                'commercial resources', 'technical resources', 'project support'
            ],
            'schedule': [
                'project plan', 'timeline', 'milestones', 'critical path',
                'Gantt chart', 'dependencies', 'deadlines', 'programme plan',
                'integrated master schedule', 'IMS', 'key milestones', 'decision points',
                'IGBC', 'initial gate business case', 'OBC', 'outline business case',
                'FBC', 'full business case', 'assessment phase', 'demonstration phase',
                'manufacture phase', 'in-service phase', 'disposal phase',
                'PASE', 'planned acceptance service entry', 'schedule baseline',
                'schedule risk', 'float', 'schedule compression', 'parallel activities'
            ],
            'quality': [
                'quality plan', 'quality assurance', 'quality control',
                'standards', 'testing', 'validation', 'verification',
                'PQMP', 'project quality management plan', 'quality metrics',
                'quality reviews', 'technical reviews', 'design reviews',
                'GEAR', 'guide engineering activities reviews', 'acceptance testing',
                'qualification testing', 'certification', 'compliance', 'audit',
                'defect management', 'quality gates', 'inspection', 'MOD standards',
                'defence standards', 'JSP', 'joint service publication',
                'configuration management', 'baseline control', 'change control'
            ],
            'technical_management': [
                'technical authority', 'design authority', 'system architecture',
                'technical baseline', 'interface management', 'integration plan',
                'technical risk', 'technology readiness level', 'TRL', 'design maturity',
                'COTS', 'commercial off the shelf', 'MOTS', 'military off the shelf',
                'obsolescence management', 'configuration control', 'technical documentation',
                'engineering management plan', 'system design', 'subsystem design'
            ],
            'commercial_management': [
                'procurement strategy', 'contract management', 'supplier management',
                'competitive procurement phase', 'CPP', 'ITN', 'invitation to negotiate',
                'BAFO', 'best and final offer', 'MEAT', 'most economically advantageous tender',
                'framework agreement', 'FATS', 'framework agreement technical support',
                'commercial baseline', 'contract deliverables', 'payment milestones',
                'intellectual property rights', 'IPR', 'terms and conditions',
                'supply chain', 'subcontractor management', 'commercial risk'
            ],
            'logistics_support': [
                'integrated logistics support', 'ILS', 'ILSP', 'integrated logistics support plan',
                'support solution', 'maintenance concept', 'spares provisioning',
                'training requirements', 'technical documentation', 'support equipment',
                'DLOD', 'delegated lines of development', 'equipment DLOD', 'training DLOD',
                'infrastructure DLOD', 'personnel DLOD', 'information DLOD', 'doctrine DLOD',
                'organization DLOD', 'through life support', 'availability requirements'
            ]
         }
    
    def process_document(self, text: str, document_id: str, 
                        metadata: Optional[Dict[str, Any]] = None) -> List[Evidence]:
        """
        Process a complete document and extract evidence.
        
        Args:
            text: Document text content
            document_id: Unique identifier for the document
            metadata: Additional document metadata
            
        Returns:
            List of Evidence objects extracted from the document
        """
        logger.info(f"Processing document: {document_id}")
        
        # Create semantic chunks
        chunks = self._create_semantic_chunks(text)
        
        # Process each chunk
        evidence_list = []
        for chunk in chunks:
            # Generate embeddings
            chunk.embedding = self._generate_embedding(chunk.text)
            
            # Extract entities
            chunk.entities = self._extract_entities(chunk.text)
            
            # Identify topic/category
            chunk.topic = self._classify_chunk(chunk.text)
            
            # Create evidence object if confidence is sufficient
            evidence = self._create_evidence(chunk, document_id, metadata)
            if evidence and evidence.confidence_score >= self.config['min_confidence_threshold']:
                evidence_list.append(evidence)
        
        logger.info(f"Extracted {len(evidence_list)} evidence items from document {document_id}")
        return evidence_list
    
    def _create_semantic_chunks(self, text: str) -> List[SemanticChunk]:
        """
        Split text into semantically meaningful chunks.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of SemanticChunk objects
        """
        chunks = []
        doc = self.nlp(text)
        
        # Use sentence boundaries for initial chunking
        sentences = list(doc.sents)
        
        current_chunk = []
        current_length = 0
        start_index = 0
        
        for sent in sentences:
            sent_length = len(sent.text.split())
            
            if current_length + sent_length > self.config['chunk_size']:
                # Create chunk from accumulated sentences
                if current_chunk:
                    chunk_text = ' '.join([s.text for s in current_chunk])
                    chunks.append(SemanticChunk(
                        text=chunk_text,
                        start_index=start_index,
                        end_index=current_chunk[-1].end_char,
                        embedding=None,
                        topic=None,
                        entities=[]
                    ))
                
                # Start new chunk with overlap
                overlap_sentences = current_chunk[-2:] if len(current_chunk) >= 2 else current_chunk
                current_chunk = overlap_sentences + [sent]
                current_length = sum(len(s.text.split()) for s in current_chunk)
                start_index = overlap_sentences[0].start_char if overlap_sentences else sent.start_char
            else:
                current_chunk.append(sent)
                current_length += sent_length
        
        # Don't forget the last chunk
        if current_chunk:
            chunk_text = ' '.join([s.text for s in current_chunk])
            chunks.append(SemanticChunk(
                text=chunk_text,
                start_index=start_index,
                end_index=current_chunk[-1].end_char,
                embedding=None,
                topic=None,
                entities=[]
            ))
        
        return chunks
    
    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate semantic embedding for text."""
        return self.embedder.encode(text, convert_to_numpy=True)
    
    def _extract_entities(self, text: str) -> List[Entity]:
        """Extract named entities from text."""
        entities = []
        
        # Use transformer-based NER
        ner_results = self.ner_pipeline(text)
        
        for result in ner_results:
            if result['score'] >= self.config['entity_confidence_threshold']:
                entity_type = result['entity_group']
                if entity_type in self.relevant_entity_types:
                    entities.append(Entity(
                        text=result['word'],
                        type=entity_type,
                        start_pos=result['start'],
                        end_pos=result['end'],
                        confidence=result['score']
                    ))
        
        # Also extract MOD-specific entities using patterns
        mod_entities = self._extract_mod_specific_entities(text)
        entities.extend(mod_entities)
        
        return entities
    
    def _extract_mod_specific_entities(self, text: str) -> List[Entity]:
        """Extract MOD-specific entities like DLOD, KUR references, etc."""
        mod_entities = []
        
        # Define patterns for MOD-specific terms
        patterns = {
            'DLOD': r'\b(DLOD|Delegated Lines? of Development)\b',
            'KUR': r'\b(KUR|Key User Requirements?)\s*\d*\b',
            'PHASE': r'\b(Assessment Phase|AP|Demonstration Phase|Manufacture Phase)\b',
            'COMMITTEE': r'\b(PSEC|CIAWG|IAC|JROC)\b',
            'DOCUMENT': r'\b(PMP|FBC|OBC|IGBC|SEMP)\b'
        }
        
        import re
        for entity_type, pattern in patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                mod_entities.append(Entity(
                    text=match.group(),
                    type=f'MOD_{entity_type}',
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=0.95  # High confidence for pattern matches
                ))
        
        return mod_entities
    
    def _classify_chunk(self, text: str) -> Optional[str]:
        """Classify text chunk into PEAT categories."""
        categories = list(self.peat_categories.keys())
        
        result = self.classifier(
            text,
            candidate_labels=categories,
            multi_label=True
        )
        
        # Return top category if confidence is high enough
        if result['scores'][0] >= self.config['min_confidence_threshold']:
            return result['labels'][0]
        return None
    
    def _create_evidence(self, chunk: SemanticChunk, document_id: str,
                        metadata: Optional[Dict[str, Any]] = None) -> Optional[Evidence]:
        """Create an Evidence object from a semantic chunk."""
        if not chunk.topic:
            return None
        
        # Calculate confidence based on multiple factors
        confidence = self._calculate_evidence_confidence(chunk)
        
        # Determine relevant PEAT categories
        peat_categories = self._map_to_peat_categories(chunk)
        
        return Evidence(
            content=chunk.text,
            document_id=document_id,
            page_number=metadata.get('page_number') if metadata else None,
            entities=chunk.entities,
            embedding=chunk.embedding,
            peat_categories=peat_categories,
            confidence_score=confidence,
            metadata=metadata or {}
        )
    
    def _calculate_evidence_confidence(self, chunk: SemanticChunk) -> float:
        """
        Calculate confidence score for evidence based on multiple factors.
        
        Factors considered:
        - Entity density and relevance
        - Topic classification confidence  
        - Presence of key terms with weights
        - Document structure indicators
        """
        scores = []
        
        # Entity density score
        entity_score = min(len(chunk.entities) / 10.0, 1.0)  # Normalize to 0-1
        scores.append(entity_score * 0.25)  # 25% weight
        
        # MOD-specific entity presence
        mod_entities = [e for e in chunk.entities if e.type.startswith('MOD_')]
        mod_score = min(len(mod_entities) / 5.0, 1.0)
        scores.append(mod_score * 0.35)  # 35% weight
        
        # Weighted key term presence
        weighted_term_score = self._calculate_weighted_term_score(chunk.text)
        scores.append(weighted_term_score * 0.4)  # 40% weight
        
        return sum(scores)
    
    def _calculate_weighted_term_score(self, text: str) -> float:
        """Calculate weighted score based on presence of key terms."""
        total_weight = 0.0
        max_possible_weight = 0.0
        
        text_lower = text.lower()
        
        # Check all categories for weighted keywords
        for category, keywords in self.peat_categories.items():
            category_weights = self.category_weights.get(category, {})
            
            for keyword in keywords:
                weight = category_weights.get(keyword, 1.0)
                max_possible_weight += weight
                
                if keyword.lower() in text_lower:
                    total_weight += weight
        
        # Normalize to 0-1 range
        if max_possible_weight > 0:
            return min(total_weight / (max_possible_weight * 0.1), 1.0)  # Scale factor
        return 0.0
    
    def _map_to_peat_categories(self, chunk: SemanticChunk) -> List[str]:
        """Map chunk to relevant PEAT categories based on content with weighted keywords."""
        categories = []
        
        # Use embedding similarity to find relevant categories
        chunk_embedding = chunk.embedding.reshape(1, -1)
        
        for category, keywords in self.peat_categories.items():
            # Get weights for this category
            category_weights = self.category_weights.get(category, {})
            
            # Calculate weighted score for keywords found in chunk
            keyword_score = 0.0
            keywords_found = 0
            
            for keyword in keywords:
                if keyword.lower() in chunk.text.lower():
                    # Get weight for this keyword (default 1.0 if not specified)
                    weight = category_weights.get(keyword, 1.0)
                    keyword_score += weight
                    keywords_found += 1
            
            # Normalize keyword score
            if keywords_found > 0:
                keyword_score = keyword_score / len(keywords)
            
            # Create weighted embedding for category keywords
            weighted_keyword_texts = []
            for keyword in keywords:
                weight = category_weights.get(keyword, 1.0)
                # Repeat keyword based on weight (rounded to int)
                weighted_keyword_texts.extend([keyword] * int(weight))
            
            category_text = ' '.join(weighted_keyword_texts)
            category_embedding = self._generate_embedding(category_text).reshape(1, -1)
            
            # Calculate similarity
            similarity = cosine_similarity(chunk_embedding, category_embedding)[0][0]
            
            # Combine similarity score with keyword score
            final_score = (similarity * 0.7) + (keyword_score * 0.3)
            
            if final_score >= 0.6:  # Threshold for category relevance
                categories.append(category)
        
        return categories
    
    def find_evidence_for_peat_question(self, question: str, 
                                      evidence_list: List[Evidence]) -> List[Tuple[Evidence, float]]:
        """
        Find relevant evidence for a specific PEAT question.
        
        Args:
            question: PEAT question text
            evidence_list: List of available evidence
            
        Returns:
            List of tuples (Evidence, relevance_score) sorted by relevance
        """
        question_embedding = self._generate_embedding(question).reshape(1, -1)
        
        scored_evidence = []
        for evidence in evidence_list:
            if evidence.embedding is not None:
                evidence_embedding = evidence.embedding.reshape(1, -1)
                similarity = cosine_similarity(question_embedding, evidence_embedding)[0][0]
                
                # Boost score if entities are relevant
                entity_boost = self._calculate_entity_relevance_boost(question, evidence.entities)
                final_score = similarity + (entity_boost * 0.2)
                
                scored_evidence.append((evidence, final_score))
        
        # Sort by relevance score
        scored_evidence.sort(key=lambda x: x[1], reverse=True)
        
        return scored_evidence
    
    def _calculate_entity_relevance_boost(self, question: str, entities: List[Entity]) -> float:
        """Calculate relevance boost based on entity matches with question."""
        question_lower = question.lower()
        relevant_entities = 0
        
        for entity in entities:
            if entity.text.lower() in question_lower:
                relevant_entities += 1
        
        return min(relevant_entities / 3.0, 1.0)  # Normalize to 0-1
    
    def generate_evidence_summary(self, evidence_list: List[Evidence]) -> Dict[str, Any]:
        """
        Generate a summary of extracted evidence by category.
        
        Args:
            evidence_list: List of evidence objects
            
        Returns:
            Summary dictionary with statistics and categorization
        """
        summary = {
            'total_evidence': len(evidence_list),
            'by_category': {},
            'confidence_distribution': {
                'high': 0,    # > 0.8
                'medium': 0,  # 0.6 - 0.8
                'low': 0      # < 0.6
            },
            'entity_types': {},
            'documents_covered': set()
        }
        
        for evidence in evidence_list:
            # Category distribution
            for category in evidence.peat_categories:
                summary['by_category'][category] = summary['by_category'].get(category, 0) + 1
            
            # Confidence distribution
            if evidence.confidence_score > 0.8:
                summary['confidence_distribution']['high'] += 1
            elif evidence.confidence_score >= 0.6:
                summary['confidence_distribution']['medium'] += 1
            else:
                summary['confidence_distribution']['low'] += 1
            
            # Entity type distribution
            for entity in evidence.entities:
                entity_type = entity.type
                summary['entity_types'][entity_type] = summary['entity_types'].get(entity_type, 0) + 1
            
            # Documents covered
            summary['documents_covered'].add(evidence.document_id)
        
        summary['documents_covered'] = list(summary['documents_covered'])
        
        return summary
    
    def export_evidence_for_knowledge_graph(self, evidence_list: List[Evidence]) -> List[Dict[str, Any]]:
        """
        Export evidence in format suitable for knowledge graph creation.
        
        Args:
            evidence_list: List of evidence objects
            
        Returns:
            List of dictionaries formatted for Neo4j import
        """
        graph_data = []
        
        for evidence in evidence_list:
            # Create evidence node data
            evidence_node = {
                'type': 'Evidence',
                'properties': {
                    'id': f"{evidence.document_id}_{hash(evidence.content)[:8]}",
                    'content': evidence.content,
                    'confidence_score': evidence.confidence_score,
                    'categories': evidence.peat_categories,
                    'document_id': evidence.document_id,
                    'page_number': evidence.page_number,
                    'timestamp': datetime.now().isoformat()
                }
            }
            graph_data.append(evidence_node)
            
            # Create entity nodes and relationships
            for entity in evidence.entities:
                entity_node = {
                    'type': 'Entity',
                    'properties': {
                        'id': f"entity_{hash(entity.text + entity.type)[:8]}",
                        'text': entity.text,
                        'entity_type': entity.type,
                        'confidence': entity.confidence
                    }
                }
                graph_data.append(entity_node)
                
                # Create relationship
                relationship = {
                    'type': 'MENTIONS',
                    'from': evidence_node['properties']['id'],
                    'to': entity_node['properties']['id'],
                    'properties': {
                        'position': f"{entity.start_pos}-{entity.end_pos}"
                    }
                }
                graph_data.append(relationship)
        
        return graph_data
    
    def save_configurations(self, categories_output: Optional[str] = None,
                          weights_output: Optional[str] = None):
        """
        Save current configurations to JSON files.
        
        Args:
            categories_output: Output path for categories (default: uses init path)
            weights_output: Output path for weights (default: uses init path)
        """
        # Save categories
        if categories_output or self.categories_file:
            output_path = categories_output or self.categories_file
            with open(output_path, 'w') as f:
                json.dump(self.peat_categories, f, indent=2)
            logger.info(f"Saved categories to {output_path}")
        
        # Save weights
        if weights_output or self.weights_file:
            output_path = weights_output or self.weights_file
            with open(output_path, 'w') as f:
                json.dump(self.category_weights, f, indent=2)
            logger.info(f"Saved weights to {output_path}")
    
    def add_category_keywords(self, category: str, keywords: List[str], 
                            weights: Optional[Dict[str, float]] = None):
        """
        Add keywords to a category with optional weights.
        
        Args:
            category: Category name
            keywords: List of keywords to add
            weights: Optional dictionary of keyword weights
        """
        if category not in self.peat_categories:
            self.peat_categories[category] = []
        
        # Add keywords
        for keyword in keywords:
            if keyword not in self.peat_categories[category]:
                self.peat_categories[category].append(keyword)
        
        # Add weights
        if weights:
            if category not in self.category_weights:
                self.category_weights[category] = {}
            self.category_weights[category].update(weights)
        
        logger.info(f"Added {len(keywords)} keywords to category '{category}'")
    
    def update_keyword_weight(self, category: str, keyword: str, weight: float):
        """Update the weight for a specific keyword in a category."""
        if category not in self.category_weights:
            self.category_weights[category] = {}
        
        self.category_weights[category][keyword] = weight
        logger.info(f"Updated weight for '{keyword}' in '{category}' to {weight}")