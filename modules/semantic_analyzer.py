"""
Enhanced Semantic Analysis Module for PEAT Evidence Assessment
This module provides semantic processing capabilities for MOD project documents
to automatically identify and assess evidence against PEAT requirements.
"""

import logging
from typing import List, Dict, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer
import torch
import spacy
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import json
import re
from pathlib import Path

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
    chunk_type: Optional[str]  # New field for chunk type
    page_number: Optional[int]
    entities: List[Entity]
    embedding: Optional[np.ndarray]
    peat_categories: List[str]
    confidence_score: float
    metadata: Dict[str, Any]
    id: str = field(default="", init=False)
    
    def __post_init__(self):
        """Generate unique ID for evidence."""
        self.id = f"{self.document_id}_{hash(self.content)}_{self.chunk_type or 'full'}"


@dataclass
class SemanticChunk:
    """Represents a semantically meaningful chunk of text."""
    text: str
    start_index: int
    end_index: int
    embedding: Optional[np.ndarray]
    topic: Optional[str]
    entities: List[Entity]
    chunk_type: Optional[str] = None  # New field


class SemanticProcessor:
    """
    Enhanced semantic processor with support for chunk-based processing.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 categories_file: Optional[str] = None,
                 weights_file: Optional[str] = None):
        """Initialize the semantic processor with required models."""
        self.config = config or self._get_default_config()
        
        # Initialize models
        logger.info("Loading semantic models...")
        self.embedder = SentenceTransformer(self.config['embedding_model'])
        self.nlp = spacy.load(self.config['spacy_model'])
        
        # Initialize NER pipeline
        self.ner_pipeline = pipeline(
            "ner",
            model=self.config['ner_model'],
            aggregation_strategy="simple"
        )
        
        # Initialize zero-shot classifier
        self.classifier = pipeline(
            "zero-shot-classification",
            model=self.config['classification_model']
        )
        
        # Load PEAT categories
        self.peat_categories = self._load_peat_categories(categories_file)
        
        # Entity types relevant to MOD projects
        self.relevant_entity_types = {
            'ORG', 'PERSON', 'DATE', 'MONEY', 'PERCENT', 'LOC', 'PRODUCT'
        }
        
        # Chunk type mappings
        self.chunk_type_to_category = {
            'risk_management': ['risk_management', 'safety_security'],
            'governance': ['governance', 'stakeholder_management'],
            'schedule': ['schedule_management', 'technical_management'],
            'cost_management': ['financial_management', 'cost_control'],
            'commercial_management': ['commercial_management', 'procurement'],
            'approvals_envelope': ['governance', 'stakeholder_management'],
            'execution_strategy': ['technical_management', 'logistics_support']
        }
        
        logger.info("Semantic processor initialized successfully")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration with lowered thresholds."""
        return {
            'embedding_model': 'all-MiniLM-L6-v2',
            'spacy_model': 'en_core_web_sm',
            'ner_model': 'dbmdz/bert-large-cased-finetuned-conll03-english',
            'classification_model': 'facebook/bart-large-mnli',
            'chunk_size': 256,
            'chunk_overlap': 50,
            
            # LOWERED THRESHOLDS for more comprehensive output
            'min_confidence_threshold': 0.2,  # Was 0.4, now much lower
            'entity_confidence_threshold': 0.3,  # Was 0.6, now lower
            'similarity_threshold': 0.5,  # Was 0.7, now lower
            'peat_relevance_threshold': 0.3,  # New: lower PEAT connection threshold
            'evidence_inclusion_threshold': 0.2,  # New: very low barrier for evidence inclusion
        }
    
    def _load_peat_categories(self, categories_file: Optional[str]) -> Dict[str, List[str]]:
        """Load PEAT categories and keywords."""
        if categories_file and Path(categories_file).exists():
            with open(categories_file, 'r') as f:
                return json.load(f)
        
        # Default PEAT categories with keywords
        return {
            'governance': [
                'governance', 'steering committee', 'board', 'authority', 'decision making',
                'accountability', 'oversight', 'approval', 'delegation', 'terms of reference',
                'project mandate', 'sponsorship', 'senior responsible owner', 'SRO',
                'project board', 'project steering group', 'PSG', 'escalation'
            ],
            'risk_management': [
                'risk', 'risk management', 'risk register', 'risk assessment', 'risk mitigation',
                'risk appetite', 'risk tolerance', 'issue', 'assumption', 'dependency',
                'RAID', 'risk owner', 'risk review', 'risk reporting', 'risk escalation',
                'monte carlo', 'sensitivity analysis', 'contingency', 'risk workshop'
            ],
            'schedule_management': [
                'schedule', 'milestone', 'timeline', 'gantt', 'critical path', 'float',
                'baseline', 'earned value', 'SPI', 'schedule variance', 'dependency',
                'predecessor', 'successor', 'work breakdown structure', 'WBS',
                'integrated master schedule', 'IMS', 'schedule risk analysis', 'SRA'
            ],
            'financial_management': [
                'cost', 'budget', 'financial', 'funding', 'expenditure', 'forecast',
                'cash flow', 'cost baseline', 'cost variance', 'CPI', 'earned value',
                'whole life cost', 'WLC', 'cost breakdown structure', 'CBS',
                'cost estimation', 'financial authority', 'affordability'
            ],
            'benefits_management': [
                'benefit', 'outcome', 'benefit realisation', 'benefit profile',
                'benefit owner', 'benefit tracking', 'dis-benefit', 'value',
                'benefit dependency network', 'benefit realisation plan',
                'benefit measurement', 'KPI', 'success criteria'
            ],
            'stakeholder_management': [
                'stakeholder', 'engagement', 'communication', 'consultation',
                'stakeholder analysis', 'stakeholder map', 'RACI', 'communication plan',
                'stakeholder register', 'influence', 'interest', 'user', 'customer'
            ],
            'quality_management': [
                'quality', 'quality assurance', 'quality control', 'quality plan',
                'quality criteria', 'quality review', 'quality gate', 'acceptance criteria',
                'verification', 'validation', 'V&V', 'test', 'inspection', 'audit'
            ],
            'change_management': [
                'change control', 'change request', 'change board', 'baseline change',
                'configuration management', 'version control', 'change impact',
                'change authority', 'change freeze', 'change log'
            ],
            'safety_security': [
                'safety', 'security', 'safety case', 'ALARP', 'as low as reasonably practicable',
                'hazard', 'hazard log', 'safety management plan', 'security classification',
                'protective marking', 'security risk assessment', 'accreditation'
            ],
            'technical_management': [
                'technical', 'design', 'architecture', 'interface', 'requirement',
                'specification', 'system engineering management plan', 'SEMP',
                'technical review', 'design review', 'CDR', 'PDR', 'TRR',
                'technical baseline', 'interface management', 'integration plan',
                'technical risk', 'technology readiness level', 'TRL', 'design maturity',
                'COTS', 'commercial off the shelf', 'MOTS', 'military off the shelf',
                'obsolescence management', 'configuration control', 'technical documentation',
                'engineering management plan', 'system design', 'subsystem design'
            ],
            'commercial_management': [
                'procurement', 'contract', 'supplier', 'vendor', 'commercial',
                'competitive procurement phase', 'CPP', 'ITN', 'invitation to negotiate',
                'BAFO', 'best and final offer', 'MEAT', 'most economically advantageous tender',
                'framework agreement', 'FATS', 'framework agreement technical support',
                'commercial baseline', 'contract deliverables', 'payment milestones',
                'intellectual property rights', 'IPR', 'terms and conditions',
                'supply chain', 'subcontractor management', 'commercial risk'
            ],
            'logistics_support': [
                'integrated logistics support', 'ILS', 'ILSP', 'integrated logistics support plan',
                'support solution', 'maintenance', 'spares', 'provisioning',
                'training', 'technical documentation', 'support equipment',
                'DLOD', 'delegated lines of development', 'equipment DLOD', 'training DLOD',
                'infrastructure DLOD', 'personnel DLOD', 'information DLOD', 'doctrine DLOD',
                'organization DLOD', 'through life support', 'availability'
            ]
        }
    
    def process_document(self, text: str, document_id: str,
                        metadata: Optional[Dict[str, Any]] = None) -> List[Evidence]:
        """Process a complete document and extract evidence."""
        logger.info(f"Processing document: {document_id}")
        
        # Extract chunk type from metadata if available
        chunk_type = metadata.get('chunk_type') if metadata else None
        
        # Create semantic chunks
        chunks = self._create_semantic_chunks(text, chunk_type)
        
        # Process each chunk
        evidence_list = []
        for chunk in chunks:
            # Generate embeddings
            chunk.embedding = self._generate_embedding(chunk.text)
            
            # Extract entities
            chunk.entities = self._extract_entities(chunk.text)
            
            # Identify topic/category with chunk type consideration
            chunk.topic = self._classify_chunk(chunk.text, chunk_type)
            
            # Create evidence object if confidence is sufficient
            evidence = self._create_evidence(chunk, document_id, metadata)
            if evidence and evidence.confidence_score >= self.config['min_confidence_threshold']:
                evidence_list.append(evidence)
        
        logger.info(f"Extracted {len(evidence_list)} evidence items from document {document_id}")
        return evidence_list
    
    def process_document_chunks(self, doc_dir: Path) -> Dict[str, List[Evidence]]:
        """Process individual document chunks instead of full.md."""
        chunk_files = [
            'risk_management.md', 'governance.md', 'schedule.md',
            'cost_management.md', 'commercial_management.md',
            'approvals_envelope.md', 'execution_strategy.md',
            'benefits_management.md', 'stakeholder_management.md'
        ]
        
        all_evidence = {}
        doc_name = doc_dir.name
        
        for chunk_file in chunk_files:
            chunk_path = doc_dir / chunk_file
            if chunk_path.exists():
                chunk_type = chunk_file.replace('.md', '')
                logger.info(f"Processing chunk: {chunk_type} from {doc_name}")
                
                with open(chunk_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                evidence = self.process_document(
                    content,
                    doc_name,
                    metadata={
                        'chunk_type': chunk_type,
                        'source_path': str(chunk_path)
                    }
                )
                all_evidence[chunk_type] = evidence
        
        return all_evidence
    
    def _create_semantic_chunks(self, text: str, chunk_type: Optional[str] = None) -> List[SemanticChunk]:
        """Split text into semantically meaningful chunks."""
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
                        entities=[],
                        chunk_type=chunk_type
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
                entities=[],
                chunk_type=chunk_type
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
        
        # Enhanced patterns for MOD-specific terms
        patterns = {
            'DLOD': r'\b(DLOD|Delegated Lines? of Development)\b',
            'KUR': r'\b(KUR|Key User Requirements?)\s*\d*\b',
            'PHASE': r'\b(Assessment Phase|AP|Demonstration Phase|Manufacture Phase|Initial Gate|Main Gate)\b',
            'COMMITTEE': r'\b(PSEC|CIAWG|IAC|JROC|Investment Approvals Committee)\b',
            'DOCUMENT': r'\b(PMP|FBC|OBC|IGBC|SEMP|ILSP|SSMP|SRD)\b',
            'STANDARD': r'\b(JSP\s*\d+|GovS\s*\d+|DCMA-14|ISO\s*\d+)\b',
            'MILESTONE': r'\b(IOC|FOC|ISD|Contract Award|CDR|PDR|TRR)\b'
        }
        
        for entity_type, pattern in patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                mod_entities.append(Entity(
                    text=match.group(),
                    type=f'MOD_{entity_type}',
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=0.95
                ))
        
        return mod_entities
    
    def _classify_chunk(self, text: str, chunk_type: Optional[str] = None) -> Optional[str]:
        """Classify text chunk into PEAT categories."""
        # If chunk type is provided, use predefined mappings
        if chunk_type and chunk_type in self.chunk_type_to_category:
            primary_categories = self.chunk_type_to_category[chunk_type]
            
            # Still run classification but boost scores for expected categories
            categories = list(self.peat_categories.keys())
            result = self.classifier(
                text,
                candidate_labels=categories,
                multi_label=True
            )
            
            # Boost scores for expected categories
            boosted_scores = []
            for label, score in zip(result['labels'], result['scores']):
                if label in primary_categories:
                    boosted_scores.append(min(score * 1.2, 1.0))  # 20% boost
                else:
                    boosted_scores.append(score)
            
            # Return top category
            max_idx = boosted_scores.index(max(boosted_scores))
            if boosted_scores[max_idx] >= self.config['min_confidence_threshold']:
                return result['labels'][max_idx]
        else:
            # Standard classification without boost
            categories = list(self.peat_categories.keys())
            result = self.classifier(
                text,
                candidate_labels=categories,
                multi_label=True
            )
            
            if result['scores'][0] >= self.config['min_confidence_threshold']:
                return result['labels'][0]
        
        return None
    
    def _create_evidence(self, chunk: SemanticChunk, document_id: str,
                        metadata: Optional[Dict[str, Any]] = None) -> Optional[Evidence]:
        """Create an Evidence object from a semantic chunk."""
        # Calculate relevance score based on entities and classification
        relevance_score = self._calculate_relevance_score(chunk)
        
        if relevance_score < 0.5:
            return None
        
        # Determine PEAT categories
        peat_categories = []
        if chunk.topic:
            peat_categories.append(chunk.topic)
        
        # Add categories based on chunk type
        if chunk.chunk_type and chunk.chunk_type in self.chunk_type_to_category:
            peat_categories.extend(self.chunk_type_to_category[chunk.chunk_type])
        
        # Remove duplicates
        peat_categories = list(set(peat_categories))
        
        return Evidence(
            content=chunk.text,
            document_id=document_id,
            chunk_type=chunk.chunk_type,
            page_number=None,
            entities=chunk.entities,
            embedding=chunk.embedding,
            peat_categories=peat_categories,
            confidence_score=relevance_score,
            metadata=metadata or {}
        )
    
    def _calculate_relevance_score(self, chunk: SemanticChunk) -> float:
        """Calculate relevance score for a chunk."""
        score = 0.0
        
        # Base score from classification confidence
        if chunk.topic:
            score += 0.5
        
        # Boost for MOD-specific entities
        mod_entities = [e for e in chunk.entities if e.type.startswith('MOD_')]
        if mod_entities:
            score += min(len(mod_entities) * 0.1, 0.3)
        
        # Boost for other relevant entities
        other_entities = [e for e in chunk.entities if not e.type.startswith('MOD_')]
        if other_entities:
            score += min(len(other_entities) * 0.05, 0.2)
        
        # Boost for chunk type alignment
        if chunk.chunk_type:
            score += 0.1
        
        return min(score, 1.0)
    
    def find_similar_evidence(self, evidence: Evidence, 
                            evidence_list: List[Evidence],
                            threshold: float = None) -> List[Tuple[Evidence, float]]:
        """Find similar evidence based on embeddings."""
        threshold = threshold or self.config['similarity_threshold']
        similar_evidence = []
        
        if evidence.embedding is None:
            return similar_evidence
        
        for other in evidence_list:
            if other.id == evidence.id or other.embedding is None:
                continue
            
            similarity = cosine_similarity(
                evidence.embedding.reshape(1, -1),
                other.embedding.reshape(1, -1)
            )[0, 0]
            
            if similarity >= threshold:
                similar_evidence.append((other, similarity))
        
        # Sort by similarity score
        similar_evidence.sort(key=lambda x: x[1], reverse=True)
        return similar_evidence
    
    def cluster_evidence(self, evidence_list: List[Evidence], n_clusters: int = 5) -> Dict[int, List[Evidence]]:
        """Cluster evidence based on semantic similarity."""
        # Extract embeddings
        embeddings = []
        valid_evidence = []
        
        for evidence in evidence_list:
            if evidence.embedding is not None:
                embeddings.append(evidence.embedding)
                valid_evidence.append(evidence)
        
        if len(embeddings) < n_clusters:
            logger.warning(f"Not enough evidence for {n_clusters} clusters")
            n_clusters = max(1, len(embeddings) // 2)
        
        # Perform clustering
        embeddings_matrix = np.array(embeddings)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(embeddings_matrix)
        
        # Group evidence by cluster
        cluster_dict = {}
        for idx, cluster_id in enumerate(clusters):
            if cluster_id not in cluster_dict:
                cluster_dict[cluster_id] = []
            cluster_dict[cluster_id].append(valid_evidence[idx])
        
        return cluster_dict