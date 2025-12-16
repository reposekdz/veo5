import torch
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import json
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import sqlite3
from datetime import datetime
from .base_model import BaseMultimodalModel

class AdvancedKnowledgeBase(BaseMultimodalModel):
    """Advanced knowledge base with vector search and semantic retrieval"""
    
    def __init__(self, device: str = "cuda", db_path: str = "./knowledge_base.db"):
        super().__init__("knowledge_base", device)
        self.embedding_model = None
        self.vector_index = None
        self.db_path = db_path
        self.documents = []
        self.embeddings = None
        
    def load_model(self):
        """Load knowledge base models"""
        if self.is_loaded:
            return
            
        try:
            # Load embedding model
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Initialize database
            self._init_database()
            
            # Load existing knowledge base
            self._load_existing_kb()
            
            self.is_loaded = True
            self.logger.info("Knowledge base loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load knowledge base: {e}")
            raise
    
    def unload_model(self):
        """Unload models from memory"""
        if not self.is_loaded:
            return
            
        del self.embedding_model
        del self.vector_index
        
        self.embedding_model = None
        self.vector_index = None
        self.is_loaded = False
        self.optimize_memory()
    
    def generate(self, *args, **kwargs):
        """Generate knowledge base response"""
        return self.search(*args, **kwargs)
    
    def _init_database(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT,
                content TEXT,
                source TEXT,
                category TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS embeddings (
                doc_id INTEGER,
                embedding BLOB,
                FOREIGN KEY (doc_id) REFERENCES documents (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _load_existing_kb(self):
        """Load existing knowledge base from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Load documents
        cursor.execute("SELECT * FROM documents")
        rows = cursor.fetchall()
        
        self.documents = []
        embeddings_list = []
        
        for row in rows:
            doc = {
                "id": row[0],
                "title": row[1],
                "content": row[2],
                "source": row[3],
                "category": row[4],
                "metadata": json.loads(row[5]) if row[5] else {},
                "created_at": row[6],
                "updated_at": row[7]
            }
            self.documents.append(doc)
            
            # Load embedding
            cursor.execute("SELECT embedding FROM embeddings WHERE doc_id = ?", (row[0],))
            emb_row = cursor.fetchone()
            if emb_row:
                embedding = pickle.loads(emb_row[0])
                embeddings_list.append(embedding)
        
        conn.close()
        
        # Build vector index
        if embeddings_list:
            self.embeddings = np.array(embeddings_list)
            self._build_vector_index()
    
    def _build_vector_index(self):
        """Build FAISS vector index"""
        if self.embeddings is None or len(self.embeddings) == 0:
            return
            
        dimension = self.embeddings.shape[1]
        self.vector_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        self.vector_index.add(self.embeddings)
    
    def add_document(
        self,
        title: str,
        content: str,
        source: str = "manual",
        category: str = "general",
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """Add document to knowledge base"""
        
        if not self.is_loaded:
            raise RuntimeError("Knowledge base not loaded")
        
        if metadata is None:
            metadata = {}
        
        # Generate embedding
        embedding = self.embedding_model.encode([content])[0]
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO documents (title, content, source, category, metadata)
            VALUES (?, ?, ?, ?, ?)
        ''', (title, content, source, category, json.dumps(metadata)))
        
        doc_id = cursor.lastrowid
        
        # Store embedding
        cursor.execute('''
            INSERT INTO embeddings (doc_id, embedding)
            VALUES (?, ?)
        ''', (doc_id, pickle.dumps(embedding)))
        
        conn.commit()
        conn.close()
        
        # Update in-memory structures
        doc = {
            "id": doc_id,
            "title": title,
            "content": content,
            "source": source,
            "category": category,
            "metadata": metadata,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        self.documents.append(doc)
        
        # Update embeddings and index
        if self.embeddings is None:
            self.embeddings = np.array([embedding])
        else:
            self.embeddings = np.vstack([self.embeddings, embedding])
        
        self._build_vector_index()
        
        return doc_id
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        category: Optional[str] = None,
        source: Optional[str] = None,
        threshold: float = 0.3
    ) -> List[Dict[str, Any]]:
        """Search knowledge base using semantic similarity"""
        
        if not self.is_loaded or self.vector_index is None:
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search vector index
        scores, indices = self.vector_index.search(query_embedding, min(top_k * 2, len(self.documents)))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if score < threshold:
                continue
                
            doc = self.documents[idx]
            
            # Apply filters
            if category and doc["category"] != category:
                continue
            if source and doc["source"] != source:
                continue
            
            result = {
                "document": doc,
                "score": float(score),
                "relevance": "high" if score > 0.7 else "medium" if score > 0.5 else "low"
            }
            results.append(result)
            
            if len(results) >= top_k:
                break
        
        return results
    
    def add_research_papers(self, papers: List[Dict[str, Any]]):
        """Add research papers to knowledge base"""
        
        for paper in papers:
            title = paper.get("title", "Unknown Title")
            abstract = paper.get("abstract", "")
            
            if not abstract:
                continue
            
            metadata = {
                "authors": paper.get("authors", []),
                "url": paper.get("url", ""),
                "published": paper.get("published", ""),
                "categories": paper.get("categories", []),
                "citations": paper.get("citations", 0)
            }
            
            self.add_document(
                title=title,
                content=abstract,
                source="research_paper",
                category="academic",
                metadata=metadata
            )
    
    def add_web_content(self, web_results: List[Dict[str, Any]]):
        """Add web search results to knowledge base"""
        
        for result in web_results:
            title = result.get("title", "Unknown Title")
            snippet = result.get("snippet", "")
            
            if not snippet:
                continue
            
            metadata = {
                "url": result.get("url", ""),
                "source_site": result.get("source", "")
            }
            
            self.add_document(
                title=title,
                content=snippet,
                source="web_search",
                category="web",
                metadata=metadata
            )
    
    def get_document(self, doc_id: int) -> Optional[Dict[str, Any]]:
        """Get document by ID"""
        
        for doc in self.documents:
            if doc["id"] == doc_id:
                return doc
        
        return None
    
    def update_document(
        self,
        doc_id: int,
        title: Optional[str] = None,
        content: Optional[str] = None,
        category: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Update existing document"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get current document
        cursor.execute("SELECT * FROM documents WHERE id = ?", (doc_id,))
        row = cursor.fetchone()
        
        if not row:
            conn.close()
            return False
        
        # Update fields
        new_title = title if title is not None else row[1]
        new_content = content if content is not None else row[2]
        new_category = category if category is not None else row[4]
        new_metadata = metadata if metadata is not None else json.loads(row[5]) if row[5] else {}
        
        cursor.execute('''
            UPDATE documents 
            SET title = ?, content = ?, category = ?, metadata = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (new_title, new_content, new_category, json.dumps(new_metadata), doc_id))
        
        # Update embedding if content changed
        if content is not None:
            new_embedding = self.embedding_model.encode([new_content])[0]
            cursor.execute('''
                UPDATE embeddings SET embedding = ? WHERE doc_id = ?
            ''', (pickle.dumps(new_embedding), doc_id))
        
        conn.commit()
        conn.close()
        
        # Update in-memory structures
        for i, doc in enumerate(self.documents):
            if doc["id"] == doc_id:
                self.documents[i].update({
                    "title": new_title,
                    "content": new_content,
                    "category": new_category,
                    "metadata": new_metadata,
                    "updated_at": datetime.now().isoformat()
                })
                
                if content is not None:
                    self.embeddings[i] = new_embedding
                    self._build_vector_index()
                
                break
        
        return True
    
    def delete_document(self, doc_id: int) -> bool:
        """Delete document from knowledge base"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Delete from database
        cursor.execute("DELETE FROM embeddings WHERE doc_id = ?", (doc_id,))
        cursor.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        
        if cursor.rowcount == 0:
            conn.close()
            return False
        
        conn.commit()
        conn.close()
        
        # Update in-memory structures
        doc_index = None
        for i, doc in enumerate(self.documents):
            if doc["id"] == doc_id:
                doc_index = i
                break
        
        if doc_index is not None:
            self.documents.pop(doc_index)
            if self.embeddings is not None:
                self.embeddings = np.delete(self.embeddings, doc_index, axis=0)
                self._build_vector_index()
        
        return True
    
    def get_categories(self) -> List[str]:
        """Get all categories in knowledge base"""
        
        categories = set()
        for doc in self.documents:
            categories.add(doc["category"])
        
        return list(categories)
    
    def get_sources(self) -> List[str]:
        """Get all sources in knowledge base"""
        
        sources = set()
        for doc in self.documents:
            sources.add(doc["source"])
        
        return list(sources)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics"""
        
        total_docs = len(self.documents)
        
        # Category distribution
        categories = {}
        sources = {}
        
        for doc in self.documents:
            cat = doc["category"]
            src = doc["source"]
            
            categories[cat] = categories.get(cat, 0) + 1
            sources[src] = sources.get(src, 0) + 1
        
        return {
            "total_documents": total_docs,
            "categories": categories,
            "sources": sources,
            "vector_index_size": self.vector_index.ntotal if self.vector_index else 0,
            "embedding_dimension": self.embeddings.shape[1] if self.embeddings is not None else 0
        }
    
    def export_knowledge_base(self, output_path: str):
        """Export knowledge base to JSON file"""
        
        export_data = {
            "documents": self.documents,
            "stats": self.get_stats(),
            "exported_at": datetime.now().isoformat()
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    def import_knowledge_base(self, input_path: str):
        """Import knowledge base from JSON file"""
        
        with open(input_path, 'r', encoding='utf-8') as f:
            import_data = json.load(f)
        
        documents = import_data.get("documents", [])
        
        for doc in documents:
            self.add_document(
                title=doc["title"],
                content=doc["content"],
                source=doc.get("source", "imported"),
                category=doc.get("category", "general"),
                metadata=doc.get("metadata", {})
            )