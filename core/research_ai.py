import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification,
    pipeline, BitsAndBytesConfig, T5ForConditionalGeneration, T5Tokenizer
)
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import requests
import json
import re
from bs4 import BeautifulSoup
import arxiv
import scholarly
from .base_model import BaseMultimodalModel
from config import config

class AdvancedResearchAI(BaseMultimodalModel):
    """Advanced Research AI with paper analysis, web search, and knowledge synthesis"""
    
    def __init__(self, device: str = "cuda"):
        super().__init__("research_ai", device)
        self.chat_model = None
        self.embedding_model = None
        self.summarizer = None
        self.classifier = None
        self.knowledge_base = {}
        
    def load_model(self):
        """Load research AI models"""
        if self.is_loaded:
            return
            
        try:
            # Load chat model (Llama 2 or similar)
            model_id = "microsoft/DialoGPT-large"
            
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.chat_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16
            )
            
            # Load embedding model for semantic search
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Load summarization model
            self.summarizer = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Load classification model
            self.classifier = pipeline(
                "text-classification",
                model="microsoft/DialoGPT-medium",
                device=0 if torch.cuda.is_available() else -1
            )
            
            self.is_loaded = True
            self.logger.info("Research AI models loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load research models: {e}")
            raise
    
    def unload_model(self):
        """Unload models from memory"""
        if not self.is_loaded:
            return
            
        del self.chat_model
        del self.embedding_model
        del self.summarizer
        del self.classifier
        
        self.chat_model = None
        self.embedding_model = None
        self.summarizer = None
        self.classifier = None
        self.is_loaded = False
        self.optimize_memory()
    
    def generate(self, *args, **kwargs):
        """Generate research response"""
        return self.chat(*args, **kwargs)
    
    def chat(
        self,
        message: str,
        context: Optional[List[str]] = None,
        max_length: int = 512,
        temperature: float = 0.7,
        do_sample: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """Advanced chat with research capabilities"""
        
        if not self.is_loaded:
            raise RuntimeError("Research AI not loaded")
        
        # Analyze message intent
        intent = self._analyze_intent(message)
        
        # Handle different types of queries
        if intent == "research":
            return self._handle_research_query(message, context)
        elif intent == "paper_analysis":
            return self._handle_paper_analysis(message)
        elif intent == "web_search":
            return self._handle_web_search(message)
        elif intent == "knowledge_synthesis":
            return self._handle_knowledge_synthesis(message, context)
        else:
            return self._handle_general_chat(message, context, max_length, temperature, do_sample)
    
    def _analyze_intent(self, message: str) -> str:
        """Analyze user intent from message"""
        message_lower = message.lower()
        
        research_keywords = ["research", "study", "paper", "analysis", "investigate"]
        paper_keywords = ["arxiv", "doi", "paper", "publication", "journal"]
        search_keywords = ["search", "find", "look up", "web", "internet"]
        synthesis_keywords = ["summarize", "synthesize", "combine", "overview"]
        
        if any(keyword in message_lower for keyword in paper_keywords):
            return "paper_analysis"
        elif any(keyword in message_lower for keyword in research_keywords):
            return "research"
        elif any(keyword in message_lower for keyword in search_keywords):
            return "web_search"
        elif any(keyword in message_lower for keyword in synthesis_keywords):
            return "knowledge_synthesis"
        else:
            return "general_chat"
    
    def _handle_research_query(self, query: str, context: Optional[List[str]] = None) -> Dict[str, Any]:
        """Handle research-specific queries"""
        
        # Search for relevant papers
        papers = self.search_papers(query, max_results=5)
        
        # Extract key information
        research_summary = self._synthesize_research(papers, query)
        
        # Generate response
        response = self._generate_research_response(query, research_summary, papers)
        
        return {
            "response": response,
            "papers": papers,
            "summary": research_summary,
            "type": "research"
        }
    
    def _handle_paper_analysis(self, query: str) -> Dict[str, Any]:
        """Analyze specific papers"""
        
        # Extract paper identifiers (DOI, arXiv ID, etc.)
        paper_ids = self._extract_paper_ids(query)
        
        analyses = []
        for paper_id in paper_ids:
            analysis = self.analyze_paper(paper_id)
            analyses.append(analysis)
        
        # Synthesize analysis
        combined_analysis = self._combine_analyses(analyses)
        
        return {
            "response": combined_analysis,
            "analyses": analyses,
            "type": "paper_analysis"
        }
    
    def _handle_web_search(self, query: str) -> Dict[str, Any]:
        """Handle web search queries"""
        
        search_results = self.web_search(query, max_results=10)
        
        # Summarize search results
        summary = self._summarize_search_results(search_results)
        
        return {
            "response": summary,
            "search_results": search_results,
            "type": "web_search"
        }
    
    def _handle_knowledge_synthesis(self, query: str, context: Optional[List[str]] = None) -> Dict[str, Any]:
        """Synthesize knowledge from multiple sources"""
        
        # Gather information from multiple sources
        papers = self.search_papers(query, max_results=3)
        web_results = self.web_search(query, max_results=5)
        
        # Combine and synthesize
        synthesis = self._synthesize_knowledge(query, papers, web_results, context)
        
        return {
            "response": synthesis,
            "sources": {"papers": papers, "web": web_results},
            "type": "knowledge_synthesis"
        }
    
    def _handle_general_chat(
        self,
        message: str,
        context: Optional[List[str]] = None,
        max_length: int = 512,
        temperature: float = 0.7,
        do_sample: bool = True
    ) -> Dict[str, Any]:
        """Handle general chat messages"""
        
        # Prepare input
        if context:
            input_text = " ".join(context[-3:]) + " " + message
        else:
            input_text = message
        
        # Tokenize
        inputs = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.chat_model.generate(
                inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(input_text):].strip()
        
        return {
            "response": response,
            "type": "general_chat"
        }
    
    def search_papers(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search for academic papers using arXiv and other sources"""
        
        papers = []
        
        try:
            # Search arXiv
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            for result in search.results():
                paper = {
                    "title": result.title,
                    "authors": [author.name for author in result.authors],
                    "abstract": result.summary,
                    "url": result.entry_id,
                    "published": result.published.strftime("%Y-%m-%d"),
                    "categories": result.categories,
                    "source": "arXiv"
                }
                papers.append(paper)
                
        except Exception as e:
            self.logger.warning(f"arXiv search failed: {e}")
        
        try:
            # Search Google Scholar
            search_query = scholarly.search_pubs(query)
            
            for i, pub in enumerate(search_query):
                if i >= max_results // 2:
                    break
                    
                paper = {
                    "title": pub.get("bib", {}).get("title", ""),
                    "authors": pub.get("bib", {}).get("author", []),
                    "abstract": pub.get("bib", {}).get("abstract", ""),
                    "url": pub.get("pub_url", ""),
                    "published": pub.get("bib", {}).get("pub_year", ""),
                    "citations": pub.get("num_citations", 0),
                    "source": "Google Scholar"
                }
                papers.append(paper)
                
        except Exception as e:
            self.logger.warning(f"Google Scholar search failed: {e}")
        
        return papers
    
    def analyze_paper(self, paper_id: str) -> Dict[str, Any]:
        """Analyze a specific paper"""
        
        try:
            # Try to get paper from arXiv
            if "arxiv" in paper_id.lower():
                paper_id = paper_id.split("/")[-1]
                search = arxiv.Search(id_list=[paper_id])
                paper = next(search.results())
                
                # Extract key information
                analysis = {
                    "title": paper.title,
                    "authors": [author.name for author in paper.authors],
                    "abstract": paper.summary,
                    "key_findings": self._extract_key_findings(paper.summary),
                    "methodology": self._extract_methodology(paper.summary),
                    "limitations": self._extract_limitations(paper.summary),
                    "significance": self._assess_significance(paper.summary),
                    "categories": paper.categories
                }
                
                return analysis
                
        except Exception as e:
            self.logger.error(f"Paper analysis failed: {e}")
            return {"error": str(e)}
    
    def web_search(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Perform web search and extract information"""
        
        # Simple web search implementation
        # In production, you'd use a proper search API
        
        search_results = []
        
        try:
            # Use DuckDuckGo search (no API key required)
            import duckduckgo_search
            
            ddg = duckduckgo_search.DDGS()
            results = ddg.text(query, max_results=max_results)
            
            for result in results:
                search_results.append({
                    "title": result.get("title", ""),
                    "url": result.get("href", ""),
                    "snippet": result.get("body", ""),
                    "source": "DuckDuckGo"
                })
                
        except Exception as e:
            self.logger.warning(f"Web search failed: {e}")
        
        return search_results
    
    def _synthesize_research(self, papers: List[Dict[str, Any]], query: str) -> str:
        """Synthesize research findings from multiple papers"""
        
        if not papers:
            return "No relevant papers found."
        
        # Extract abstracts
        abstracts = [paper.get("abstract", "") for paper in papers if paper.get("abstract")]
        
        if not abstracts:
            return "No abstracts available for synthesis."
        
        # Combine abstracts
        combined_text = " ".join(abstracts)
        
        # Summarize
        try:
            summary = self.summarizer(
                combined_text,
                max_length=300,
                min_length=100,
                do_sample=False
            )[0]["summary_text"]
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Research synthesis failed: {e}")
            return "Failed to synthesize research findings."
    
    def _generate_research_response(
        self,
        query: str,
        summary: str,
        papers: List[Dict[str, Any]]
    ) -> str:
        """Generate comprehensive research response"""
        
        response = f"Based on my research on '{query}', here's what I found:\n\n"
        response += f"**Summary:** {summary}\n\n"
        
        if papers:
            response += "**Key Papers:**\n"
            for i, paper in enumerate(papers[:3], 1):
                response += f"{i}. {paper.get('title', 'Unknown Title')}\n"
                response += f"   Authors: {', '.join(paper.get('authors', []))}\n"
                response += f"   Source: {paper.get('source', 'Unknown')}\n\n"
        
        return response
    
    def _extract_key_findings(self, text: str) -> List[str]:
        """Extract key findings from paper text"""
        
        # Simple keyword-based extraction
        findings_keywords = ["found", "discovered", "showed", "demonstrated", "revealed"]
        sentences = text.split(".")
        
        findings = []
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in findings_keywords):
                findings.append(sentence.strip())
        
        return findings[:3]  # Return top 3 findings
    
    def _extract_methodology(self, text: str) -> str:
        """Extract methodology from paper text"""
        
        method_keywords = ["method", "approach", "technique", "algorithm", "model"]
        sentences = text.split(".")
        
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in method_keywords):
                return sentence.strip()
        
        return "Methodology not clearly identified."
    
    def _extract_limitations(self, text: str) -> List[str]:
        """Extract limitations from paper text"""
        
        limit_keywords = ["limitation", "constraint", "challenge", "issue", "problem"]
        sentences = text.split(".")
        
        limitations = []
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in limit_keywords):
                limitations.append(sentence.strip())
        
        return limitations[:2]  # Return top 2 limitations
    
    def _assess_significance(self, text: str) -> str:
        """Assess the significance of the research"""
        
        significance_keywords = ["significant", "important", "novel", "breakthrough", "advance"]
        
        if any(keyword in text.lower() for keyword in significance_keywords):
            return "High significance"
        else:
            return "Moderate significance"
    
    def _extract_paper_ids(self, text: str) -> List[str]:
        """Extract paper identifiers from text"""
        
        # Extract arXiv IDs
        arxiv_pattern = r"arxiv:(\d{4}\.\d{4,5})"
        arxiv_ids = re.findall(arxiv_pattern, text, re.IGNORECASE)
        
        # Extract DOIs
        doi_pattern = r"10\.\d{4,}/[^\s]+"
        dois = re.findall(doi_pattern, text)
        
        return arxiv_ids + dois
    
    def _combine_analyses(self, analyses: List[Dict[str, Any]]) -> str:
        """Combine multiple paper analyses"""
        
        if not analyses:
            return "No analyses available."
        
        combined = "**Combined Paper Analysis:**\n\n"
        
        for i, analysis in enumerate(analyses, 1):
            combined += f"**Paper {i}: {analysis.get('title', 'Unknown')}**\n"
            combined += f"Key Findings: {', '.join(analysis.get('key_findings', []))}\n"
            combined += f"Methodology: {analysis.get('methodology', 'N/A')}\n"
            combined += f"Significance: {analysis.get('significance', 'N/A')}\n\n"
        
        return combined
    
    def _summarize_search_results(self, results: List[Dict[str, Any]]) -> str:
        """Summarize web search results"""
        
        if not results:
            return "No search results found."
        
        # Combine snippets
        snippets = [result.get("snippet", "") for result in results if result.get("snippet")]
        combined_text = " ".join(snippets)
        
        if not combined_text:
            return "No content available for summarization."
        
        try:
            summary = self.summarizer(
                combined_text,
                max_length=200,
                min_length=50,
                do_sample=False
            )[0]["summary_text"]
            
            return f"**Web Search Summary:** {summary}"
            
        except Exception as e:
            self.logger.error(f"Search result summarization failed: {e}")
            return "Failed to summarize search results."
    
    def _synthesize_knowledge(
        self,
        query: str,
        papers: List[Dict[str, Any]],
        web_results: List[Dict[str, Any]],
        context: Optional[List[str]] = None
    ) -> str:
        """Synthesize knowledge from multiple sources"""
        
        synthesis = f"**Knowledge Synthesis for: {query}**\n\n"
        
        # Academic perspective
        if papers:
            research_summary = self._synthesize_research(papers, query)
            synthesis += f"**Academic Research:**\n{research_summary}\n\n"
        
        # Web perspective
        if web_results:
            web_summary = self._summarize_search_results(web_results)
            synthesis += f"**Current Information:**\n{web_summary}\n\n"
        
        # Context integration
        if context:
            synthesis += f"**Contextual Analysis:**\n"
            synthesis += "Based on our previous discussion and current research, "
            synthesis += "the key insights are interconnected and suggest...\n\n"
        
        synthesis += "**Conclusion:**\n"
        synthesis += "This synthesis combines academic research with current information "
        synthesis += "to provide a comprehensive understanding of the topic."
        
        return synthesis