from fastapi import APIRouter, HTTPException, Query, Body
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import sys
sys.path.append('..')
from core.multimodal_ai import MultimodalAI

# Initialize router
router = APIRouter(prefix="/research", tags=["research"])

# Initialize AI system (shared instance)
ai_system = None

def get_ai_system():
    global ai_system
    if ai_system is None:
        ai_system = MultimodalAI()
    return ai_system

# Request models
class ResearchQuery(BaseModel):
    query: str = Field(..., description="Research query")
    max_results: int = Field(10, ge=1, le=50, description="Maximum number of results")
    include_papers: bool = Field(True, description="Include academic papers")
    include_web: bool = Field(True, description="Include web search")

class ChatMessage(BaseModel):
    message: str = Field(..., description="Chat message")
    user_id: str = Field("default", description="User ID")
    conversation_id: str = Field("default", description="Conversation ID")
    temperature: float = Field(0.8, ge=0.1, le=2.0, description="Response creativity")

class PaperAnalysisRequest(BaseModel):
    paper_id: str = Field(..., description="Paper ID (arXiv ID or DOI)")

class WebSearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    max_results: int = Field(10, ge=1, le=20, description="Maximum results")

class KnowledgeBaseQuery(BaseModel):
    query: str = Field(..., description="Knowledge base query")
    top_k: int = Field(5, ge=1, le=20, description="Number of results")
    category: Optional[str] = Field(None, description="Filter by category")
    threshold: float = Field(0.3, ge=0.0, le=1.0, description="Similarity threshold")

class PersonalityUpdate(BaseModel):
    helpful: Optional[float] = Field(None, ge=0.0, le=1.0)
    creative: Optional[float] = Field(None, ge=0.0, le=1.0)
    analytical: Optional[float] = Field(None, ge=0.0, le=1.0)
    empathetic: Optional[float] = Field(None, ge=0.0, le=1.0)
    humorous: Optional[float] = Field(None, ge=0.0, le=1.0)

# Research endpoints
@router.post("/query")
async def research_query(request: ResearchQuery):
    """Perform comprehensive research query"""
    try:
        ai = get_ai_system()
        result = ai.research(
            query=request.query,
            max_results=request.max_results
        )
        
        return {
            "status": "success",
            "result": result,
            "query": request.query
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/papers/search")
async def search_papers(
    query: str = Query(..., description="Search query"),
    max_results: int = Query(10, ge=1, le=50, description="Maximum results")
):
    """Search academic papers"""
    try:
        ai = get_ai_system()
        papers = ai.search_papers(query, max_results)
        
        return {
            "status": "success",
            "papers": papers,
            "total": len(papers),
            "query": query
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/papers/analyze")
async def analyze_paper(request: PaperAnalysisRequest):
    """Analyze specific academic paper"""
    try:
        ai = get_ai_system()
        analysis = ai.analyze_paper(request.paper_id)
        
        return {
            "status": "success",
            "analysis": analysis,
            "paper_id": request.paper_id
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/web/search")
async def web_search(request: WebSearchRequest):
    """Perform web search"""
    try:
        ai = get_ai_system()
        results = ai.web_search(request.query, request.max_results)
        
        return {
            "status": "success",
            "results": results,
            "total": len(results),
            "query": request.query
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Chat endpoints
@router.post("/chat")
async def chat_message(request: ChatMessage):
    """Send message to conversational AI"""
    try:
        ai = get_ai_system()
        response = ai.chat(
            message=request.message,
            user_id=request.user_id,
            conversation_id=request.conversation_id,
            temperature=request.temperature
        )
        
        return {
            "status": "success",
            "response": response
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/chat/history/{conversation_id}")
async def get_chat_history(conversation_id: str):
    """Get conversation history"""
    try:
        ai = get_ai_system()
        history = ai.get_conversation_history(conversation_id)
        
        return {
            "status": "success",
            "history": history,
            "conversation_id": conversation_id,
            "total_turns": len(history)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat/personality")
async def update_personality(request: PersonalityUpdate):
    """Update AI personality traits"""
    try:
        ai = get_ai_system()
        
        # Build traits dictionary
        traits = {}
        for trait, value in request.dict().items():
            if value is not None:
                traits[trait] = value
        
        ai.set_personality(traits)
        
        return {
            "status": "success",
            "updated_traits": traits,
            "message": "Personality updated successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Knowledge base endpoints
@router.post("/knowledge/search")
async def search_knowledge_base(request: KnowledgeBaseQuery):
    """Search knowledge base"""
    try:
        ai = get_ai_system()
        
        # Load knowledge base model
        kb_model = ai.model_manager.load_model("knowledge_base")
        
        results = kb_model.search(
            query=request.query,
            top_k=request.top_k,
            category=request.category,
            threshold=request.threshold
        )
        
        return {
            "status": "success",
            "results": results,
            "total": len(results),
            "query": request.query
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/knowledge/stats")
async def get_knowledge_base_stats():
    """Get knowledge base statistics"""
    try:
        ai = get_ai_system()
        kb_model = ai.model_manager.load_model("knowledge_base")
        stats = kb_model.get_stats()
        
        return {
            "status": "success",
            "stats": stats
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/knowledge/categories")
async def get_knowledge_categories():
    """Get all knowledge base categories"""
    try:
        ai = get_ai_system()
        kb_model = ai.model_manager.load_model("knowledge_base")
        categories = kb_model.get_categories()
        
        return {
            "status": "success",
            "categories": categories,
            "total": len(categories)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Advanced research endpoints
@router.post("/synthesis")
async def knowledge_synthesis(
    query: str = Body(..., description="Synthesis query"),
    sources: List[str] = Body(["papers", "web"], description="Sources to include"),
    max_sources: int = Body(10, ge=1, le=20, description="Max sources per type")
):
    """Synthesize knowledge from multiple sources"""
    try:
        ai = get_ai_system()
        
        # Gather information from requested sources
        papers = []
        web_results = []
        
        if "papers" in sources:
            papers = ai.search_papers(query, max_sources)
        
        if "web" in sources:
            web_results = ai.web_search(query, max_sources)
        
        # Perform synthesis using research AI
        synthesis_result = ai.research(
            f"Synthesize knowledge about: {query}",
            context=[f"Papers: {len(papers)} found", f"Web results: {len(web_results)} found"]
        )
        
        return {
            "status": "success",
            "synthesis": synthesis_result,
            "sources_used": {
                "papers": len(papers),
                "web_results": len(web_results)
            },
            "query": query
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/trending")
async def get_trending_topics(
    category: str = Query("general", description="Category filter"),
    limit: int = Query(10, ge=1, le=50, description="Number of topics")
):
    """Get trending research topics"""
    try:
        # This would typically analyze recent papers and searches
        # For now, return mock trending topics
        
        trending_topics = [
            {"topic": "Large Language Models", "papers": 1250, "growth": "+45%"},
            {"topic": "Computer Vision", "papers": 980, "growth": "+32%"},
            {"topic": "Reinforcement Learning", "papers": 756, "growth": "+28%"},
            {"topic": "Neural Architecture Search", "papers": 543, "growth": "+67%"},
            {"topic": "Multimodal AI", "papers": 432, "growth": "+89%"},
            {"topic": "Federated Learning", "papers": 387, "growth": "+23%"},
            {"topic": "Graph Neural Networks", "papers": 321, "growth": "+41%"},
            {"topic": "Quantum Computing", "papers": 298, "growth": "+15%"},
            {"topic": "Edge AI", "papers": 276, "growth": "+52%"},
            {"topic": "Explainable AI", "papers": 234, "growth": "+38%"}
        ]
        
        return {
            "status": "success",
            "trending_topics": trending_topics[:limit],
            "category": category,
            "last_updated": "2024-01-15T10:00:00Z"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/recommendations")
async def get_research_recommendations(
    user_id: str = Query("default", description="User ID"),
    limit: int = Query(5, ge=1, le=20, description="Number of recommendations")
):
    """Get personalized research recommendations"""
    try:
        ai = get_ai_system()
        
        # Get user's conversation history to understand interests
        history = ai.get_conversation_history(f"research_{user_id}")
        
        # Extract topics from recent conversations
        recent_topics = []
        for turn in history[-10:]:  # Last 10 turns
            message = turn.get("user_message", "")
            if len(message) > 10:
                recent_topics.append(message[:100])
        
        if not recent_topics:
            # Default recommendations
            recommendations = [
                {"title": "Introduction to Machine Learning", "type": "paper", "relevance": 0.8},
                {"title": "Deep Learning Fundamentals", "type": "paper", "relevance": 0.7},
                {"title": "AI Ethics and Safety", "type": "paper", "relevance": 0.6}
            ]
        else:
            # Generate recommendations based on interests
            recommendations = []
            for i, topic in enumerate(recent_topics[:limit]):
                recommendations.append({
                    "title": f"Related research to: {topic[:50]}...",
                    "type": "paper",
                    "relevance": 0.9 - (i * 0.1)
                })
        
        return {
            "status": "success",
            "recommendations": recommendations,
            "user_id": user_id,
            "based_on_history": len(recent_topics) > 0
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))