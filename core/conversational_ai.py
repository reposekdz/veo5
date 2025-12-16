import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM,
    BlenderbotTokenizer, BlenderbotForConditionalGeneration,
    GPT2LMHeadModel, GPT2Tokenizer, BitsAndBytesConfig
)
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import json
import re
from datetime import datetime
from .base_model import BaseMultimodalModel

class AdvancedConversationalAI(BaseMultimodalModel):
    """Advanced conversational AI with memory, personality, and multi-turn dialogue"""
    
    def __init__(self, device: str = "cuda"):
        super().__init__("conversational_ai", device)
        self.chat_model = None
        self.tokenizer = None
        self.conversation_memory = []
        self.user_profile = {}
        self.personality_traits = {
            "helpful": 0.9,
            "creative": 0.8,
            "analytical": 0.85,
            "empathetic": 0.7,
            "humorous": 0.6
        }
        
    def load_model(self):
        """Load conversational AI models"""
        if self.is_loaded:
            return
            
        try:
            # Load main chat model (using DialoGPT or similar)
            model_id = "microsoft/DialoGPT-large"
            
            # Quantization for memory efficiency
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.chat_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            
            # Load additional specialized models
            self._load_specialized_models()
            
            self.is_loaded = True
            self.logger.info("Conversational AI models loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load conversational models: {e}")
            # Fallback to smaller model
            self._load_fallback_model()
    
    def _load_specialized_models(self):
        """Load specialized models for different conversation types"""
        try:
            # Creative writing model
            self.creative_tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
            self.creative_model = GPT2LMHeadModel.from_pretrained("gpt2-medium").to(self.device)
            
            # Question answering model
            from transformers import pipeline
            self.qa_pipeline = pipeline(
                "question-answering",
                model="distilbert-base-cased-distilled-squad",
                device=0 if torch.cuda.is_available() else -1
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to load specialized models: {e}")
    
    def _load_fallback_model(self):
        """Load fallback model if main model fails"""
        try:
            model_id = "microsoft/DialoGPT-medium"
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.chat_model = AutoModelForCausalLM.from_pretrained(model_id).to(self.device)
            self.is_loaded = True
            self.logger.info("Loaded fallback conversational model")
        except Exception as e:
            self.logger.error(f"Fallback model loading failed: {e}")
            raise
    
    def unload_model(self):
        """Unload models from memory"""
        if not self.is_loaded:
            return
            
        del self.chat_model
        del self.tokenizer
        if hasattr(self, 'creative_model'):
            del self.creative_model
            del self.creative_tokenizer
        if hasattr(self, 'qa_pipeline'):
            del self.qa_pipeline
            
        self.chat_model = None
        self.tokenizer = None
        self.is_loaded = False
        self.optimize_memory()
    
    def generate(self, *args, **kwargs):
        """Generate conversational response"""
        return self.chat(*args, **kwargs)
    
    def chat(
        self,
        message: str,
        user_id: str = "default",
        conversation_id: str = "default",
        max_length: int = 512,
        temperature: float = 0.8,
        top_p: float = 0.9,
        do_sample: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """Advanced multi-turn conversation with memory and personality"""
        
        if not self.is_loaded:
            raise RuntimeError("Conversational AI not loaded")
        
        # Update user profile
        self._update_user_profile(user_id, message)
        
        # Analyze message intent and emotion
        intent = self._analyze_intent(message)
        emotion = self._detect_emotion(message)
        
        # Get conversation context
        context = self._get_conversation_context(conversation_id)
        
        # Generate response based on intent
        if intent == "creative":
            response = self._generate_creative_response(message, context, temperature)
        elif intent == "question":
            response = self._generate_qa_response(message, context)
        elif intent == "casual":
            response = self._generate_casual_response(message, context, temperature, top_p)
        elif intent == "emotional":
            response = self._generate_empathetic_response(message, emotion, context)
        else:
            response = self._generate_default_response(message, context, max_length, temperature, top_p, do_sample)
        
        # Store conversation
        self._store_conversation(conversation_id, message, response, intent, emotion)
        
        return {
            "response": response,
            "intent": intent,
            "emotion": emotion,
            "conversation_id": conversation_id,
            "user_id": user_id,
            "timestamp": datetime.now().isoformat()
        }
    
    def _analyze_intent(self, message: str) -> str:
        """Analyze user intent from message"""
        message_lower = message.lower()
        
        # Creative intents
        creative_keywords = ["write", "create", "story", "poem", "creative", "imagine"]
        if any(keyword in message_lower for keyword in creative_keywords):
            return "creative"
        
        # Question intents
        question_keywords = ["what", "how", "why", "when", "where", "who", "?"]
        if any(keyword in message_lower for keyword in question_keywords):
            return "question"
        
        # Emotional intents
        emotional_keywords = ["feel", "sad", "happy", "angry", "worried", "excited"]
        if any(keyword in message_lower for keyword in emotional_keywords):
            return "emotional"
        
        # Casual conversation
        casual_keywords = ["hello", "hi", "thanks", "bye", "chat", "talk"]
        if any(keyword in message_lower for keyword in casual_keywords):
            return "casual"
        
        return "general"
    
    def _detect_emotion(self, message: str) -> str:
        """Detect emotion in user message"""
        message_lower = message.lower()
        
        # Simple emotion detection
        emotions = {
            "happy": ["happy", "joy", "excited", "great", "awesome", "wonderful"],
            "sad": ["sad", "depressed", "down", "unhappy", "terrible", "awful"],
            "angry": ["angry", "mad", "furious", "annoyed", "frustrated"],
            "worried": ["worried", "anxious", "concerned", "nervous", "scared"],
            "neutral": []
        }
        
        for emotion, keywords in emotions.items():
            if any(keyword in message_lower for keyword in keywords):
                return emotion
        
        return "neutral"
    
    def _get_conversation_context(self, conversation_id: str, max_turns: int = 5) -> List[Dict[str, str]]:
        """Get recent conversation context"""
        context = []
        for turn in self.conversation_memory:
            if turn["conversation_id"] == conversation_id:
                context.append({
                    "user": turn["user_message"],
                    "assistant": turn["assistant_response"]
                })
        
        return context[-max_turns:]  # Return last N turns
    
    def _generate_creative_response(
        self,
        message: str,
        context: List[Dict[str, str]],
        temperature: float = 1.0
    ) -> str:
        """Generate creative response using specialized model"""
        
        if not hasattr(self, 'creative_model'):
            return self._generate_default_response(message, context, temperature=temperature)
        
        try:
            # Prepare creative prompt
            prompt = self._build_creative_prompt(message, context)
            
            # Tokenize
            inputs = self.creative_tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.creative_model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 200,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.creative_tokenizer.eos_token_id
                )
            
            # Decode
            response = self.creative_tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response[len(prompt):].strip()
            
            return self._post_process_response(response)
            
        except Exception as e:
            self.logger.error(f"Creative generation failed: {e}")
            return self._generate_default_response(message, context)
    
    def _generate_qa_response(self, message: str, context: List[Dict[str, str]]) -> str:
        """Generate question-answering response"""
        
        if not hasattr(self, 'qa_pipeline'):
            return self._generate_default_response(message, context)
        
        try:
            # Build context from conversation
            context_text = " ".join([turn["assistant"] for turn in context[-3:]])
            
            if not context_text:
                context_text = "I am an AI assistant designed to help with various tasks and questions."
            
            # Use QA pipeline
            result = self.qa_pipeline(question=message, context=context_text)
            
            if result["score"] > 0.3:  # Confidence threshold
                return f"{result['answer']} (Confidence: {result['score']:.2f})"
            else:
                return self._generate_default_response(message, context)
                
        except Exception as e:
            self.logger.error(f"QA generation failed: {e}")
            return self._generate_default_response(message, context)
    
    def _generate_casual_response(
        self,
        message: str,
        context: List[Dict[str, str]],
        temperature: float = 0.8,
        top_p: float = 0.9
    ) -> str:
        """Generate casual conversation response"""
        
        # Add personality to casual responses
        personality_prompt = self._build_personality_prompt()
        
        return self._generate_default_response(
            message, context, 
            temperature=temperature, 
            top_p=top_p,
            personality_prompt=personality_prompt
        )
    
    def _generate_empathetic_response(
        self,
        message: str,
        emotion: str,
        context: List[Dict[str, str]]
    ) -> str:
        """Generate empathetic response based on detected emotion"""
        
        empathy_prompts = {
            "sad": "I understand you're feeling down. ",
            "angry": "I can sense your frustration. ",
            "worried": "I hear that you're concerned about this. ",
            "happy": "I'm glad to hear you're feeling positive! "
        }
        
        empathy_prefix = empathy_prompts.get(emotion, "I understand. ")
        
        # Generate base response
        base_response = self._generate_default_response(message, context, temperature=0.7)
        
        return empathy_prefix + base_response
    
    def _generate_default_response(
        self,
        message: str,
        context: List[Dict[str, str]],
        max_length: int = 512,
        temperature: float = 0.8,
        top_p: float = 0.9,
        do_sample: bool = True,
        personality_prompt: str = ""
    ) -> str:
        """Generate default conversational response"""
        
        # Build conversation history
        conversation_history = ""
        for turn in context[-3:]:  # Last 3 turns
            conversation_history += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n"
        
        # Add current message
        full_prompt = personality_prompt + conversation_history + f"User: {message}\nAssistant:"
        
        # Tokenize
        inputs = self.tokenizer.encode(full_prompt, return_tensors="pt").to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.chat_model.generate(
                inputs,
                max_length=min(inputs.shape[1] + 150, max_length),
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the new response
        response = response[len(full_prompt):].strip()
        
        return self._post_process_response(response)
    
    def _build_creative_prompt(self, message: str, context: List[Dict[str, str]]) -> str:
        """Build prompt for creative generation"""
        
        creative_prefix = "You are a creative AI assistant. Generate imaginative and engaging content.\n\n"
        
        # Add context
        context_text = ""
        for turn in context[-2:]:
            context_text += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n"
        
        return creative_prefix + context_text + f"User: {message}\nAssistant:"
    
    def _build_personality_prompt(self) -> str:
        """Build personality prompt based on traits"""
        
        traits = []
        if self.personality_traits["helpful"] > 0.8:
            traits.append("helpful")
        if self.personality_traits["creative"] > 0.7:
            traits.append("creative")
        if self.personality_traits["analytical"] > 0.8:
            traits.append("analytical")
        if self.personality_traits["empathetic"] > 0.6:
            traits.append("empathetic")
        if self.personality_traits["humorous"] > 0.5:
            traits.append("humorous")
        
        if traits:
            return f"You are a {', '.join(traits)} AI assistant. "
        else:
            return "You are a friendly AI assistant. "
    
    def _post_process_response(self, response: str) -> str:
        """Post-process generated response"""
        
        # Remove repetitions
        sentences = response.split(".")
        unique_sentences = []
        for sentence in sentences:
            if sentence.strip() and sentence.strip() not in unique_sentences:
                unique_sentences.append(sentence.strip())
        
        response = ". ".join(unique_sentences)
        
        # Ensure proper ending
        if response and not response.endswith((".", "!", "?")):
            response += "."
        
        # Limit length
        if len(response) > 500:
            response = response[:500] + "..."
        
        return response
    
    def _update_user_profile(self, user_id: str, message: str):
        """Update user profile based on interactions"""
        
        if user_id not in self.user_profile:
            self.user_profile[user_id] = {
                "message_count": 0,
                "interests": [],
                "communication_style": "neutral",
                "preferred_topics": []
            }
        
        profile = self.user_profile[user_id]
        profile["message_count"] += 1
        
        # Extract interests (simple keyword matching)
        interests_keywords = {
            "technology": ["tech", "computer", "AI", "programming", "software"],
            "science": ["science", "research", "study", "experiment", "theory"],
            "art": ["art", "creative", "design", "music", "painting"],
            "sports": ["sport", "game", "play", "team", "competition"]
        }
        
        message_lower = message.lower()
        for interest, keywords in interests_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                if interest not in profile["interests"]:
                    profile["interests"].append(interest)
    
    def _store_conversation(
        self,
        conversation_id: str,
        user_message: str,
        assistant_response: str,
        intent: str,
        emotion: str
    ):
        """Store conversation turn in memory"""
        
        turn = {
            "conversation_id": conversation_id,
            "user_message": user_message,
            "assistant_response": assistant_response,
            "intent": intent,
            "emotion": emotion,
            "timestamp": datetime.now().isoformat()
        }
        
        self.conversation_memory.append(turn)
        
        # Keep only recent conversations (memory management)
        if len(self.conversation_memory) > 1000:
            self.conversation_memory = self.conversation_memory[-500:]
    
    def get_conversation_history(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Get full conversation history"""
        
        history = []
        for turn in self.conversation_memory:
            if turn["conversation_id"] == conversation_id:
                history.append(turn)
        
        return history
    
    def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """Get user profile"""
        
        return self.user_profile.get(user_id, {})
    
    def set_personality(self, traits: Dict[str, float]):
        """Set personality traits"""
        
        for trait, value in traits.items():
            if trait in self.personality_traits:
                self.personality_traits[trait] = max(0.0, min(1.0, value))
    
    def clear_conversation(self, conversation_id: str):
        """Clear specific conversation"""
        
        self.conversation_memory = [
            turn for turn in self.conversation_memory 
            if turn["conversation_id"] != conversation_id
        ]
    
    def get_conversation_stats(self) -> Dict[str, Any]:
        """Get conversation statistics"""
        
        total_conversations = len(set(turn["conversation_id"] for turn in self.conversation_memory))
        total_turns = len(self.conversation_memory)
        
        # Intent distribution
        intents = [turn["intent"] for turn in self.conversation_memory]
        intent_counts = {intent: intents.count(intent) for intent in set(intents)}
        
        # Emotion distribution
        emotions = [turn["emotion"] for turn in self.conversation_memory]
        emotion_counts = {emotion: emotions.count(emotion) for emotion in set(emotions)}
        
        return {
            "total_conversations": total_conversations,
            "total_turns": total_turns,
            "intent_distribution": intent_counts,
            "emotion_distribution": emotion_counts,
            "active_users": len(self.user_profile),
            "personality_traits": self.personality_traits
        }