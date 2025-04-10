import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import pickle
import os
import time
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
import json
import logging
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure required packages are installed
try:
    import transformers
    import torch
    import accelerate
    import fastapi
    import uvicorn
    import pydantic
except ImportError:
    print("Required packages not found. Please install them with:")
    print("pip install transformers torch accelerate fastapi uvicorn pydantic")
    exit(1)

# Hugging Face Token
hf_token = ""

# File to store memory bank
MEMORY_FILE = "memory_bank.pkl"

# Configuration for new features
USE_ADVANCED_MEMORY = True
MEMORY_FADE_RATE = 0.99
DUPLICATE_THRESHOLD = 0.95

# Load Tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it", token=hf_token)
except Exception as e:
    logger.error(f"Failed to load tokenizer: {e}")
    exit(1)

# Check if MPS (Metal) is available, otherwise fall back to CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Load Base Model
try:
    base_model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-3-1b-it",
        torch_dtype=torch.float16,
        device_map={"": device},
        token=hf_token
    ).eval()
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    exit(1)

# Neural Graffiti Spray Layer
class SprayLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.W = nn.Linear(dim, dim)
        self.lambda_base = nn.Parameter(torch.ones(dim) * 0.1)
        self.register_buffer('state', torch.zeros(dim))

    def forward(self, x, recency_factor=1.0):
        lambda_ = self.lambda_base * recency_factor
        dx = -lambda_ * (self.state - self.W(x))
        self.state = self.state + dx
        return self.state

# Graffiti Adapter Module
class GraffitiAdapter(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.spray = SprayLayer(hidden_size).half()
        self.alpha = nn.Parameter(torch.tensor(0.2).half())
        if USE_ADVANCED_MEMORY:
            self.attention = nn.MultiheadAttention(hidden_size, num_heads=4).half()

    def forward(self, hidden, memory_embed, recency_factor=1.0):
        spray_vector = self.spray(memory_embed, recency_factor)
        if USE_ADVANCED_MEMORY and hasattr(self, 'attention'):
            spray_vector = spray_vector.unsqueeze(0).unsqueeze(0)
            attn_output, _ = self.attention(spray_vector, spray_vector, spray_vector)
            spray_vector = attn_output.squeeze(0).squeeze(0)
        return hidden + self.alpha * spray_vector.unsqueeze(0).unsqueeze(1)

# Memory Functions
memory_bank = []

def load_memory_bank(filename=MEMORY_FILE):
    global memory_bank
    if os.path.exists(filename):
        try:
            with open(filename, "rb") as f:
                memory_bank = pickle.load(f)
            memory_bank = [(t[0].to(device), t[1], time.time(), 1.0, t[2] if len(t) > 2 else "misc") 
                           for t in memory_bank]
            logger.info(f"Loaded memory bank from {filename} with {len(memory_bank)} entries.")
        except Exception as e:
            logger.error(f"Error loading memory bank: {e}")
            memory_bank = []
    else:
        logger.info(f"No memory bank file found at {filename}. Starting with empty memory bank.")
        memory_bank = []

def save_memory_bank(filename=MEMORY_FILE):
    try:
        global memory_bank
        pre_prune_count = len(memory_bank)
        current_time = time.time()
        memory_bank = [mem for mem in memory_bank if 
                       mem[3] * (MEMORY_FADE_RATE ** ((current_time - mem[2]) / 3600)) >= 0.1]
        logger.info(f"Pruned {pre_prune_count - len(memory_bank)} faded memories.")
        
        save_data = [(mem[0], mem[1], mem[4]) for mem in memory_bank]
        with open(filename, "wb") as f:
            pickle.dump(save_data, f)
        logger.info(f"Saved memory bank to {filename} with {len(memory_bank)} entries.")
    except Exception as e:
        logger.error(f"Error saving memory bank: {e}")

def extract_topic(sentence):
    sentence_lower = sentence.strip().lower()
    
    # Helper function to extract key term or proper noun
    def get_key_term(text_lower, text_original):
        stopwords = {"is", "am", "are", "was", "were", "the", "a", "an", "in", "to", "and", "do", "you"}
        words = text_lower.split()
        original_words = text_original.strip().split()  # Preserve original capitalization
        
        # Check for proper nouns (capitalized words)
        proper_nouns = []
        for i, word in enumerate(original_words):
            if word[0].isupper() and word.lower() not in stopwords:
                # Check for multi-word proper nouns
                if i > 0 and original_words[i-1][0].isupper():
                    proper_nouns[-1] = f"{proper_nouns[-1]}_{word.lower()}"
                else:
                    proper_nouns.append(word.lower())
        if proper_nouns:
            return "_".join(proper_nouns[:2])  # Limit to two words
        
        # Fallback to significant non-stopword
        for word in words:
            if word not in stopwords:
                return word
        return "misc"
    
    # Handle questions
    if any(q in sentence_lower for q in ["what", "where", "who", "do"]):
        question_match = re.search(r"(?:what|where|who|do)(?:'s| is)? my (\w+(?:\s+\w+)?)", sentence_lower)
        if question_match:
            return f"my_{question_match.group(1)}"
        # Extract key term for "do you know X" pattern
        know_match = re.search(r"do you know ([\w\s]+?)(?:\?|$)", sentence_lower)
        if know_match:
            key_phrase = know_match.group(1).strip()
            # Check for proper nouns in the phrase
            phrase_start_idx = sentence_lower.index("do you know") + len("do you know ")
            original_phrase = sentence.strip()[phrase_start_idx:]
            original_words = original_phrase.split()
            proper_nouns = [w.lower() for w in original_words if w[0].isupper()]
            if proper_nouns:
                return "_".join(proper_nouns[:2])  # e.g., "david_potter"
            return key_phrase.split()[-1] if key_phrase else "misc"
    
    # Generate tag from key term
    tag = get_key_term(sentence_lower, sentence)
    
    # Refine tag with context if applicable
    if "like" in sentence_lower or "enjoy" in sentence_lower:
        tag = f"pref_{tag}"
    elif "live" in sentence_lower or "from" in sentence_lower:
        tag = f"loc_{tag}"
    elif "feel" in sentence_lower:
        tag = f"emo_{tag}"
    
    return tag if tag != "misc" or not tag else "misc"

def store_memory(embedding, text):
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    for sentence in sentences:
        # Dynamically generate tag
        tag = extract_topic(sentence)
        
        # Assign importance based on sentence length and keywords
        importance = 2.0 if any(kw in sentence.lower() for kw in ["name", "live", "favorite"]) else 1.0
        importance += len(sentence.split()) / 10.0
        
        if USE_ADVANCED_MEMORY:
            same_tag_memories = [mem for mem in memory_bank if mem[4] == tag]
            if same_tag_memories:
                embeddings = torch.stack([mem[0] for mem in same_tag_memories]).to(device)
                similarities = F.cosine_similarity(embedding.unsqueeze(0), embeddings, dim=1)
                max_sim, max_idx = similarities.max(), similarities.argmax()
                if max_sim > DUPLICATE_THRESHOLD:
                    old_mem = memory_bank[max_idx]
                    new_weight = old_mem[3] + 0.5
                    memory_bank[max_idx] = (old_mem[0], old_mem[1], time.time(), new_weight, tag)
                    logger.info(f"Reinforced memory: '{sentence}' with tag '{tag}', new weight {new_weight}")
                    continue
        
        memory_bank.append((embedding.detach().clone(), sentence, time.time(), importance, tag))
        logger.info(f"Stored memory: '{sentence}' with tag '{tag}' and importance {importance}")

def recall_memory(query_embedding, query_text, top_k=3):
    if not memory_bank:
        return []
    
    # Dynamically generate tag for the query
    query_tag = extract_topic(query_text)
    logger.info(f"Generated query tag: '{query_tag}' for '{query_text}'")
    
    embeddings = torch.stack([mem[0] for mem in memory_bank]).to(device)
    similarities = F.cosine_similarity(query_embedding.unsqueeze(0), embeddings, dim=1)
    
    if USE_ADVANCED_MEMORY:
        current_time = time.time()
        weights = torch.tensor([mem[3] * (MEMORY_FADE_RATE ** ((current_time - mem[2]) / 3600)) 
                              for mem in memory_bank], device=device)
        similarities = similarities * weights

    # Exact or partial tag matching
    exact_matches = [i for i, mem in enumerate(memory_bank) if mem[4] == query_tag]
    if exact_matches:
        tag_similarities = similarities[exact_matches]
        top_k_indices = torch.topk(tag_similarities, min(top_k, len(exact_matches)), dim=0).indices
        top_k_indices = [exact_matches[idx] for idx in top_k_indices.tolist()]
    else:
        # Fallback to semantic similarity if no exact tag match
        top_k_indices = torch.topk(similarities, min(top_k, len(memory_bank)), dim=0).indices.tolist()

    recalled = [memory_bank[i] for i in top_k_indices]
    logger.info(f"Recalled {len(recalled)} memories for tag '{query_tag}': {[mem[1] for mem in recalled]}")
    return recalled

def fuse_embeddings(current, recalled):
    if recalled:
        vectors = torch.stack([current] + [r[0] for r in recalled], dim=0)
        return torch.mean(vectors, dim=0)
    return current

# Wrapped Gemma Model with Graffiti Adapter
class GraffitiWrappedModel(nn.Module):
    def __init__(self, base_model, graffiti_adapter):
        super().__init__()
        self.base_model = base_model
        self.graffiti_adapter = graffiti_adapter

    def forward(self, input_ids, memory_embed=None, recency_factor=1.0, **kwargs):
        outputs = self.base_model.model(
            input_ids=input_ids,
            output_hidden_states=True,
            **kwargs
        )
        hidden_states = outputs.hidden_states[-1]
        if memory_embed is not None:
            hidden_states = self.graffiti_adapter(hidden_states, memory_embed, recency_factor)
        logits = self.base_model.lm_head(hidden_states)
        return logits

# Initialize Adapter and Model
graffiti_adapter = GraffitiAdapter(hidden_size=base_model.config.hidden_size).to(device)
model = GraffitiWrappedModel(base_model, graffiti_adapter).eval()

# Graffiti Text Generator
@torch.no_grad()
def graffiti_generate(messages, max_new_tokens=100):
    try:
        conversation_history = ""
        user_input = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role in ["system", "assistant", "user"]:
                conversation_history += f"<start_of_turn>{role}\n{content}\n<end_of_turn>\n"
            if role == "user":
                user_input = content

        prompt = f"{conversation_history}<start_of_turn>model\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        outputs = base_model.model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[-1].squeeze(0)
        sentence_embedding = torch.mean(hidden, dim=0)
        recalled = recall_memory(sentence_embedding, user_input)
        fused = fuse_embeddings(sentence_embedding, recalled)
        
        if recalled:
            current_time = time.time()
            avg_age = sum((current_time - mem[2]) / 3600 for mem in recalled) / len(recalled)
            recency_factor = max(1.0, 2.0 - avg_age)
        else:
            recency_factor = 1.0
        
        spray_vector = graffiti_adapter.spray(fused, recency_factor)
        store_memory(spray_vector, user_input)

        if recalled:
            context = "Context: " + " ".join([mem[1] for mem in recalled]) + "\n"
            prompt = prompt.replace("<start_of_turn>model\n", f"{context}<start_of_turn>model\n")
        
        final_inputs = tokenizer(prompt, return_tensors="pt").to(device)

        logger.info(f"Neural Graffiti injected. Recalled memories: {len(recalled)}, Spray state: {graffiti_adapter.spray.state.mean().item():.4f}")

        generated_ids = base_model.generate(
            input_ids=final_inputs["input_ids"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
            attention_mask=final_inputs.get("attention_mask", None)
        )

        response = tokenizer.decode(generated_ids[0][final_inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
        logger.info(f"Generated response: '{response}'")
        if not response:
            response = "Hello from Graffiti-Gemma!"
        return response
    except Exception as e:
        logger.error(f"Error in graffiti_generate: {e}")
        return "Error generating response"

# FastAPI Setup
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

class ChatRequest(BaseModel):
    model: str
    messages: list
    max_tokens: int = 100
    temperature: float = 1.0

async def chat_completions_handler(request: ChatRequest):
    logger.info(f"Received messages: {request.messages}")
    load_memory_bank()
    try:
        response_text = graffiti_generate(request.messages, max_new_tokens=request.max_tokens)
        save_memory_bank()

        async def stream_response():
            chunk = {
                "id": "chatcmpl-123",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": response_text},
                        "finish_reason": None
                    }
                ]
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            yield f"data: {json.dumps({'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
            yield "data: [DONE]\n\n"

        logger.info(f"Streaming response: '{response_text}'")
        return StreamingResponse(stream_response(), media_type="text/event-stream")
    except Exception as e:
        logger.error(f"Error in chat_completions: {e}")
        async def error_stream():
            yield f"data: {json.dumps({'choices': [{'index': 0, 'delta': {'content': 'Internal Server Error: Please check server logs.'}, 'finish_reason': 'error'}]})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(error_stream(), media_type="text/event-stream")

@app.post("/v1/chat/completions")
async def chat_completions_v1(request: ChatRequest):
    return await chat_completions_handler(request)

@app.post("/chat/completions")
async def chat_completions(request: ChatRequest):
    return await chat_completions_handler(request)

@app.get("/v1/models")
async def list_models():
    response = {
        "object": "list",
        "data": [
            {
                "id": "graffiti-gemma",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "xai"
            }
        ]
    }
    return response

@app.options("/v1/models")
async def options_models():
    return {"message": "OK"}

if __name__ == "__main__":
    load_memory_bank()
    print("Starting OpenAI-compatible API server")
    uvicorn.run(app, host="0.0.0.0", port=5000)
