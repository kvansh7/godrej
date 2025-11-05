import os
import json
import hashlib
import traceback
import logging
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import numpy as np
from typing import TypedDict, List, Dict, Any, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv
from bson import json_util
import tempfile
import io
from openai import OpenAI
import re

from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredPowerPointLoader,
    Docx2txtLoader,
)

# Initialize Flask app
app = Flask(__name__)
# CORS configuration
CORS(app, resources={
    r"/api/*": {
        "origins": "*",
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration
app.config['MAX_CONTENT_LENGTH'] = int(os.getenv('MAX_FILE_SIZE', 16 * 1024 * 1024))  # 16MB default
ALLOWED_EXTENSIONS = {'pdf', 'pptx', 'ppt', 'docx'}
TOP_K_LIMIT = int(os.getenv('TOP_K_LIMIT', 100))
BATCH_SIZE_LIMIT = int(os.getenv('BATCH_SIZE_LIMIT', 20))
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small')
EMBEDDING_DIMENSIONS = int(os.getenv('EMBEDDING_DIMENSIONS', 1536))  # Default for text-embedding-3-small
GEMINI_MODEL = os.getenv('LLM_MODEL', 'gemini-2.0-flash')
OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')

# MongoDB setup with better connection handling
try:
    uri = os.getenv("MONGODB_URI")
    if not uri:
        raise ValueError("MONGODB_URI not found in environment variables")
    
    client = MongoClient(
        uri, 
        server_api=ServerApi('1'),
        serverSelectionTimeoutMS=5000,
        connectTimeoutMS=10000
    )
    # Test connection
    client.admin.command('ping')
    logger.info("MongoDB connection successful")
    
    db = client['vendor_matching_db']
    vendors_collection = db['vendors']
    ps_collection = db['problem_statements']
    vendor_capabilities_collection = db['vendor_capabilities']
    ps_analysis_collection = db['ps_analysis']
    vendor_embeddings_collection = db['vendor_embeddings']
    ps_embeddings_collection = db['ps_embeddings']
except Exception as e:
    logger.error(f"MongoDB connection failed: {str(e)}")
    raise

# Initialize OpenAI client for embeddings
openai_client = None
try:
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        openai_client = OpenAI(api_key=openai_api_key)
        logger.info(f"Initialized OpenAI embeddings model: {EMBEDDING_MODEL} with {EMBEDDING_DIMENSIONS} dimensions")
    else:
        raise ValueError("OPENAI_API_KEY not found - Required for embeddings")
except Exception as e:
    logger.error(f"Failed to initialize OpenAI client: {str(e)}")
    raise

# Initialize LLMs
llm_gemini = None
llm_openai = None

try:
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if google_api_key:
        os.environ["GOOGLE_API_KEY"] = google_api_key
        llm_gemini = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0)
        logger.info(f"Initialized Gemini LLM: {GEMINI_MODEL}")
    else:
        logger.warning("GOOGLE_API_KEY not found - Gemini will not be available")
except Exception as e:
    logger.error(f"Failed to initialize Gemini LLM: {str(e)}")

try:
    if openai_api_key:
        llm_openai = ChatOpenAI(model=OPENAI_MODEL, temperature=0, api_key=openai_api_key)
        logger.info(f"Initialized OpenAI LLM: {OPENAI_MODEL}")
    else:
        logger.warning("OPENAI_API_KEY not found - OpenAI will not be available")
except Exception as e:
    logger.error(f"Failed to initialize OpenAI LLM: {str(e)}")

# Verify at least one LLM is available
if llm_gemini is None and llm_openai is None:
    raise ValueError("At least one LLM (Gemini or OpenAI) must be configured")


def get_llm(provider: str = 'gemini'):
    """Get the appropriate LLM based on provider"""
    if provider == 'openai':
        if llm_openai is None:
            raise ValueError("OpenAI LLM not configured. Please check OPENAI_API_KEY")
        return llm_openai
    elif provider == 'gemini':
        if llm_gemini is None:
            raise ValueError("Gemini LLM not configured. Please check GOOGLE_API_KEY")
        return llm_gemini
    else:
        raise ValueError(f"Unknown AI provider: {provider}")


def generate_embedding(text: str, dimensions: int = None) -> np.ndarray:
    """
    Generate embeddings using OpenAI's text-embedding-3-small model
    
    Args:
        text: Text to embed
        dimensions: Optional dimension reduction (default uses EMBEDDING_DIMENSIONS from config)
    
    Returns:
        numpy array of embeddings
    """
    try:
        if not openai_client:
            raise ValueError("OpenAI client not initialized")
        
        # Use configured dimensions if not specified
        if dimensions is None:
            dimensions = EMBEDDING_DIMENSIONS
        
        # Create embedding with optional dimension parameter
        params = {
            "input": text,
            "model": EMBEDDING_MODEL
        }
        
        # Only add dimensions parameter if it's different from default
        # text-embedding-3-small default is 1536, text-embedding-3-large default is 3072
        if dimensions != 1536 or EMBEDDING_MODEL == "text-embedding-3-large":
            params["dimensions"] = dimensions
        
        response = openai_client.embeddings.create(**params)
        
        embedding = np.array(response.data[0].embedding)
        logger.debug(f"Generated embedding with {len(embedding)} dimensions")
        
        return embedding
        
    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}")
        raise


def generate_embeddings_batch(texts: List[str], dimensions: int = None) -> List[np.ndarray]:
    """
    Generate embeddings for multiple texts in a single API call
    
    Args:
        texts: List of texts to embed (max 2048 items)
        dimensions: Optional dimension reduction
    
    Returns:
        List of numpy arrays of embeddings
    """
    try:
        if not openai_client:
            raise ValueError("OpenAI client not initialized")
        
        if len(texts) > 2048:
            raise ValueError("Maximum 2048 texts can be embedded in one batch")
        
        # Use configured dimensions if not specified
        if dimensions is None:
            dimensions = EMBEDDING_DIMENSIONS
        
        # Create embeddings
        params = {
            "input": texts,
            "model": EMBEDDING_MODEL
        }
        
        if dimensions != 1536 or EMBEDDING_MODEL == "text-embedding-3-large":
            params["dimensions"] = dimensions
        
        response = openai_client.embeddings.create(**params)
        
        embeddings = [np.array(item.embedding) for item in response.data]
        logger.info(f"Generated {len(embeddings)} embeddings with {len(embeddings[0])} dimensions each")
        
        return embeddings
        
    except Exception as e:
        logger.error(f"Error generating batch embeddings: {str(e)}")
        raise


# Graph State
class GraphState(TypedDict):
    problem_statement: str
    vendors: List[Dict[str, str]]
    ps_analysis: Dict[str, Any]
    ps_embedding: np.ndarray
    vendor_capabilities: List[Dict[str, Any]]
    vendor_embeddings: List[np.ndarray]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def serialize_mongo_doc(doc):
    """Convert MongoDB document to JSON serializable format"""
    return json.loads(json_util.dumps(doc)) if doc else None


def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def validate_file(file) -> bool:
    """Validate uploaded file"""
    if not file or not file.filename:
        raise ValueError("No file provided")
    if not allowed_file(file.filename):
        raise ValueError(f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}")
    
    # Check file size
    file.seek(0, 2)
    size = file.tell()
    file.seek(0)
    
    if size == 0:
        raise ValueError("File is empty")
    if size > app.config['MAX_CONTENT_LENGTH']:
        raise ValueError(f"File too large. Maximum size: {app.config['MAX_CONTENT_LENGTH'] / (1024*1024):.1f}MB")
    
    return True


def load_document(file_bytes: bytes, file_extension: str) -> str:
    """Load document from bytes and extract text"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        if file_extension.lower() == "pdf":
            loader = PyPDFLoader(tmp_path)
        elif file_extension.lower() in ["pptx", "ppt"]:
            loader = UnstructuredPowerPointLoader(tmp_path)
        elif file_extension.lower() == "docx":
            loader = Docx2txtLoader(tmp_path)
        else:
            raise ValueError("Unsupported file type")
        
        docs = loader.load()
        text = "\n".join([doc.page_content for doc in docs])
        
        if not text or len(text.strip()) < 50:
            raise ValueError("Document content is too short or empty")
        
        return text
    finally:
        try:
            os.remove(tmp_path)
        except Exception as e:
            logger.warning(f"Failed to delete temporary file {tmp_path}: {str(e)}")


def get_content_hash(content: str) -> str:
    """Generate SHA-256 hash of content"""
    return hashlib.sha256(content.encode()).hexdigest()


def load_cached_analysis(collection, content_hash: str) -> Dict[str, Any]:
    """Load cached analysis from MongoDB"""
    try:
        doc = collection.find_one({"content_hash": content_hash})
        return doc.get("data") if doc else None
    except Exception as e:
        logger.error(f"Error loading cached analysis: {str(e)}")
        return None


def save_analysis(collection, content_hash: str, data: Dict[str, Any]):
    """Save analysis to MongoDB cache"""
    try:
        collection.update_one(
            {"content_hash": content_hash},
            {"$set": {"content_hash": content_hash, "data": data}},
            upsert=True
        )
    except Exception as e:
        logger.error(f"Error saving analysis: {str(e)}")


def load_embedding(collection, content_hash: str) -> np.ndarray:
    """Load cached embedding from MongoDB"""
    try:
        doc = collection.find_one({"content_hash": content_hash})
        if doc and "embedding" in doc:
            return np.array(doc["embedding"])
        return None
    except Exception as e:
        logger.error(f"Error loading cached embedding: {str(e)}")
        return None


def save_embedding(collection, content_hash: str, embedding: np.ndarray):
    """Save embedding to MongoDB cache"""
    try:
        collection.update_one(
            {"content_hash": content_hash},
            {"$set": {"content_hash": content_hash, "embedding": embedding.tolist()}},
            upsert=True
        )
    except Exception as e:
        logger.error(f"Error saving embedding: {str(e)}")


def create_text_representation(data: Dict[str, Any]) -> str:
    """Create text representation from structured data for embedding"""
    text_parts = []
    for key, value in data.items():
        if key != "name":
            if isinstance(value, (dict, list)):
                text_parts.append(f"{key}: {json.dumps(value)}")
            else:
                text_parts.append(f"{key}: {value}")
    return " ".join(text_parts)


def validate_evaluation_params(params: List[Dict[str, Any]]) -> bool:
    """Validate evaluation parameters"""
    if not params or not isinstance(params, list):
        raise ValueError("evaluation_params must be a non-empty list")
    
    total_weight = 0
    for param in params:
        if 'name' not in param or 'weight' not in param:
            raise ValueError("Each parameter must have 'name' and 'weight' fields")
        
        name = param['name'].strip()
        if not name:
            raise ValueError("Parameter name cannot be empty")
        
        weight = param['weight']
        if not isinstance(weight, (int, float)) or weight < 0:
            raise ValueError(f"Parameter weight must be a non-negative number: {name}")
        
        total_weight += weight
    
    if abs(total_weight - 100) > 0.01:  # Allow small floating point errors
        raise ValueError(f"Total weight must equal 100%, got {total_weight}%")
    
    return True


def search_vendors_with_openai(problem_statement: str, ps_analysis: Dict[str, Any], count: int = 5) -> Dict[str, Any]:
    """
    Use OpenAI Responses API with web search to find relevant vendors based on problem statement
    """
    try:
        if not llm_openai:
            raise ValueError("OpenAI is required for web search")
        
        if not openai_client:
            raise ValueError("OpenAI client not initialized")
        
        # Extract information from problem statement analysis
        domains = ps_analysis.get('primary_technical_domains', [])
        tools = ps_analysis.get('required_tools_or_frameworks', [])
        requirements = ps_analysis.get('key_technical_requirements', [])
        deployment = ps_analysis.get('deployment_constraints', [])
        complexity = ps_analysis.get('project_complexity', '')
        
        # Log what we extracted
        logger.info(f"PS Analysis - Domains: {domains}")
        logger.info(f"PS Analysis - Tools: {tools}")
        logger.info(f"PS Analysis - Requirements: {requirements}")
        
        # Build search queries - create multiple specific queries
        search_queries = []
        
        # If domains exist, search for companies in those domains
        if domains and len(domains) > 0:
            if isinstance(domains, list):
                for domain in domains[:2]:  # Top 2 domains
                    search_queries.append(f"top companies specializing in {domain}")
            else:
                search_queries.append(f"top companies specializing in {domains}")
        
        # If tools exist, search for companies using those tools
        if tools and len(tools) > 0:
            if isinstance(tools, list):
                tools_str = ', '.join(tools[:3])  # Top 3 tools
                search_queries.append(f"companies using {tools_str}")
            else:
                search_queries.append(f"companies using {tools}")
        
        # Fallback: use the raw problem statement
        if not search_queries:
            logger.warning("No specific domains or tools found, using raw problem statement")
            # Extract key phrases from problem statement
            ps_lines = problem_statement.split('\n')
            description = ""
            for line in ps_lines:
                if line.startswith('Description:'):
                    description = line.replace('Description:', '').strip()
                    break
            
            if description:
                search_queries.append(f"technology companies for {description[:100]}")
            else:
                search_queries.append("technology consulting companies software development")
        
        logger.info(f"Generated search queries: {search_queries}")
        
        # Create a comprehensive search prompt
        search_prompt = f"""You are helping find technology vendors/companies. Use web search to find {count} real, currently active companies.

SEARCH CRITERIA:
- Technical domains: {', '.join(domains) if isinstance(domains, list) and domains else 'general software/technology'}
- Technologies/Tools: {', '.join(tools) if isinstance(tools, list) and tools else 'modern tech stack'}
- Requirements: {', '.join(requirements) if isinstance(requirements, list) and requirements else 'enterprise solutions'}

SEARCH QUERIES TO USE:
{chr(10).join([f"- {q}" for q in search_queries])}

INSTRUCTIONS:
1. Use web search RIGHT NOW to find {count} real companies
2. Search for companies that match the criteria above
3. Find companies that are currently active and have websites
4. For EACH company provide:
   - Exact company name
   - What they do (2-3 sentences)
   - Technologies they use
   - Their website URL

DO NOT respond without searching. Search the web now and provide {count} companies."""
        
        logger.info(f"Searching web for vendors using OpenAI Responses API...")
        logger.info(f"Search prompt length: {len(search_prompt)}")
        
        # Use Responses API with explicit tool configuration
        try:
            response = openai_client.responses.create(
                model="gpt-4o",  # gpt-4o supports web search
                input=search_prompt,
                tools=[{
                    "type": "web_search_preview"
                }],
                temperature=0,
                tool_choice="auto"  # Let model decide when to use tools
            )
        except Exception as api_error:
            logger.error(f"API call failed: {str(api_error)}")
            raise
        
        # Extract the text content from response output
        search_results = ""
        citations = []
        web_search_performed = False
        
        # Debug: Log the raw response structure
        logger.info(f"Response type: {type(response)}")
        logger.info(f"Response output length: {len(response.output) if hasattr(response, 'output') else 'No output'}")
        
        # Parse the response output - look for web_search_call items
        for idx, item in enumerate(response.output):
            logger.info(f"Item {idx} type: {item.type}")
            
            # Check if web search was actually called
            if item.type == "web_search_call":
                web_search_performed = True
                logger.info(f"Web search was performed! Status: {getattr(item, 'status', 'unknown')}")
            
            if item.type == "message":
                if hasattr(item, 'content'):
                    logger.info(f"Item {idx} has content with {len(item.content)} items")
                    for content_idx, content_item in enumerate(item.content):
                        logger.info(f"Content item {content_idx} type: {content_item.type}")
                        
                        if content_item.type == "output_text":
                            text_content = content_item.text
                            search_results += text_content + "\n"
                            logger.info(f"Extracted text length: {len(text_content)}")
                            
                            # Extract citations/annotations if available
                            if hasattr(content_item, 'annotations') and content_item.annotations:
                                logger.info(f"Found {len(content_item.annotations)} annotations")
                                for annotation in content_item.annotations:
                                    if hasattr(annotation, 'type') and annotation.type == "url_citation":
                                        citations.append({
                                            "title": getattr(annotation, 'title', 'Unknown'),
                                            "url": getattr(annotation, 'url', '')
                                        })
                                        logger.info(f"Added citation: {annotation.title}")
        
        if not web_search_performed:
            logger.warning("Web search tool was NOT used by the model!")
            logger.warning("Model response: " + search_results[:500])
            # Return error indicating web search wasn't used
            return {
                "vendors": [],
                "search_results_raw": search_results,
                "sources_count": 0,
                "search_successful": False,
                "error": "Web search was not performed by the model. The model may need a more specific prompt or the domains/tools might be too vague."
            }
        
        logger.info(f"Total search results length: {len(search_results)}")
        logger.info(f"Search results preview: {search_results[:500]}")
        
        # Parse the response to extract vendor information
        vendors_found = parse_vendor_search_results(search_results)
        logger.info(f"Parsed {len(vendors_found)} vendors from search results")
        
        # Add citations to vendors where applicable
        for vendor in vendors_found:
            if not vendor.get("web_sources"):
                vendor["web_sources"] = []
            # Add citations that aren't already in web_sources
            for citation in citations:
                if citation not in vendor["web_sources"]:
                    vendor["web_sources"].append(citation)
        
        logger.info(f"Found {len(vendors_found)} vendors from web search with {len(citations)} sources")
        
        return {
            "vendors": vendors_found,
            "search_results_raw": search_results,
            "sources_count": len(citations),
            "search_successful": True
        }
        
    except Exception as e:
        logger.error(f"Web search error: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "vendors": [],
            "search_results_raw": "",
            "sources_count": 0,
            "search_successful": False,
            "error": str(e)
        }


def parse_vendor_search_results(search_text: str) -> List[Dict[str, Any]]:
    """
    Parse the web search results to extract structured vendor information
    """
    if not search_text or len(search_text.strip()) < 20:
        logger.warning("Search text is empty or too short")
        return []
    
    vendors = []
    
    logger.info(f"Parsing search text of length {len(search_text)}")
    
    # Try multiple splitting patterns to handle different formats
    # Pattern 1: Numbered list with ** for company names
    vendor_sections = re.split(r'\n(?=\d+\.\s+\*\*)', search_text, flags=re.MULTILINE)
    
    # If that didn't work, try other patterns
    if len(vendor_sections) <= 1:
        # Pattern 2: Lines starting with numbers
        vendor_sections = re.split(r'\n(?=\d+\.)', search_text, flags=re.MULTILINE)
    
    # If still didn't work, try splitting by double newlines
    if len(vendor_sections) <= 1:
        vendor_sections = re.split(r'\n\n+', search_text)
    
    logger.info(f"Split into {len(vendor_sections)} sections")
    
    for idx, section in enumerate(vendor_sections):
        section = section.strip()
        
        if len(section) < 30:  # Skip very short sections
            logger.debug(f"Skipping short section {idx}: {section[:50]}")
            continue
        
        logger.info(f"Processing section {idx} (length: {len(section)})")
        logger.debug(f"Section preview: {section[:200]}")
            
        vendor_data = {
            "name": "",
            "description": "",
            "capabilities": {},
            "web_sources": []
        }
        
        # Extract name - try multiple patterns
        name_match = None
        
        # Pattern 1: **Company Name**
        name_match = re.search(r'\*\*([^*]+?)\*\*', section)
        
        # Pattern 2: Number. Company Name
        if not name_match:
            name_match = re.search(r'^\d+\.\s+([A-Z][A-Za-z\s&\.,-]+?)(?:\n|:|\()', section, re.MULTILINE)
        
        # Pattern 3: Just company name at start
        if not name_match:
            name_match = re.search(r'^([A-Z][A-Za-z\s&\.,-]{3,50}?)(?:\n|:)', section, re.MULTILINE)
        
        if name_match:
            vendor_data["name"] = name_match.group(1).strip()
            # Remove common prefixes like numbers
            vendor_data["name"] = re.sub(r'^\d+\.\s+', '', vendor_data["name"])
            logger.info(f"Found vendor name: {vendor_data['name']}")
        else:
            logger.warning(f"Could not extract name from section {idx}")
            continue
        
        # Extract description - get first few meaningful lines
        lines = section.split('\n')
        description_lines = []
        
        for line in lines:
            line = line.strip()
            # Skip empty lines, URLs, and lines that are just markers
            if (line and 
                not line.startswith(('http', 'www', '*', '#')) and
                not re.match(r'^\d+\.', line) and
                len(line) > 20):
                # Remove markdown bold markers
                line = re.sub(r'\*\*([^*]+)\*\*', r'\1', line)
                description_lines.append(line)
            
            if len(description_lines) >= 3:
                break
        
        vendor_data["description"] = ' '.join(description_lines[:3])
        logger.info(f"Description length: {len(vendor_data['description'])}")
        
        # Extract URLs
        url_matches = re.findall(r'https?://[^\s<>"{}|\\^`\[\]]+', section)
        logger.info(f"Found {len(url_matches)} URLs")
        
        for url in url_matches[:2]:  # Limit to 2 URLs
            vendor_data["web_sources"].append({
                "url": url,
                "title": vendor_data["name"] or "Company Website"
            })
        
        # Store full text for LLM evaluation
        vendor_data["full_text"] = section.strip()
        
        # Only add if we found a name and some description
        if vendor_data["name"] and len(vendor_data["description"]) > 20:
            vendors.append(vendor_data)
            logger.info(f"Added vendor: {vendor_data['name']}")
        else:
            logger.warning(f"Skipping vendor - insufficient data. Name: {vendor_data['name']}, Desc length: {len(vendor_data['description'])}")
    
    logger.info(f"Total vendors parsed: {len(vendors)}")
    return vendors[:10]  # Limit to 10 vendors


def evaluate_web_vendors(
    ps_analysis: Dict[str, Any],
    web_vendors: List[Dict[str, Any]],
    evaluation_params: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Evaluate web-discovered vendors using LLM
    """
    results = []
    
    # Build evaluation criteria
    param_keys = [normalize_param_name(param['name']) for param in evaluation_params]
    criteria_list = [f"{i+1}. {param['name']} (Weight: {param['weight']}%)" 
                     for i, param in enumerate(evaluation_params)]
    criteria_text = "\n".join(criteria_list)
    
    # Build score format
    score_format_lines = [f'    "{key}": 0-100,' for key in param_keys]
    score_format = "\n".join(score_format_lines)
    
    prompt = PromptTemplate.from_template("""
You are evaluating vendors found through web search against problem statement requirements.

Problem Requirements:
{ps_analysis}

Vendor Information from Web Search:
{vendor_info}

Evaluation Criteria (score each from 0 to 100):
{criteria_text}

For this vendor, analyze the information provided and give:
1. A score (0-100) for EACH criterion listed above
2. A justification explaining the scores based on web information
3. Key strengths (2-4 bullet points)
4. Potential concerns or gaps (2-4 bullet points)

Return a JSON object with this EXACT structure:
{{
  "name": "Vendor Name",
{score_format}
  "justification": "Detailed explanation",
  "strengths": ["Strength 1", "Strength 2"],
  "concerns": ["Concern 1", "Concern 2"]
}}

Return ONLY the JSON object, no other text.
""")
    
    llm = get_llm('openai')  # Use OpenAI for consistency
    parser = JsonOutputParser()
    chain = prompt | llm | parser
    
    for vendor in web_vendors:
        try:
            logger.info(f"Evaluating web vendor: {vendor['name']}")
            
            vendor_info = f"""
Name: {vendor['name']}
Description: {vendor['description']}
Full Information: {vendor.get('full_text', vendor['description'])}
"""
            
            result = chain.invoke({
                "ps_analysis": json.dumps(ps_analysis, indent=2),
                "vendor_info": vendor_info,
                "criteria_text": criteria_text,
                "score_format": score_format
            })
            
            # Extract scores
            scores = {}
            for param_key in param_keys:
                score_key = f"{param_key}_score"
                score = result.get(param_key, 0)
                scores[score_key] = float(score)
            
            # Calculate composite score
            composite_score = calculate_composite_score(scores, evaluation_params)
            
            # Build result object
            result_obj = {
                "name": vendor["name"],
                "description": vendor.get("description", ""),
                "composite_score": composite_score,
                "justification": result.get("justification", ""),
                "strengths": result.get("strengths", []),
                "concerns": result.get("concerns", []),
                "web_sources": vendor.get("web_sources", []),
                "source": "web_search"
            }
            result_obj.update(scores)
            
            results.append(result_obj)
            logger.info(f"Evaluated {vendor['name']}: composite score {composite_score}")
            
        except Exception as e:
            logger.error(f"Error evaluating web vendor {vendor.get('name', 'Unknown')}: {str(e)}")
            continue
    
    # Sort by composite score
    results.sort(key=lambda x: x["composite_score"], reverse=True)
    return results


# ============================================================================
# CORE PROCESSING FUNCTIONS
# ============================================================================

def process_vendor_profile(vendor_name: str, vendor_text: str, ai_provider: str = 'gemini') -> Tuple[Dict[str, Any], np.ndarray]:
    """Extract vendor capabilities and generate embedding using OpenAI"""
    vendor_hash = get_content_hash(f"{vendor_name}:{vendor_text}")

    # Check MongoDB for capabilities
    capabilities = load_cached_analysis(vendor_capabilities_collection, vendor_hash)
    if not capabilities:
        logger.info(f"Analyzing vendor capabilities for: {vendor_name}")
        prompt = PromptTemplate.from_template("""
        From this vendor profile, extract:
        1. Key technical domains (e.g., NLP, CV, ML)
        2. Tools and frameworks used
        3. Core capabilities (e.g., scalability, real-time processing)
        4. Industry experience
        5. Team size and project scale

        Vendor Profile: {vendor_text}

        Provide structured output in JSON format.
        """)
        llm = get_llm(ai_provider)
        chain = prompt | llm | JsonOutputParser()
        capabilities = chain.invoke({"vendor_text": vendor_text})
        capabilities["name"] = vendor_name
        save_analysis(vendor_capabilities_collection, vendor_hash, capabilities)
    else:
        logger.info(f"Using cached capabilities for: {vendor_name}")

    # Check MongoDB for embedding
    embedding = load_embedding(vendor_embeddings_collection, vendor_hash)
    if embedding is None:
        logger.info(f"Generating OpenAI embedding for: {vendor_name}")
        text_representation = create_text_representation(capabilities)
        embedding = generate_embedding(text_representation)
        save_embedding(vendor_embeddings_collection, vendor_hash, embedding)
    else:
        logger.info(f"Using cached embedding for: {vendor_name}")

    return capabilities, embedding


def process_problem_statement(problem_statement: str, ai_provider: str = 'gemini') -> Tuple[Dict[str, Any], np.ndarray]:
    """Analyze problem statement and generate embedding using OpenAI"""
    ps_hash = get_content_hash(problem_statement)

    # Check MongoDB for analysis
    analysis = load_cached_analysis(ps_analysis_collection, ps_hash)
    if not analysis:
        logger.info("Analyzing problem statement")
        prompt = PromptTemplate.from_template("""
        Analyze this problem statement and extract:
        1. Primary technical domains (e.g., NLP, CV, ML)
        2. Required tools or frameworks
        3. Key technical requirements (e.g., real-time, accuracy)
        4. Deployment constraints (e.g., cloud, edge)
        5. Project complexity (e.g., research, production)

        Problem Statement: {problem_statement}

        Provide structured analysis in JSON format.
        """)
        llm = get_llm(ai_provider)
        chain = prompt | llm | JsonOutputParser()
        analysis = chain.invoke({"problem_statement": problem_statement})
        save_analysis(ps_analysis_collection, ps_hash, analysis)
    else:
        logger.info("Using cached problem statement analysis")

    # Check MongoDB for embedding
    embedding = load_embedding(ps_embeddings_collection, ps_hash)
    if embedding is None:
        logger.info("Generating OpenAI embedding for problem statement")
        text_representation = create_text_representation(analysis)
        embedding = generate_embedding(text_representation)
        save_embedding(ps_embeddings_collection, ps_hash, embedding)
    else:
        logger.info("Using cached problem statement embedding")

    return analysis, embedding


def shortlist_vendors(
    ps_embedding: np.ndarray, 
    vendor_embeddings: List[np.ndarray], 
    vendor_capabilities: List[Dict[str, Any]], 
    top_k: int = 20
) -> List[Dict[str, Any]]:
    """Calculate semantic similarity and shortlist top vendors"""
    similarities = cosine_similarity([ps_embedding], np.array(vendor_embeddings))[0]
    similarity_results = [
        {
            "name": cap["name"],
            "semantic_similarity_score": float(similarities[i]),
            "similarity_percentage": float(similarities[i]) * 100,
            "vendor_capabilities": cap
        }
        for i, cap in enumerate(vendor_capabilities)
    ]
    similarity_results.sort(key=lambda x: x["semantic_similarity_score"], reverse=True)
    return similarity_results[:top_k]


def normalize_param_name(name: str) -> str:
    """Normalize parameter name to a valid Python identifier"""
    import re
    # Convert to lowercase and replace special characters with underscore
    normalized = name.lower()
    normalized = re.sub(r'[^a-z0-9]+', '_', normalized)
    # Remove leading/trailing underscores
    normalized = normalized.strip('_')
    return normalized


def calculate_composite_score(scores: Dict[str, float], evaluation_params: List[Dict[str, Any]]) -> float:
    """Calculate weighted composite score from individual criteria scores using dynamic parameters"""
    composite = 0.0
    for param in evaluation_params:
        param_key = normalize_param_name(param['name'])
        score_key = f"{param_key}_score"
        score = scores.get(score_key, 0)
        weight = param['weight'] / 100.0  # Convert percentage to decimal
        composite += score * weight
    
    return round(composite, 2)


def evaluate_shortlist(
    ps_analysis: Dict[str, Any], 
    shortlist: List[Dict[str, Any]], 
    evaluation_params: List[Dict[str, Any]],
    batch_size: int = 5,
    ai_provider: str = 'gemini'
) -> List[Dict[str, Any]]:
    """Evaluate shortlisted vendors using multi-criteria LLM evaluation in batches with dynamic parameters"""
    
    # Build criteria description from evaluation_params
    criteria_list = []
    param_keys = []
    for i, param in enumerate(evaluation_params, 1):
        param_key = normalize_param_name(param['name'])
        criteria_list.append(f"{i}. {param['name']} (Weight: {param['weight']}%)")
        param_keys.append(param_key)
    
    criteria_text = "\n".join(criteria_list)
    
    prompt = PromptTemplate.from_template("""
You are evaluating vendors against problem statement requirements.

Problem Requirements:
{ps_analysis}

Vendors to evaluate:
{vendor_batch}

Evaluation Criteria (score each from 0 to 100):
{criteria_text}

For each vendor, analyze their capabilities and provide:
1. A score (0-100) for EACH criterion listed above
2. A justification explaining the scores
3. Key strengths (2-4 bullet points)
4. Potential concerns (2-4 bullet points)

You must return a valid JSON array with this EXACT structure:
[
  {{
    "name": "Vendor Name",
{score_format}
    "justification": "Detailed explanation of scores",
    "strengths": ["Strength 1", "Strength 2"],
    "concerns": ["Concern 1", "Concern 2"]
  }}
]

Return ONLY the JSON array, no other text.
""")
    
    # Build score format dynamically
    score_format_lines = []
    for param_key in param_keys:
        score_format_lines.append(f'    "{param_key}": 0-100,')
    score_format = "\n".join(score_format_lines)
    
    llm = get_llm(ai_provider)
    parser = JsonOutputParser()
    chain = prompt | llm | parser

    results = []
    for i in range(0, len(shortlist), batch_size):
        batch = shortlist[i:i + batch_size]
        batch_capabilities = [v["vendor_capabilities"] for v in batch]
        
        try:
            logger.info(f"Evaluating batch {i//batch_size + 1}/{(len(shortlist)-1)//batch_size + 1} with {ai_provider}")
            
            # Make the LLM call
            batch_results = chain.invoke({
                "ps_analysis": json.dumps(ps_analysis, indent=2),
                "vendor_batch": json.dumps(batch_capabilities, indent=2),
                "criteria_text": criteria_text,
                "score_format": score_format
            })
            
            logger.info(f"Received {len(batch_results) if isinstance(batch_results, list) else 1} results from LLM")
            
            # Ensure batch_results is a list
            if not isinstance(batch_results, list):
                batch_results = [batch_results]
            
            for j, result in enumerate(batch_results):
                if j >= len(batch):
                    break
                
                logger.info(f"Processing result for vendor: {result.get('name', 'Unknown')}")
                    
                # Extract individual scores based on dynamic parameters
                scores = {}
                missing_scores = []
                for param_key in param_keys:
                    score_key = f"{param_key}_score"
                    # Get score from result
                    score = result.get(param_key, 0)
                    if score == 0:
                        missing_scores.append(param_key)
                    scores[score_key] = float(score)
                
                if missing_scores:
                    logger.warning(f"Missing scores for {missing_scores} in vendor {result.get('name')}")
                
                # Calculate composite score with dynamic weights
                composite_score = calculate_composite_score(scores, evaluation_params)
                
                # Build result object with dynamic scores
                result_obj = {
                    **batch[j],
                    "composite_score": composite_score,
                    "justification": result.get("justification", "No justification provided"),
                    "strengths": result.get("strengths", []),
                    "concerns": result.get("concerns", [])
                }
                
                # Add all individual scores
                result_obj.update(scores)
                
                results.append(result_obj)
                logger.info(f"Successfully processed vendor {result.get('name')} with composite score {composite_score}")
                
        except Exception as e:
            logger.error(f"LLM evaluation error for batch with {ai_provider}: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            
            # Fallback to semantic similarity with default scores
            logger.warning(f"Falling back to semantic similarity for batch {i//batch_size + 1}")
            for vendor in batch:
                semantic_score = vendor.get("semantic_similarity_score", 0) * 100
                
                # Create fallback scores for all parameters
                scores = {}
                for param_key in param_keys:
                    score_key = f"{param_key}_score"
                    # Use semantic score with slight variations
                    scores[score_key] = semantic_score * (0.7 + (hash(param_key) % 30) / 100)
                
                composite_score = calculate_composite_score(scores, evaluation_params)
                
                result_obj = {
                    **vendor,
                    "composite_score": composite_score,
                    "justification": f"Fallback to semantic similarity due to evaluation error: {str(e)}",
                    "strengths": ["Automated evaluation based on semantic similarity"],
                    "concerns": ["Manual review recommended - automated evaluation used"]
                }
                result_obj.update(scores)
                results.append(result_obj)
    
    # Sort by composite score
    results.sort(key=lambda x: x["composite_score"], reverse=True)
    return results


def validate_matching_params(top_k: int, batch_size: int):
    """Validate matching parameters"""
    if not isinstance(top_k, int) or top_k < 1 or top_k > TOP_K_LIMIT:
        raise ValueError(f"top_k must be between 1 and {TOP_K_LIMIT}")
    if not isinstance(batch_size, int) or batch_size < 1 or batch_size > BATCH_SIZE_LIMIT:
        raise ValueError(f"batch_size must be between 1 and {BATCH_SIZE_LIMIT}")


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.before_request
def log_request():
    """Log all incoming requests"""
    logger.info(f"{request.method} {request.path} from {request.remote_addr}")


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        client.admin.command('ping')
        return jsonify({
            "status": "healthy", 
            "database": "connected",
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dimensions": EMBEDDING_DIMENSIONS,
            "gemini_available": llm_gemini is not None,
            "openai_available": llm_openai is not None
        }), 200
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({"status": "unhealthy", "database": "disconnected"}), 503


@app.route('/api/dashboard', methods=['GET'])
def get_dashboard():
    """Get dashboard statistics"""
    try:
        vendors = list(vendors_collection.find())
        ps_list = list(ps_collection.find())
        cached_analyses = (
            vendor_capabilities_collection.count_documents({}) + 
            ps_analysis_collection.count_documents({})
        )
        
        recent_vendors = [v['name'] for v in vendors[-3:]] if vendors else []
        recent_ps = [ps['title'] for ps in ps_list[-3:]] if ps_list else []
        
        data = {
            "total_vendors": len(vendors),
            "total_ps": len(ps_list),
            "cached_analyses": cached_analyses,
            "recent_vendors": recent_vendors,
            "recent_ps": recent_ps
        }
        return jsonify(data), 200
    except Exception as e:
        logger.error(f"Dashboard error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": "Failed to fetch dashboard data"}), 500


@app.route('/api/vendor_submission', methods=['POST'])
def vendor_submission():
    """Submit and process vendor profile"""
    try:
        if 'file' not in request.files or not request.form.get('vendor_name'):
            return jsonify({"error": "Vendor name and file are required"}), 400
        
        vendor_name = request.form.get('vendor_name').strip()
        if not vendor_name or len(vendor_name) > 100:
            return jsonify({"error": "Invalid vendor name (max 100 characters)"}), 400
            
        file = request.files['file']
        validate_file(file)
        
        # Read file bytes directly (in memory)
        file_bytes = file.read()
        file_ext = secure_filename(file.filename).rsplit('.', 1)[1]

        # Process document in memory
        text = load_document(file_bytes, file_ext)
        vendor_data = {"name": vendor_name, "text": text}
        
        # Store in MongoDB
        vendors_collection.update_one(
            {"name": vendor_name},
            {"$set": vendor_data},
            upsert=True
        )
        
        # Process vendor profile and cache (using default gemini for initial processing)
        capabilities, embedding = process_vendor_profile(vendor_name, text, 'gemini' if llm_gemini else 'openai')
        
        logger.info(f"Vendor '{vendor_name}' onboarded successfully")
        return jsonify({"message": f"Vendor '{vendor_name}' onboarded and cached!"}), 200
    except ValueError as ve:
        logger.warning(f"Validation error: {str(ve)}")
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        logger.error(f"Vendor submission error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": "Failed to process vendor submission"}), 500


@app.route('/api/ps_submission', methods=['POST'])
def ps_submission():
    """Submit and process problem statement"""
    try:
        data = request.json
        title = data.get('title', '').strip()
        description = data.get('description', '').strip()
        outcomes = data.get('outcomes', '').strip()
        
        if not title or not description or not outcomes:
            return jsonify({"error": "Title, description, and outcomes are required"}), 400
        
        if len(title) > 200:
            return jsonify({"error": "Title too long (max 200 characters)"}), 400
        
        ps_id = hashlib.md5(title.encode()).hexdigest()[:8]
        problem_statement = f"Title: {title}\nDescription: {description}\nOutcomes: {outcomes}"
        ps_data = {
            "id": ps_id,
            "title": title,
            "description": description,
            "outcomes": outcomes,
            "full_statement": problem_statement
        }
        
        # Store in MongoDB
        ps_collection.update_one(
            {"id": ps_id},
            {"$set": ps_data},
            upsert=True
        )
        
        # Process and cache (using default gemini for initial processing)
        analysis, embedding = process_problem_statement(problem_statement, 'gemini' if llm_gemini else 'openai')
        
        logger.info(f"Problem statement '{title}' (ID: {ps_id}) processed successfully")
        return jsonify({"message": f"PS '{title}' (ID: {ps_id}) processed and cached!"}), 200
    except ValueError as ve:
        logger.warning(f"Validation error: {str(ve)}")
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        logger.error(f"PS submission error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": "Failed to process problem statement"}), 500


@app.route('/api/vendor_matching', methods=['POST'])
def vendor_matching():
    """Match vendors to problem statement with multi-criteria evaluation and dynamic parameters"""
    try:
        data = request.json
        ps_id = data.get('ps_id')
        top_k = int(data.get('top_k', 20))
        batch_size = int(data.get('batch_size', 5))
        ai_provider = data.get('ai_provider', 'gemini')
        evaluation_params = data.get('evaluation_params', [
            {'name': 'Domain Fit', 'weight': 40},
            {'name': 'Tools Fit', 'weight': 30},
            {'name': 'Experience', 'weight': 20},
            {'name': 'Scalability', 'weight': 10}
        ])
        
        # Validate inputs
        validate_matching_params(top_k, batch_size)
        validate_evaluation_params(evaluation_params)
        
        # Validate AI provider
        if ai_provider not in ['gemini', 'openai']:
            return jsonify({"error": "ai_provider must be 'gemini' or 'openai'"}), 400
        
        logger.info(f"Starting matching for PS ID: {ps_id}, top_k: {top_k}, batch_size: {batch_size}, AI: {ai_provider}")
        logger.info(f"Evaluation parameters: {evaluation_params}")
        
        # Fetch selected problem statement
        selected_ps = ps_collection.find_one({"id": ps_id})
        if not selected_ps:
            return jsonify({"error": "Problem statement not found"}), 404
        
        problem_statement = selected_ps["full_statement"]
        vendors = list(vendors_collection.find())
        if not vendors:
            return jsonify({"error": "No vendors available"}), 400
        
        # Get PS hash and load cached data (will generate if not exists)
        ps_hash = get_content_hash(problem_statement)
        ps_analysis, ps_embedding = process_problem_statement(problem_statement, ai_provider)
        
        # Process vendors - use cached data where available
        vendor_capabilities = []
        vendor_embeddings = []
        vendors_processed = 0
        vendors_from_cache = 0
        
        for vendor in vendors:
            vendor_hash = get_content_hash(f"{vendor['name']}:{vendor['text']}")
            
            # Try to load from cache first
            cap = load_cached_analysis(vendor_capabilities_collection, vendor_hash)
            emb = load_embedding(vendor_embeddings_collection, vendor_hash)
            
            # If not in cache, process and cache
            if cap is None or emb is None:
                logger.info(f"Processing vendor (not in cache): {vendor['name']}")
                cap, emb = process_vendor_profile(vendor["name"], vendor["text"], ai_provider)
                vendors_processed += 1
            else:
                logger.info(f"Using cached data for vendor: {vendor['name']}")
                vendors_from_cache += 1
            
            vendor_capabilities.append(cap)
            vendor_embeddings.append(emb)
        
        logger.info(f"Vendors from cache: {vendors_from_cache}, Newly processed: {vendors_processed}")
        
        # Shortlist vendors
        shortlist = shortlist_vendors(ps_embedding, vendor_embeddings, vendor_capabilities, top_k=top_k)
        logger.info(f"Shortlisted {len(shortlist)} vendors")
        
        # Evaluate shortlist with multi-criteria and dynamic parameters
        final_results = evaluate_shortlist(ps_analysis, shortlist, evaluation_params, batch_size=batch_size, ai_provider=ai_provider)
        logger.info(f"Evaluation complete: {len(final_results)} results")
        
        # Convert selected_ps to JSON-serializable format
        selected_ps_serializable = json.loads(json_util.dumps(selected_ps))
        
        # Prepare response
        response = {
            "problem_statement": selected_ps_serializable,
            "results": final_results,
            "total_vendors_analyzed": len(vendors),
            "shortlisted_vendors": len(final_results),
            "top_composite_score": final_results[0]["composite_score"] if final_results else 0,
            "evaluation_params": evaluation_params,
            "ai_provider": ai_provider,
            "cache_stats": {
                "vendors_from_cache": vendors_from_cache,
                "vendors_processed": vendors_processed
            }
        }
        
        return jsonify(response), 200
        
    except ValueError as ve:
        logger.warning(f"Validation error: {str(ve)}")
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        logger.error(f"Matching error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": "An internal error occurred during matching"}), 500


@app.route('/api/download_results/<ps_id>', methods=['GET'])
def download_results(ps_id):
    """Download matching results as JSON"""
    try:
        # Get evaluation params from query string if provided
        evaluation_params_str = request.args.get('evaluation_params')
        if evaluation_params_str:
            evaluation_params = json.loads(evaluation_params_str)
            validate_evaluation_params(evaluation_params)
        else:
            evaluation_params = [
                {'name': 'Domain Fit', 'weight': 40},
                {'name': 'Tools Fit', 'weight': 30},
                {'name': 'Experience', 'weight': 20},
                {'name': 'Scalability', 'weight': 10}
            ]
        
        ai_provider = request.args.get('ai_provider', 'gemini')
        
        selected_ps = ps_collection.find_one({"id": ps_id})
        if not selected_ps:
            return jsonify({"error": "Problem statement not found"}), 404
        
        problem_statement = selected_ps["full_statement"]
        vendors = list(vendors_collection.find())
        if not vendors:
            return jsonify({"error": "No vendors available"}), 400
        
        # Process and match
        ps_analysis, ps_embedding = process_problem_statement(problem_statement, ai_provider)
        vendor_capabilities = []
        vendor_embeddings = []
        for vendor in vendors:
            cap, emb = process_vendor_profile(vendor["name"], vendor["text"], ai_provider)
            vendor_capabilities.append(cap)
            vendor_embeddings.append(emb)
        
        shortlist = shortlist_vendors(ps_embedding, vendor_embeddings, vendor_capabilities)
        final_results = evaluate_shortlist(ps_analysis, shortlist, evaluation_params, ai_provider=ai_provider)
        
        # Convert to JSON-serializable format
        selected_ps_serializable = json.loads(json_util.dumps(selected_ps))
        results = {
            "problem_statement": selected_ps_serializable,
            "results": final_results,
            "evaluation_params": evaluation_params,
            "ai_provider": ai_provider,
            "generated_at": json_util.default({"$date": {"$numberLong": str(int(os.times().elapsed * 1000))}})
        }
        results_json = json.dumps(results, indent=2, default=json_util.default)
        
        logger.info(f"Downloaded results for PS: {ps_id}")
        return send_file(
            io.BytesIO(results_json.encode()),
            mimetype='application/json',
            as_attachment=True,
            download_name=f"matching_results_{ps_id}.json"
        )
    except Exception as e:
        logger.error(f"Download error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": "Failed to generate download"}), 500


@app.route('/api/problem_statements', methods=['GET'])
def get_problem_statements():
    """Get all problem statements"""
    try:
        ps_list = list(ps_collection.find())
        ps_options = [
            json.loads(json_util.dumps({"id": ps["id"], "title": ps["title"]})) 
            for ps in ps_list
        ]
        return jsonify(ps_options), 200
    except Exception as e:
        logger.error(f"Error fetching problem statements: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": "Failed to fetch problem statements"}), 500


@app.route('/api/problem_statements/<ps_id>/analysis', methods=['GET'])
def get_problem_statement_analysis(ps_id):
    """Get problem statement with its analysis for debugging"""
    try:
        selected_ps = ps_collection.find_one({"id": ps_id})
        if not selected_ps:
            return jsonify({"error": "Problem statement not found"}), 404
        
        problem_statement = selected_ps["full_statement"]
        
        # Get analysis
        ps_analysis, ps_embedding = process_problem_statement(problem_statement, 'openai')
        
        return jsonify({
            "problem_statement": json.loads(json_util.dumps(selected_ps)),
            "analysis": ps_analysis,
            "has_embedding": ps_embedding is not None,
            "embedding_shape": ps_embedding.shape if ps_embedding is not None else None
        }), 200
    except Exception as e:
        logger.error(f"Error fetching PS analysis: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": "Failed to fetch problem statement analysis"}), 500


@app.route('/api/clear_cache', methods=['POST'])
def clear_cache():
    """Clear all cached data"""
    try:
        vendor_capabilities_collection.delete_many({})
        ps_analysis_collection.delete_many({})
        vendor_embeddings_collection.delete_many({})
        ps_embeddings_collection.delete_many({})
        logger.info("Cache cleared successfully")
        return jsonify({"message": "Cache cleared successfully!"}), 200
    except Exception as e:
        logger.error(f"Cache clear error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": "Failed to clear cache"}), 500


@app.route('/api/web_search_vendors', methods=['POST', 'OPTIONS'])
def web_search_vendors():
    """
    Search web for vendors matching the problem statement
    """
    # Handle OPTIONS request for CORS
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        data = request.json
        ps_id = data.get('ps_id')
        count = int(data.get('count', 5))
        evaluation_params = data.get('evaluation_params', [
            {'name': 'Domain Fit', 'weight': 40},
            {'name': 'Tools Fit', 'weight': 30},
            {'name': 'Experience', 'weight': 20},
            {'name': 'Scalability', 'weight': 10}
        ])
        
        # Validate
        if count < 3 or count > 10:
            return jsonify({"error": "count must be between 3 and 10"}), 400
        
        validate_evaluation_params(evaluation_params)
        
        logger.info(f"Web search request for PS ID: {ps_id}, count: {count}")
        
        # Fetch problem statement
        selected_ps = ps_collection.find_one({"id": ps_id})
        if not selected_ps:
            return jsonify({"error": "Problem statement not found"}), 404
        
        problem_statement = selected_ps["full_statement"]
        
        # Get PS analysis - Force regeneration if cache is empty or invalid
        ps_hash = get_content_hash(problem_statement)
        ps_analysis = load_cached_analysis(ps_analysis_collection, ps_hash)
        
        # Check if analysis is valid (has required fields)
        if not ps_analysis or not any([
            ps_analysis.get('primary_technical_domains'),
            ps_analysis.get('required_tools_or_frameworks'),
            ps_analysis.get('key_technical_requirements')
        ]):
            logger.info("PS analysis is missing or invalid, regenerating...")
            # Force regenerate analysis
            ps_analysis, ps_embedding = process_problem_statement(problem_statement, 'gemini')
            logger.info(f"Regenerated PS analysis: {ps_analysis}")
        else:
            logger.info(f"Using cached PS analysis: {ps_analysis}")
        
        # Search web for vendors
        search_results = search_vendors_with_openai(problem_statement, ps_analysis, count)
        
        if not search_results["search_successful"]:
            return jsonify({
                "error": "Web search failed",
                "details": search_results.get("error", "Unknown error")
            }), 500
        
        web_vendors = search_results["vendors"]
        
        if not web_vendors:
            return jsonify({
                "message": "No vendors found in web search",
                "total_found": 0,
                "vendors": [],
                "sources_count": 0,
                "top_score": 0
            }), 200
        
        # Evaluate the web vendors
        evaluated_vendors = evaluate_web_vendors(ps_analysis, web_vendors, evaluation_params)
        
        # Prepare response
        response = {
            "problem_statement_id": ps_id,
            "total_found": len(evaluated_vendors),
            "sources_count": search_results["sources_count"],
            "top_score": evaluated_vendors[0]["composite_score"] if evaluated_vendors else 0,
            "vendors": evaluated_vendors,
            "evaluation_params": evaluation_params
        }
        
        logger.info(f"Web search complete: found {len(evaluated_vendors)} vendors")
        return jsonify(response), 200
        
    except ValueError as ve:
        logger.warning(f"Validation error: {str(ve)}")
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        logger.error(f"Web search vendors error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": "An internal error occurred during web search"}), 500


@app.route('/api/vendors', methods=['GET'])
def get_all_vendors():
    """Get all vendors with their information"""
    try:
        vendors = list(vendors_collection.find())
        
        # Enrich vendor data with capabilities if available
        enriched_vendors = []
        for vendor in vendors:
            vendor_hash = get_content_hash(f"{vendor['name']}:{vendor['text']}")
            capabilities = load_cached_analysis(vendor_capabilities_collection, vendor_hash)
            
            vendor_info = {
                "name": vendor["name"],
                "text_preview": vendor["text"][:500] + "..." if len(vendor["text"]) > 500 else vendor["text"],
                "full_text_length": len(vendor["text"]),
                "capabilities": capabilities if capabilities else None,
                "has_embedding": load_embedding(vendor_embeddings_collection, vendor_hash) is not None
            }
            enriched_vendors.append(vendor_info)
        
        # Sort by name
        enriched_vendors.sort(key=lambda x: x["name"].lower())
        
        logger.info(f"Fetched {len(enriched_vendors)} vendors")
        return jsonify({
            "total": len(enriched_vendors),
            "vendors": enriched_vendors
        }), 200
    except Exception as e:
        logger.error(f"Error fetching vendors: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": "Failed to fetch vendors"}), 500


@app.route('/api/vendors/<vendor_name>', methods=['GET'])
def get_vendor_details(vendor_name):
    """Get detailed information for a specific vendor"""
    try:
        vendor = vendors_collection.find_one({"name": vendor_name})
        if not vendor:
            return jsonify({"error": "Vendor not found"}), 404
        
        vendor_hash = get_content_hash(f"{vendor['name']}:{vendor['text']}")
        capabilities = load_cached_analysis(vendor_capabilities_collection, vendor_hash)
        embedding = load_embedding(vendor_embeddings_collection, vendor_hash)
        
        vendor_details = {
            "name": vendor["name"],
            "full_text": vendor["text"],
            "text_length": len(vendor["text"]),
            "capabilities": capabilities,
            "has_embedding": embedding is not None,
            "embedding_dimensions": len(embedding) if embedding is not None else 0
        }
        
        return jsonify(vendor_details), 200
    except Exception as e:
        logger.error(f"Error fetching vendor details: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": "Failed to fetch vendor details"}), 500


@app.route('/api/vendors/<vendor_name>', methods=['DELETE'])
def delete_vendor(vendor_name):
    """Delete a vendor and its cached data"""
    try:
        vendor = vendors_collection.find_one({"name": vendor_name})
        if not vendor:
            return jsonify({"error": "Vendor not found"}), 404
        
        # Delete vendor
        vendors_collection.delete_one({"name": vendor_name})
        
        # Delete cached data
        vendor_hash = get_content_hash(f"{vendor['name']}:{vendor['text']}")
        vendor_capabilities_collection.delete_one({"content_hash": vendor_hash})
        vendor_embeddings_collection.delete_one({"content_hash": vendor_hash})
        
        logger.info(f"Deleted vendor: {vendor_name}")
        return jsonify({"message": f"Vendor '{vendor_name}' deleted successfully"}), 200
    except Exception as e:
        logger.error(f"Error deleting vendor: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": "Failed to delete vendor"}), 500


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error"""
    return jsonify({"error": f"File too large. Maximum size: {app.config['MAX_CONTENT_LENGTH'] / (1024*1024):.1f}MB"}), 413


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({"error": "Internal server error occurred"}), 500


if __name__ == '__main__':
    logger.info("Starting Flask application")
    logger.info(f"Using OpenAI embedding model: {EMBEDDING_MODEL} with {EMBEDDING_DIMENSIONS} dimensions")
    app.run(debug=True, host='0.0.0.0', port=5000)