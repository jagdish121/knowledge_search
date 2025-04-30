import re
import PyPDF2
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Load embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

def preprocess_text(text):
    # Basic preprocessing
    text = re.sub(r'\s+', ' ', text)  # Remove multiple spaces/newlines
    text = text.strip()
    return text

def chunk_text(text, chunk_size=500):
    words = text.split()
    chunks = [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ''
    for page in reader.pages:
        text += page.extract_text()
    return text

def embed_texts(chunks):
    embeddings = embedder.encode(chunks).tolist()
    return embeddings

def initialize_qdrant(collection_name, cluster_url, qdrant_key):
    client = QdrantClient(
    url=cluster_url, 
    api_key=qdrant_key,
)  # In-memory for free local use
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )
    return client

def split_text(text, max_length=512):
    sentences = text.split(". ")
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < max_length:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    if current_chunk:
        chunks.append(current_chunk.strip())  # Add any remaining chunk
    return chunks

# Preprocess text (e.g., lowercase, remove punctuation)
def preprocess_text_lowercase(text):
    return text.lower()  # Simple preprocessing, you can enhance it

# Function to upload FAQ data to Qdrant
def upload_to_qdrant(client, collection_name, faq_texts, embeddings):
    points = []
    for i, faq_text in enumerate(faq_texts):
        # Split large FAQ text into smaller chunks
        small_chunks = split_text(faq_text)
        for small_chunk in small_chunks:
            point = PointStruct(
                id=i,
                vector=embeddings[i],  # Assuming embeddings are precomputed
                payload={"text": small_chunk}
            )
            points.append(point)
    # Insert points into Qdrant
    client.upsert(collection_name=collection_name, points=points)

# Function to search Qdrant for the relevant FAQ chunk
def search_qdrant(client, collection_name, query, top_k=5):
    # Preprocess and encode query to vector
    query_vector = embedder.encode(query).tolist()
    
    # Perform search
    search_result = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=top_k
    )
    
    # Return the text from the search result
    return [hit.payload['text'] for hit in search_result]
