# config_models.py
import os
from pydantic import BaseModel, Field
from typing import Optional
from dotenv import load_dotenv

class EmbeddingConfig(BaseModel):
    """Configuration for HuggingFace Embeddings."""
    # If using a specific model from HuggingFace Hub directly (e.g., for sentence-transformers)
    # model_name: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", description="Default embedding model name")
    # If using an HF Inference Endpoint for embeddings
    endpoint_url_env_var: str = Field(default="HF_EMBED_ENDPOINT", description="Environment variable for the embedding endpoint URL.")
    # Common parameters for HuggingFaceEndpointEmbeddings
    task: str = Field(default="feature-extraction", description="Task for the embedding endpoint.")
    # You can add other HuggingFaceEndpointEmbeddings parameters if needed, e.g., model_kwargs

class TextLoaderConfig(BaseModel):
    """Configuration for loading text documents."""
    document_path: str = Field(description="Path to the text file containing documents.")
    # encoding: str = Field(default="utf-8") # Example: if you need to specify encoding

class TextSplitterConfig(BaseModel):
    """Configuration for text splitting."""
    chunk_size: int = Field(default=1000, description="Size of text chunks.")
    chunk_overlap: int = Field(default=200, description="Overlap between text chunks.")
    # add_start_index: bool = Field(default=True) # Example of another parameter for RecursiveCharacterTextSplitter

class VectorStoreConfig(BaseModel):
    """Configuration for the vector store."""
    persist_directory: Optional[str] = Field(default="faiss_index_store", description="Directory to save/load the FAISS index. If None, in-memory only.")
    # You might add specific FAISS parameters here if needed

class RetrieverConfig(BaseModel):
    """Configuration for the retrieval component."""
    loader_config: TextLoaderConfig
    splitter_config: TextSplitterConfig
    embedding_config: EmbeddingConfig
    vector_store_config: VectorStoreConfig
    indexing_batch_size: int = Field(default=32, description="Batch size for indexing documents into FAISS.")
    # top_k_retrieval: int = Field(default=4, description="Number of documents to retrieve by the retriever.") # Example

class PromptConfig(BaseModel):
    """Configuration for RAG prompts."""
    template_string: str = Field(
        default=(
            "You are an assistant for question-answering tasks. Use the following pieces of retrieved context "
            "to answer the question. If you don't know the answer, just say that you don't know. "
            "Use three sentences maximum and keep the answer concise.\n"
            "Question: {query}\n"
            "Context: {context}\n"
            "Answer:"
        ),
        description="The template string for the RAG prompt."
    )
    input_variables: list[str] = Field(default_factory=lambda: ["context", "query"])

class LLMConfig(BaseModel):
    """Configuration for the HuggingFace LLM Endpoint."""
    endpoint_url_env_var: str = Field(default="HF_LLM_ENDPOINT", description="Environment variable for the LLM endpoint URL.")
    # Common parameters for HuggingFaceEndpoint
    task: str = Field(default="text-generation", description="Task for the LLM endpoint.")
    max_new_tokens: int = Field(default=512, description="Maximum new tokens to generate.")
    top_k: Optional[int] = Field(default=10, description="Top-k sampling parameter.")
    top_p: Optional[float] = Field(default=0.95, description="Top-p (nucleus) sampling parameter.")
    typical_p: Optional[float] = Field(default=0.95, description="Typical-p sampling parameter.")
    temperature: float = Field(default=0.7, description="Sampling temperature.")
    repetition_penalty: Optional[float] = Field(default=1.03, description="Repetition penalty.")
    # You can add other HuggingFaceEndpoint parameters as needed

class AppSettings(BaseModel):
    """Master configuration for the application, loading from environment variables."""
    hf_token_env_var: str = Field(default="HF_TOKEN", description="Environment variable for the HuggingFace API Token.")
    
    # Nested configurations
    retriever_config: RetrieverConfig
    prompt_config: PromptConfig = PromptConfig() # Use default prompt config
    llm_config: LLMConfig = LLMConfig() # Use default LLM config

    # Chainlit specific settings (optional)
    assistant_name: str = Field(default="Paul Graham Essay Bot")

    # Actual loaded values from env vars
    hf_llm_endpoint_url: Optional[str] = None
    hf_embed_endpoint_url: Optional[str] = None
    hf_api_token: Optional[str] = None

    class Config:
        env_file = ".env" # Specify .env file for Pydantic to load from (optional, can also use load_dotenv explicitly)
        extra = "ignore" # Ignore extra fields from .env

# Function to load configuration and environment variables
def load_settings() -> AppSettings:
    """Loads settings, ensuring environment variables are accessible."""
    load_dotenv() # Explicitly load .env, useful for broader compatibility

    # Example: Define specific paths for your document loader here or make them env vars too
    # For this example, let's assume paul_graham_essays.txt is in a 'data' subdirectory
    doc_path = os.path.join(os.path.dirname(__file__), "data", "paul_graham_essays.txt")
    # Create dummy data dir and file if it doesn't exist for the example to run
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(doc_path):
        with open(doc_path, "w") as f:
            f.write("This is a sample essay by Paul Graham about startups and programming. It contains wisdom.\n")
            f.write("Another paragraph discussing the importance of Lisp and hackers.\n")


    retriever_cfg = RetrieverConfig(
        loader_config=TextLoaderConfig(document_path=doc_path),
        splitter_config=TextSplitterConfig(chunk_size=500, chunk_overlap=50), # Adjusted for example
        embedding_config=EmbeddingConfig(), # Uses default HF_EMBED_ENDPOINT
        vector_store_config=VectorStoreConfig(persist_directory="faiss_pg_essays_index") # Example persist path
    )

    settings = AppSettings(
        retriever_config=retriever_cfg
        # prompt_config and llm_config will use their defaults unless overridden here
    )

    # Load actual endpoint URLs and token from environment variables
    settings.hf_llm_endpoint_url = os.getenv(settings.llm_config.endpoint_url_env_var)
    settings.hf_embed_endpoint_url = os.getenv(settings.retriever_config.embedding_config.endpoint_url_env_var)
    settings.hf_api_token = os.getenv(settings.hf_token_env_var)

    # Basic validation
    if not settings.hf_llm_endpoint_url:
        raise ValueError(f"Environment variable {settings.llm_config.endpoint_url_env_var} not set.")
    if not settings.hf_embed_endpoint_url:
        raise ValueError(f"Environment variable {settings.retriever_config.embedding_config.endpoint_url_env_var} not set.")
    if not settings.hf_api_token:
        raise ValueError(f"Environment variable {settings.hf_token_env_var} not set.")
    
    print(f"Settings loaded. LLM Endpoint: {settings.hf_llm_endpoint_url}, Embed Endpoint: {settings.hf_embed_endpoint_url}")
    return settings

# To test loading (optional, can be removed)
# if __name__ == "__main__":
#     try:
#         app_settings = load_settings()
#         print("Successfully loaded settings:")
#         print(f"  Assistant Name: {app_settings.assistant_name}")
#         print(f"  Document Path: {app_settings.retriever_config.loader_config.document_path}")
#         print(f"  LLM Max New Tokens: {app_settings.llm_config.max_new_tokens}")
#         print(f"  HF Token Loaded: {'Yes' if app_settings.hf_api_token else 'No'}")
#     except ValueError as e:
#         print(f"Error loading settings: {e}")
#     except Exception as e:
#         print(f"An unexpected error occurred: {e}")
