# services.py
import os
import asyncio
from tqdm.asyncio import tqdm
from typing import List

from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEndpointEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableSequence
from langchain.schema.output_parser import StrOutputParser
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.documents import Document

# Assuming config_models.py is in the same directory
from config_models import AppSettings, RetrieverConfig, LLMConfig, PromptConfig

class DocumentService:
    """
    Handles loading, splitting, embedding documents, and managing the vector store.
    """
    def __init__(self, retriever_config: RetrieverConfig, embed_endpoint_url: str, api_token: str):
        self.config = retriever_config
        self.embed_endpoint_url = embed_endpoint_url
        self.api_token = api_token
        
        self._vectorstore: Optional[FAISS] = None
        self._retriever: Optional[VectorStoreRetriever] = None
        
        # Initialize embeddings
        self.hf_embeddings = HuggingFaceEndpointEmbeddings(
            model=self.embed_endpoint_url, # This is the HF Inference Endpoint URL
            task=self.config.embedding_config.task,
            huggingfacehub_api_token=self.api_token
        )
        print(f"DocumentService: Initialized embeddings with endpoint: {self.embed_endpoint_url}")

    async def _load_and_split_documents(self) -> List[Document]:
        """Loads documents from the path specified in config and splits them."""
        doc_path = self.config.loader_config.document_path
        print(f"DocumentService: Loading documents from: {doc_path}")
        if not os.path.exists(doc_path):
            raise FileNotFoundError(f"Document file not found at {doc_path}. Please create it or update the path in config_models.py.")

        text_loader = TextLoader(doc_path)
        # Langchain's load is synchronous, run in a thread to avoid blocking asyncio event loop
        documents = await asyncio.to_thread(text_loader.load)

        print(f"DocumentService: Splitting {len(documents)} document(s)...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.splitter_config.chunk_size,
            chunk_overlap=self.config.splitter_config.chunk_overlap
            # add_start_index=self.config.splitter_config.add_start_index, # If configured
        )
        split_documents = await asyncio.to_thread(text_splitter.split_documents, documents)
        print(f"DocumentService: Split into {len(split_documents)} chunks.")
        return split_documents

    async def _index_documents_batchwise(self, split_documents: List[Document]) -> FAISS:
        """Indexes documents into FAISS batchwise with progress."""
        print("DocumentService: Indexing documents into FAISS...")
        vectorstore_instance = None
        batches = [
            split_documents[i:i + self.config.indexing_batch_size]
            for i in range(0, len(split_documents), self.config.indexing_batch_size)
        ]

        first_batch_processed = False
        # Use tqdm for progress indication
        for i, batch in enumerate(tqdm(batches, desc="Indexing Batches")):
            if not first_batch_processed:
                # Create the vectorstore from the first batch
                vectorstore_instance = await FAISS.afrom_documents(batch, self.hf_embeddings)
                first_batch_processed = True
            else:
                # Add subsequent batches to the existing vectorstore
                if vectorstore_instance: # Should always be true after first batch
                    await vectorstore_instance.aadd_documents(batch)
            # Consider a small sleep if rate limiting is an issue, though aadd_documents handles some async internally.
            # await asyncio.sleep(0.1) 
        
        if not vectorstore_instance:
             raise ValueError("DocumentService: Failed to create vector store instance. No documents processed or an error occurred.")
        return vectorstore_instance


    async def initialize_retriever(self, force_reindex: bool = False) -> VectorStoreRetriever:
        """
        Initializes the vector store and retriever.
        Tries to load from disk if persist_directory is set and index exists,
        otherwise, re-indexes and optionally saves.
        """
        persist_dir = self.config.vector_store_config.persist_directory
        
        if not force_reindex and persist_dir and os.path.exists(persist_dir):
            try:
                print(f"DocumentService: Attempting to load FAISS index from {persist_dir}")
                # FAISS.load_local is synchronous
                self._vectorstore = await asyncio.to_thread(
                    FAISS.load_local,
                    folder_path=persist_dir,
                    embeddings=self.hf_embeddings,
                    allow_dangerous_deserialization=True # Required by FAISS for loading pickled data
                )
                print("DocumentService: FAISS index loaded successfully from disk.")
            except Exception as e:
                print(f"DocumentService: Failed to load FAISS index from {persist_dir}: {e}. Re-indexing.")
                self._vectorstore = None # Ensure it's None so re-indexing occurs

        if not self._vectorstore: # If not loaded or force_reindex is True
            print("DocumentService: No existing index found or re-indexing forced. Processing documents...")
            split_documents = await self._load_and_split_documents()
            if not split_documents:
                raise ValueError("DocumentService: No documents found or loaded to index.")
            
            self._vectorstore = await self._index_documents_batchwise(split_documents)
            
            if persist_dir and self._vectorstore:
                print(f"DocumentService: Saving FAISS index to {persist_dir}")
                # FAISS.save_local is synchronous
                await asyncio.to_thread(self._vectorstore.save_local, persist_dir)
        
        if not self._vectorstore:
            raise RuntimeError("DocumentService: Vectorstore could not be initialized.")

        # Configure retriever, e.g., with top_k if set in config
        # search_kwargs = {}
        # if hasattr(self.config, 'top_k_retrieval') and self.config.top_k_retrieval:
        #     search_kwargs['k'] = self.config.top_k_retrieval
        self._retriever = self._vectorstore.as_retriever() # search_kwargs=search_kwargs
        print("DocumentService: Retriever initialized.")
        return self._retriever

    def get_retriever(self) -> VectorStoreRetriever:
        """Returns the initialized retriever. Raises error if not initialized."""
        if not self._retriever:
            raise ValueError("DocumentService: Retriever not initialized. Call initialize_retriever() first.")
        return self._retriever

class LanguageModelService:
    """Handles LLM interactions using HuggingFaceEndpoint."""
    def __init__(self, llm_config: LLMConfig, endpoint_url: str, api_token: str):
        self.config = llm_config
        self.endpoint_url = endpoint_url
        self.api_token = api_token
        
        self.llm = HuggingFaceEndpoint(
            endpoint_url=self.endpoint_url,
            huggingfacehub_api_token=self.api_token,
            task=self.config.task,
            max_new_tokens=self.config.max_new_tokens,
            top_k=self.config.top_k,
            top_p=self.config.top_p,
            typical_p=self.config.typical_p,
            temperature=self.config.temperature,
            repetition_penalty=self.config.repetition_penalty,
        )
        print(f"LanguageModelService: Initialized LLM with endpoint: {self.endpoint_url}")

    def get_llm(self) -> HuggingFaceEndpoint:
        return self.llm

class RAGChainService:
    """Builds and provides the RAG LCEL chain."""
    def __init__(self, prompt_config: PromptConfig, llm_service: LanguageModelService):
        self.prompt_config = prompt_config
        self.llm_service = llm_service
        
        self.rag_prompt_template = PromptTemplate(
            template=self.prompt_config.template_string,
            input_variables=self.prompt_config.input_variables
        )
        print("RAGChainService: Initialized with prompt template.")

    def build_chain(self, retriever: VectorStoreRetriever) -> RunnableSequence:
        """Constructs the RAG chain."""
        
        # Helper function to format retrieved documents
        def format_docs(docs: List[Document]) -> str:
            return "\n\n".join(doc.page_content for doc in docs)

        # Define the RAG chain using LCEL
        # The input to this chain will be a dictionary like {"query": "user's question"}
        rag_chain = (
            {
                "context": retriever | format_docs,  # Retrieve context then format it
                "query": RunnablePassthrough()  # Pass the original query through
            }
            | self.rag_prompt_template          # Populate the prompt template
            | self.llm_service.get_llm()       # Send to the LLM
            | StrOutputParser()                # Parse the LLM output as a string
        )
        print("RAGChainService: RAG chain built.")
        return rag_chain
