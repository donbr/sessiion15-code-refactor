# session15-code-refactor

**Key Improvements and Explanations:**

1. **config_models.py:**  
   * **Pydantic Models:** Defines classes like EmbeddingConfig, TextLoaderConfig, LLMConfig, AppSettings, etc. This provides:  
     * **Type Hinting & Validation:** Ensures configuration values are of the correct type.  
     * **Defaults:** Centralized default values.  
     * **Clear Structure:** Easy to understand what parameters are available and required.  
     * **Environment Variable Loading:** The load_settings function now explicitly loads from .env and populates the settings, including fetching the actual endpoint URLs and token. It also performs basic validation to ensure essential environment variables are set.  
     * **Document Path:** The document path is now constructed more robustly and a dummy file/directory creation is included for easier first-time setup.
2. **services.py:**  
   * **DocumentService:**  
     * Encapsulates all logic for document loading (TextLoader), splitting (RecursiveCharacterTextSplitter), embedding (HuggingFaceEndpointEmbeddings), and vector store management (FAISS).  
     * The initialize_retriever method handles loading the FAISS index from disk if available and configured, or re-indexing if not. It now uses asyncio.to_thread for synchronous Langchain/FAISS operations (like load_local, save_local, text_loader.load, text_splitter.split_documents) to prevent blocking the asyncio event loop.  
     * Batch-wise indexing (_index_documents_batchwise) is implemented with tqdm for progress.  
   * **LanguageModelService:**  
     * Manages the instantiation of HuggingFaceEndpoint for the LLM, configured via LLMConfig.  
   * **RAGChainService:**  
     * Builds the LangChain Expression Language (LCEL) chain. It takes the retriever (from DocumentService) and the LLM (from LanguageModelService) to construct the RAG pipeline.  
     * The format_docs helper is included here.  
3. **app.py (Main Application):**  
   * **Cleaner Structure:** The main file is now much more focused on orchestration and Chainlit event handling.  
   * **Centralized Settings:** APP_SETTINGS is loaded once at startup.  
   * **Service Initialization:** Services (DOCUMENT_SERVICE, LLM_SERVICE, RAG_CHAIN_SERVICE) are instantiated globally using the loaded settings.  
   * **Asynchronous Initialization (ensure_services_initialized):**  
     * The asyncio.Event (_vectorstore_initialized_event) and asyncio.Lock (_initialization_lock) are used to ensure that the potentially long-running task of initializing the DocumentService (especially document indexing) happens only once and that on_chat_start waits for it to complete.  
     * This function is called at the beginning of on_chat_start.  
   * **@cl.on_chat_start:**  
     * Waits for services to be initialized.  
     * Retrieves the retriever from DOCUMENT_SERVICE.  
     * Uses RAG_CHAIN_SERVICE to build the lcel_rag_chain.  
     * Handles potential initialization errors more gracefully by informing the user.  
   * **@cl.on_message (renamed to main_message_handler):**  
     * Retrieves the chain from the user session.  
     * Passes the user's query (now structured as {"query": message.content}) to the chain's astream method.  
   * **Error Handling:** Basic error handling is added for configuration loading and service initialization, providing feedback to the console and potentially to the user in the chat interface.

**To Use This Refactored Code:**

1. **Save the Files:** Create three files: config_models.py, services.py, and app.py in the same directory.  
2. **Create .env File:** In the same directory, create a .env file with your HuggingFace token and endpoint URLs:  
   Code snippet  
   HF_TOKEN="your_actual_huggingface_api_token"  
   HF_LLM_ENDPOINT="your_llm_inference_endpoint_url"  
   HF_EMBED_ENDPOINT="your_embedding_inference_endpoint_url"

3. **Data Directory:** The config_models.py script will attempt to create a data subdirectory and a dummy paul_graham_essays.txt file within it if they don't exist. You can replace this with your actual data.  
4. **Install Dependencies:**  
   ```bash  
   pip install chainlit langchain pydantic python-dotenv langchain-huggingface faiss-cpu tqdm sentence-transformers  
   *(Note: faiss-cpu is for CPU-based FAISS. Use faiss-gpu if you have a compatible GPU and CUDA setup.)* *(sentence-transformers might not be strictly needed if you are only using HuggingFaceEndpointEmbeddings with an inference endpoint, but it's often a useful part of the Langchain ecosystem.)*
   ```  
5. **Run Chainlit:**  
   ```bash  
   chainlit run app.py -w
   ```

This structure provides a solid foundation for your application, making it easier to manage configurations, test individual components, and extend functionality in the future.
