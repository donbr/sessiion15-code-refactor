# app.py (Modified)
import os
import chainlit as cl
from langchain.schema.runnable.config import RunnableConfig
import asyncio

# Import configurations and services
from config_models import load_settings, AppSettings
from services import DocumentService, LanguageModelService, RAGChainService

# ---- GLOBAL APP INITIALIZATION ---- #
# Load application settings once when the application starts
# This will also load .env variables and perform basic validation.
try:
    APP_SETTINGS: AppSettings = load_settings()
except ValueError as e:
    print(f"FATAL: Configuration error: {e}")
    # In a real app, you might exit or have a fallback mechanism
    # For Chainlit, it might be hard to stop server startup here directly,
    # errors will be logged, and on_chat_start might fail.
    APP_SETTINGS = None # Indicate failure
    # raise # Optionally re-raise to stop if possible

# Initialize services. These can be singletons if their state is managed appropriately.
# For Chainlit, services are typically initialized once and then accessed.
if APP_SETTINGS:
    DOCUMENT_SERVICE = DocumentService(
        retriever_config=APP_SETTINGS.retriever_config,
        embed_endpoint_url=APP_SETTINGS.hf_embed_endpoint_url,
        api_token=APP_SETTINGS.hf_api_token
    )
    LLM_SERVICE = LanguageModelService(
        llm_config=APP_SETTINGS.llm_config,
        endpoint_url=APP_SETTINGS.hf_llm_endpoint_url,
        api_token=APP_SETTINGS.hf_api_token
    )
    RAG_CHAIN_SERVICE = RAGChainService(
        prompt_config=APP_SETTINGS.prompt_config,
        llm_service=LLM_SERVICE
    )
else: # Handle case where settings failed to load
    DOCUMENT_SERVICE = None
    LLM_SERVICE = None
    RAG_CHAIN_SERVICE = None
    print("WARNING: Services not initialized due to configuration errors.")


# Global flag and lock to ensure vector store initialization happens once.
_vectorstore_initialized_event = asyncio.Event()
_initialization_lock = asyncio.Lock()

async def ensure_services_initialized():
    """
    Ensures that the DocumentService (and its vector store) is initialized.
    This is crucial before the first chat starts if the retriever is needed.
    """
    async with _initialization_lock:
        if not _vectorstore_initialized_event.is_set():
            if not DOCUMENT_SERVICE:
                print("Error: DocumentService not available for initialization.")
                # This indicates a severe config problem from startup.
                # We might want to signal this to the user in on_chat_start.
                return False # Indicate failure

            print("ensure_services_initialized: Starting DocumentService initialization (vector store)...")
            try:
                # force_reindex can be set to True to always re-index on startup,
                # or get this from AppSettings if you want it configurable.
                await DOCUMENT_SERVICE.initialize_retriever(force_reindex=False)
                _vectorstore_initialized_event.set() # Signal that initialization is complete
                print("ensure_services_initialized: DocumentService initialization complete.")
                return True # Indicate success
            except FileNotFoundError as fnf_error:
                print(f"Error during DocumentService initialization: {fnf_error}")
                # This error is critical if documents are expected.
                # The app might not be usable.
                # Consider how to communicate this to the user in on_chat_start.
                return False # Indicate failure
            except Exception as e:
                print(f"Error during DocumentService initialization: {e}")
                # Other unexpected errors.
                return False # Indicate failure
    return True # Already initialized or successfully initialized now


# ---- CHAINLIT HOOKS ---- #

@cl.author_rename
def rename(original_author: str):
    """Renames the 'Assistant' author using the name from settings."""
    if APP_SETTINGS and original_author == "Assistant":
        return APP_SETTINGS.assistant_name
    return original_author

@cl.on_chat_start
async def start_chat():
    """Called at the start of every user session."""
    if not APP_SETTINGS or not DOCUMENT_SERVICE or not RAG_CHAIN_SERVICE:
        await cl.Message(content="Critical error: Application settings or services failed to load. The chatbot may not function correctly. Please check server logs.").send()
        cl.user_session.set("lcel_rag_chain", None)
        return

    # Ensure services, especially the vector store, are initialized.
    # This will block here until initialization is done or fails.
    init_success = await ensure_services_initialized()

    if not init_success or not _vectorstore_initialized_event.is_set():
        await cl.Message(content="Error: Could not initialize the document retrieval system. The chatbot may not be able to answer questions based on documents. Please check server logs.").send()
        cl.user_session.set("lcel_rag_chain", None)
        return

    try:
        retriever = DOCUMENT_SERVICE.get_retriever()
    except ValueError as e: # If get_retriever is called before init somehow (should be caught by event)
        await cl.Message(content=f"Error: Could not get retriever. {e} Please check server logs.").send()
        cl.user_session.set("lcel_rag_chain", None)
        return

    # Build the RAG chain using the initialized retriever
    lcel_rag_chain = RAG_CHAIN_SERVICE.build_chain(retriever)
    cl.user_session.set("lcel_rag_chain", lcel_rag_chain)

    await cl.Message(
        content=f"Hello! I am the {APP_SETTINGS.assistant_name}. How can I help you with Paul Graham's essays today?"
    ).send()


@cl.on_message  
async def main_message_handler(message: cl.Message): # Renamed to avoid conflict with original main() if it existed
    """Called every time a message is received from a session."""
    lcel_rag_chain = cl.user_session.get("lcel_rag_chain")

    if not lcel_rag_chain:
        await cl.Message(content="The RAG chain is not available, possibly due to an initialization error. Please try restarting the chat or contact support.").send()
        return

    # Prepare the input for the chain. It expects a dictionary.
    # If your chain's RunnablePassthrough() is for the whole input, pass message.content directly.
    # If it's for a specific key like "query", then: chain_input = {"query": message.content}
    chain_input = {"query": message.content} # Matching the RAGChainService setup

    msg_ui = cl.Message(content="") # Initialize an empty message for streaming
    await msg_ui.send() 

    full_response = ""
    async for chunk in lcel_rag_chain.astream(
        chain_input, # Pass the structured input
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg_ui.stream_token(chunk)
        full_response += chunk

    # msg_ui.content = full_response # Update with full content if needed, stream_token should handle UI
    await msg_ui.update() # Ensure the final message state is saved/updated

# Optional: A way to run the initialization outside of Chainlit's direct flow if needed,
# for example, in a __main__ block or a separate setup script.
# However, for Chainlit, ensure_services_initialized called from on_chat_start
# is a common pattern for async setup that needs to complete before handling requests.

# To run this app:
# 1. Save the files as config_models.py, services.py, app.py in the same directory.
# 2. Create a .env file in that directory with your HF_TOKEN, HF_LLM_ENDPOINT, HF_EMBED_ENDPOINT.
#    Example .env:
#    HF_TOKEN="your_huggingface_token"
#    HF_LLM_ENDPOINT="your_llm_inference_endpoint_url"
#    HF_EMBED_ENDPOINT="your_embedding_inference_endpoint_url"
# 3. Create a subdirectory named 'data' and place 'paul_graham_essays.txt' in it (or let the dummy file be created).
# 4. Install necessary packages: pip install chainlit langchain pydantic python-dotenv langchain-huggingface faiss-cpu tqdm sentence-transformers
#    (sentence-transformers might not be strictly needed if only using HF_EMBED_ENDPOINT, but good for general Langchain use)
# 5. Run: chainlit run app.py -w
