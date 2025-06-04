import gradio as gr
import pandas as pd
import os
import shutil

from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

from get_embedding_function import get_embedding_function
from query_data import CHROMA_PATH, PROMPT_TEMPLATE
from populate_database import load_documents, split_documents, add_to_chroma
from reranking import rerank_documents
from populate_database import DATA_PATH

import logging 
LOGGER = logging.getLogger(__name__)


def query_rag(query_text: str):
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=10)

    # rerank the results 
    results = rerank_documents(query_text, results)

    # Prepare dataframe data
    data = {
        "Score": [],
        "Source": [],
        "Page": [],
        "Content": []
    }
    
    for i, (doc, score) in enumerate(results):
        data["Score"].append(f"{score:.4f}")
        data["Source"].append(str(doc.metadata.get("source", "Unknown")))
        data["Page"].append(str(doc.metadata.get("page", "N/A")))
        
        # Handle content display
        content = str(doc.page_content)
        # preview = content
        data["Content"].append(content)
    
    # Create dataframe with explicit types
    df_output = pd.DataFrame(data).sort_values(by="Score", ascending=False)
    
    # return df_output first
    yield df_output, gr.update(visible=True, value=""), gr.update(visible=True, value="")
    
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)

    # return df_output, prompt 
    yield df_output, prompt, gr.update(visible=True, value="hang tight ....")

    model = OllamaLLM(model="llama3.2:latest")
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    
    yield df_output, prompt, response_text

def handle_upload(files):
    saved_files = []

    # Save uploaded files to the data directory
    for file in files:
        filename = os.path.basename(file.name)
        dest_path = os.path.join(DATA_PATH, filename)
        shutil.copy(file.name, dest_path)
        saved_files.append(filename)

    # similar to populate_database.py
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)


# Gradio Interface
with gr.Blocks(title="RAG System with Chroma DB") as demo:
    gr.Markdown("# RAG System with Local Chroma DB")
    
    with gr.Row():
        with gr.Column():
            # PDF Upload Section
            gr.Markdown("## Add New Documents to Database")
            file_input = gr.File(
                file_count="multiple",
                file_types=[".pdf"],
                label="Upload PDF Files"
            )
            upload_button = gr.Button("Add to Database")
            upload_output = gr.Textbox(label="Upload Status")
            
            # Query Section
            gr.Markdown("## Query Your Knowledge Base")
            query_text = gr.Textbox(
                label="Enter your question",
                placeholder="Type your question here..."
            )
            query_button = gr.Button("Submit Query")

            # Results Sections
            gr.Markdown("## Generated Answer")
            response_text = gr.Textbox(
                label="Answer",
                interactive=False,
                lines=10
            )
        
        with gr.Column():
            # Results Sections
            gr.Markdown("## Retrieval Results (Top 5)")
            results = gr.Dataframe(
                label="Relevant Documents",
                interactive=False,
                wrap=True,
                headers=["Rank", "Score", "Source", "Page", "Content Preview"],
                datatype=["number", "str", "str", "str", "str"]
            )
            
            gr.Markdown("## Final Prompt Structure")
            prompt = gr.Textbox(
                label="Prompt",
                interactive=False,
                lines=10
            )
            
            
    
    # Event handlers
    upload_button.click(
        fn=handle_upload,
        inputs=[file_input],
        outputs=[upload_output]
    )
    
    query_button.click(
        fn=query_rag,
        inputs=[query_text],
        outputs=[results, prompt, response_text]
    )


if __name__ == "__main__":
    # Launch the app
    demo.launch(server_port=7860)
