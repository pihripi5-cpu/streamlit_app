# ---------------------------------------------------------------------------
# 3. Groq LLM & LCEL RAG Pipeline
# ---------------------------------------------------------------------------

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama3-70b-8192", # Corrected Groq model name
    temperature=0.3
)

store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def create_rag_chain(persona_name: str, retrieved_context: str):
    system_instruction = SALES_PROMPTS.get(
        persona_name,
        SALES_PROMPTS["PragyanAI Student Counselor"]
    ).format(context=retrieved_context)

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_instruction),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])

    return prompt | llm | StrOutputParser()
# ---------------------------------------------------------------------------
# 4. Gradio Callbacks
# ---------------------------------------------------------------------------
def respond(message, history, persona_name):
    if not message.strip():
        return ""

    # Search top relevant snippets
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    relevant_docs = retriever.invoke(message)
    context_str = "\n".join([f"- {doc.page_content}" for doc in relevant_docs])

    session_id = f"pragyan_session_{persona_name.replace(' ', '_')}"
    base_chain = create_rag_chain(persona_name, context_str)

    conversational_chain = RunnableWithMessageHistory(
        base_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    return conversational_chain.invoke(
        {"input": message},
        config={"configurable": {"session_id": session_id}}
    )

def clear_chat_history(persona_name):
    session_id = f"pragyan_session_{persona_name.replace(' ', '_')}"
    if session_id in store:
        store[session_id].clear()
# ---------------------------------------------------------------------------
# 5. Gradio User Interface
# ---------------------------------------------------------------------------
with gr.Blocks(title="PragyanAI Intelligent Assistant") as demo:
    gr.Markdown("# PragyanAI Conversational Sales & FAQ Assistant")
    gr.Markdown("Answers program questions based on the **PragyanAI Presentation & FAQ Sheet**.")

    with gr.Row():
        with gr.Column(scale=1):
            persona_selector = gr.Dropdown(
                choices=list(SALES_PROMPTS.keys()),
                value="PragyanAI Student Counselor",
                label="Select PragyanAI Persona",
                interactive=True
            )
            file_uploader = gr.File(
                label="Upload Additional PDFs or Excel Sheets",
                file_count="multiple",
                file_types=[".pdf", ".xlsx", ".xls"]
            )
            upload_status = gr.Textbox(label="Knowledge Base Status", value="PragyanAI presentation FAQ pre-loaded.", interactive=False)
            file_uploader.change(fn=load_documents_into_vectorstore, inputs=[file_uploader], outputs=[upload_status])

        with gr.Column(scale=3):
            chatbot_ui = gr.ChatInterface(
                fn=respond,
                additional_inputs=[persona_selector]
            )
            clear_btn = gr.Button("Clear Memory for Selected Persona", variant="secondary")
            clear_btn.click(fn=clear_chat_history, inputs=[persona_selector], outputs=None)

if __name__ == "__main__":
    demo.launch(share=True, debug=True)
