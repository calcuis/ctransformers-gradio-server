from ctransformers import AutoModelForCausalLM

def load_llm():
    llm = AutoModelForCausalLM.from_pretrained(
        "chat.gguf",
        model_type="llama",
        max_new_tokens=2048,
        repetition_penalty=1.13,
        temperature=0.1
    )
    return llm

def llm_function(message,chat_history):
    llm=load_llm()
    response=llm(message)
    output_texts=message+response
    return output_texts

import gradio as gr

title = "chatGPT"

gr.ChatInterface(
    fn=llm_function,
    title=title,
).launch(
    # share=True
)
