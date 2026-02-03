import torch
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM,pipeline
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import time
start=time.time()

persistent_directory="db/chroma_db"
embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db= Chroma(persist_directory=persistent_directory, embedding_function=embeddings)
                             
retriever=db.as_retriever(search_kwargs={"k":3})

model_id="Qwen/Qwen2.5-1.5B-Instruct"
tokenizer=AutoTokenizer.from_pretrained(model_id)
model=AutoModelForCausalLM.from_pretrained(model_id,torch_dtype="auto", device_map="auto")

pipe=pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.1,
    return_full_text=False
)
hf_llm=HuggingFacePipeline(pipeline=pipe)
chat_history=[]
print("---Conversational Rag Started(Type exit to stop)---")
while True:
    query=input("\nYou: ")
    print("Wait...")
    if query.lower() in ["exit"]:
        break
    relevant_docs=retriever.invoke(query)
    context="\n".join([d.page_content for d in relevant_docs])
    history_str="\n".join([d.page_content for d in relevant_docs])
    prompt=f"""<|im_start|>system You are a helpful assistant. Use the context and chat history to answer.If the question does not belong to the document just give answer that this information does not exist. Context: {context}Chat History:{history_str}<|im_end|><|im_start|>user{query}<|im_end|><|im_start|>assistant"""
    response=hf_llm.invoke(prompt)
    print(f"\nAI: {response}")
    end=time.time()
    total=int(end-start)
    if total > 59:
        minutes=total/60
        seconds=total%60
    if minutes > 1:
            print("Time taken for this response is: ",int(minutes), "minutes and ", int(seconds),"seconds")
    else:
             print("Time taken for this response is: ",int(minutes), "minute and ", int(seconds),"seconds")

         
    chat_history.append((query, response))
    if len(chat_history)>3:
        chat_history.pop(0)
    