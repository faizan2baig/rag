import torch
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
from dotenv import load_dotenv

load_dotenv()
persistent_directory="db/chroma_db"
embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

db=Chroma(
    persist_directory = persistent_directory,
    embedding_function = embedding_model
)
query=input("Enter Your Query...")

retriever=db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k":5, "score_threshold":0.3}
    
)
relevent_docs=retriever.invoke(query)
context_text=""
for i, doc in enumerate (relevent_docs, 1):
    print(context_text)
    context_text+=f"Document {i}: {doc.page_content}\n"

model_id="microsoft/Phi-3.5-mini-instruct"

model=AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=False
)
model.config.use_cache=False
tokenizer=AutoTokenizer.from_pretrained(model_id)
prompt = f"<|user|>\nBased on the following documents, answer this question: {query}\n\nDocuments:\n{context_text}\n<|end|>\n<|assistant|>\n, otherwise just give answer that you don't have it."

pipe=pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=500,
    temperature=0.1,
    return_full_text=False
    
    
)
hf_llm=HuggingFacePipeline(pipeline=pipe)
print("Generating Answer")
answer=hf_llm.invoke(prompt)
print(f"\nFinal Answer:\n{answer}")

      