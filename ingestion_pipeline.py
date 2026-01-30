import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
load_dotenv()

def load_documents(docs_path="docs"):
    print("Loading documents from",docs_path)
    #check if the path valid
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exists")
    #loadfiles
    loader=DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader
    )
    documents=loader.load()
    if len(documents)==0:
        raise FileNotFoundError(f"No text files found in {docs_path}.")
    for i, doc in enumerate(documents[:2]):
        print(f"\nDocument{i+1}:")
        print(f" Source:{doc.metadata['source']}")
        print(f" Content Length: {len(doc.page_content)} charecters")
        print(f" Content preview: {doc.page_content[:100]}...")
        print(f" metadata: {doc.metadata}")
    return documents
def split_documents(documents, chunk_size=8000,chunk_overlap=0):
    print("Splitting documets into chunks")
    text_splitter=CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks=text_splitter.split_documents(documents)
    if chunks:
        for i, chunk in enumerate(chunks[:5]):
            print(f"\n---Chunk{i+1}")
            print(f"Source: Chunk.metadata['source']")
            print(f"Length: {len(chunk.page_content)} charecters")
            print(f"Content")
            print(chunk.page_content)
            print("-"*50)
        if len(chunks)>5:
            print(f"\n... and {len(chunks)-5}more chunks")
        return chunks

def create_vector_store(chunks, persist_directory="db/chroma_db"):
    print("Creating embeddings and storing in chromeDB...")
    
    embedding_model=HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Creating vector store")
    vectorstore=Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space":"cosine"}
    )
    print("finished creating vector store")
    print(f"Vector store created and saved to {persist_directory}")

    
    return vectorstore
        
def main():
    print("Hello world")
    #loading file 
    documents=load_documents(docs_path="docs")
    #chunking
    chunks=split_documents(documents)
    #embedding and storing in vector database
    vectorstore=create_vector_store(chunks)



if __name__=="__main__":
    main()
    
