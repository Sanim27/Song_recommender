from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import streamlit as st
import random
dataset=pd.read_csv("song_data.csv")
def format_data(row):
    formatted=f"""
    artist:{row['artists']},
    album:{row['album_name']},
    track:{row['track_name']},
    popularity:{row['popularity']},
    duration:{row['duration_ms']},
    explicit_words:{row['explicit']},
    speechiness:{row['speechiness']},
    danceability:{row['danceability']},
    energy:{row['energy']},
    key:{row['key']},
    loudness:{row['loudness']},
    acousticness:{row['acousticness']},
    instrumentalness:{row['instrumentalness']},
    liveness:{row['liveness']},
    valence:{row['valence']},
    tempo:{row['tempo']},
    genre:{row['track_genre']}
    """
    return formatted

text_representation=dataset.apply(format_data,axis=1)

text_list = text_representation.tolist()

documents = [Document(page_content=text) for text in text_list]

st.title("Song Recommender")

prompt=ChatPromptTemplate.from_template(
    """
    Show the top 5 songs based on the provided context and the user Questions only, if you cant suggest the best songs say the user to do web search.And just suggest the songs in a listed form and do nothing else other than that like giving decriptions.
    Please provide the most accurate response based on the songs that user might like according to their question and the context below.
    <context>
    {context}
    </context>
    Questions:{input}
    """
)
huggingfaceapi="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"
groq_api_key="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"

llm=ChatGroq(groq_api_key=groq_api_key,
model_name="llama3-70b-8192"
)

def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = HuggingFaceBgeEmbeddings(
            model_name="BAAI/llm-embedder",
            model_kwargs={"device": "cpu"},
            encode_kwargs={'normalize_embeddings': True}
        )
        st.session_state.docs = documents
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.limited_documents= st.session_state.final_documents[:10000]
        st.session_state.vector = FAISS.from_documents(st.session_state.limited_documents, st.session_state.embeddings)

prompt1=st.text_input("Type of song you would like to listen to ?")
if st.button("Documents Embedding"):
    vector_embedding()
    st.write("Vector Store DB is ready")

if st.button("Get Recommendations"):
    document_chain=create_stuff_documents_chain(llm,prompt)
    retriever=st.session_state.vector.as_retriever()
    retrieval_chain=create_retrieval_chain(retriever,document_chain)
    response=retrieval_chain.invoke({'input':prompt1})
    st.write(response['answer'])

    with st.expander("Document Similarity Search"):
        #find the relevant chunks
        for i,doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("-----------------------------------")