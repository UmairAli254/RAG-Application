import streamlit as st
from PyPDF2 import PdfReader

from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.callbacks import get_openai_callback
from streamlit_chat import message
# from langchain.schema import SystemMessage
from langchain_core.prompts import PromptTemplate
# from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
# from langchain.chains import retrieval_qa


st.set_page_config(page_title="RAG (Retrieval Augmented Generation)")
st.header("RAG (Retrieval Augmented Generation)")

openai_secret_key = st.secrets["OPENAI_KEY"]


def main():

 if "conversation" not in st.session_state:
  st.session_state["conversation"] = None
 if "chat_history" not in st.session_state:
  st.session_state["chat_history"] = None
 if "processComplete" not in st.session_state:
  st.session_state["processComplete"] = None

 with st.sidebar:
  uploaded_file = st.file_uploader(
      "Upload Your File Below: ", type=["pdf"])

  process = st.button("Process Docs")

 if process:
  file_text = pdf_file_into_text(uploaded_file)
  st.write("File is converted into Text :)")

  text_chunks = text_into_chunks(file_text)
  st.write("Text is converted into Chunks")

  vectorized_data = chunks_into_vector(text_chunks)
  st.write("Chunks are Vectorized")

  st.session_state["conversation"] = build_conversation_chain(vectorized_data)
  st.write("The Chain is Built")
  st.session_state["processComplete"] = True

 if st.session_state["processComplete"]:
  # st.write(st.session_state["conversation"])
  user_prompt = st.chat_input("Enter Your Prompt Here...")
  if user_prompt:
   display_conversation(user_prompt)


def pdf_file_into_text(uploaded_file):
 text = ""

 pdf = PdfReader(uploaded_file)
 for one in pdf.pages:
  text += one.extract_text()

 return text


def text_into_chunks(text):

 text_splitter = CharacterTextSplitter(
     separator="\n",
     chunk_size=900,
     chunk_overlap=100,
     length_function=len
 )
 chunks = text_splitter.split_text(text)
 return chunks


def chunks_into_vector(chunks):
 embedddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

 vector_data = FAISS.from_texts(chunks, embedddings)

 return vector_data


def build_conversation_chain(vectorized_data):
 llm = ChatOpenAI(
     # model="gpt-3.5-turbo",
     openai_api_key=openai_secret_key,
     temperature=0,

 )

 memory = ConversationBufferMemory(
     memory_key="chat_history",
     return_messages=True
 )

 chain = ConversationalRetrievalChain.from_llm(
     llm=llm,
     memory=memory,
     retriever=vectorized_data.as_retriever(),
     # chain_type="stuff"
     # prompt=PromptTemplate(template=""" Only give the answers about the uploaded file if user asks anything outside the file, then politely say that i can only answer about the uploaded docs
     #  """)
     # chain
 )

 return chain


def display_conversation(user_prompt):
 with get_openai_callback() as cb:
  res = st.session_state["conversation"]({"question": user_prompt})

 st.session_state["chat_history"] = res["chat_history"]
 response_container = st.container()

 with response_container:
  for one, messages in enumerate(st.session_state["chat_history"]):
   if (one % 2 == 0):
    message(messages.content, is_user=True, key=str(one))
   else:
    message(messages.content, key=str(one))


main()
