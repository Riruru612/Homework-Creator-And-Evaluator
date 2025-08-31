import streamlit as st
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import tempfile, os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
hf_api = st.secrets["HF_TOKEN"]
st.secrets["HF_TOKEN","GROQ_API"]

# Embeddings & LLM
embeddings = HuggingFaceEmbeddings(model="all-MiniLM-L6-v2")
groq_api_key = st.secrets["GROQ_API"]
llm = ChatGroq(model="openai/gpt-oss-20b", groq_api_key=groq_api_key)

# Streamlit UI
st.title("📘 Class 10 Science Homework Creator & Evaluator")

# Step 1: Upload PDF
st.subheader("Step 1: Upload Study Material (PDF only)")
documents = []

upload_file = st.file_uploader("Upload PDF(s)", type="pdf", accept_multiple_files=True)
if upload_file:
    for up in upload_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(up.getvalue())
            loader = PyPDFLoader(tmp.name)
            documents.extend(loader.load())
        os.remove(tmp.name)

if documents:
    # Split & create vectorstore
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
    splits = text_splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(splits, embeddings)
    retriever = vectorstore.as_retriever()

    # Step 2: Homework generation
    st.subheader("Step 2: Generate Homework Assignment")
    topic = st.text_input("Enter Topic (e.g., Photosynthesis)")
    difficulty = st.selectbox("Select Difficulty", ["Easy", "Medium", "Hard"])
    score = st.number_input("Total Marks", min_value=5, step=5)

    homework_prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are a teacher. Based on the given topic, difficulty, and total marks, "
         "create a homework assignment ONLY containing questions. "
         "Distribute marks across the questions. "
         "Include MCQs, short answers, and fill-in-the-blanks. "
         "Do not provide answers."),
        ("human", "{input}\n\nContext: {context}")
    ])

    homework_chain = create_stuff_documents_chain(llm, homework_prompt)
    rag_homework_chain = create_retrieval_chain(retriever, homework_chain)

    if st.button("Generate Homework"):
        query = f"Generate homework for {topic} at {difficulty} difficulty, total marks {score}."
        res = rag_homework_chain.invoke({"input": query})
        st.session_state["assignment"] = res["answer"]
        st.write("### Generated Homework")
        st.write(res["answer"])

# Step 3: Student Submission
if "assignment" in st.session_state:
    st.subheader("Step 3: Submit Homework")
    student_answer = st.text_area("Write your answers here")

    if st.button("Submit Answers"):
        if student_answer.strip() == "":
            st.error("Please provide your answers in the text box.")
        else:
            st.session_state["submission"] = student_answer
            st.success("✅ Homework submitted successfully!")

# Step 4: Evaluation
if "submission" in st.session_state:
    st.subheader("Step 4: Evaluation")

    eval_prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are a strict but fair teacher. Evaluate the student's answers based on the provided homework questions "
         "and the study context. Give detailed feedback for each question, assign marks for each, and provide a final score."),
        ("human", 
         "Homework Questions:\n{assignment}\n\nStudent Answers:\n{submission}\n\nReference Context:\n{context}")
    ])

    eval_chain = create_stuff_documents_chain(llm, eval_prompt)
    rag_eval_chain = create_retrieval_chain(retriever, eval_chain)

    if st.button("Evaluate Answers"):
        res = rag_eval_chain.invoke({
            "input": f"Evaluate student's homework submission.",
            "assignment": st.session_state["assignment"],
            "submission": st.session_state["submission"]
        })
        st.write("### 📊 Evaluation Report")
        st.write(res["answer"])
