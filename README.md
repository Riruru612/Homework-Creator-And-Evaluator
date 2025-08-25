📘 Homework Creator & Evaluator

An AI-powered Streamlit application that helps teachers create homework assignments from study materials (PDFs) and evaluate student submissions automatically.

Built using LangChain, FAISS, Hugging Face embeddings, and Groq LLMs.

⸻

🚀 Features
	•	📚 Upload Study Materials (PDFs) – Parse class notes or textbooks.
	•	📝 Homework Generator – Create assignments with MCQs, short answers, and fill-in-the-blanks.
	•	✍️ Student Submission – Students can write their answers inside the app.
	•	📊 Automatic Evaluation – AI grades answers, distributes marks, and gives detailed feedback.
	•	⚡ Powered by RAG (Retrieval-Augmented Generation) for accuracy.

⸻

🛠️ Tech Stack
	•	Streamlit – User Interface
	•	LangChain – Chains & prompts
	•	FAISS – Vector database for retrieval
	•	Hugging Face (MiniLM Embeddings) – Embeddings
	•	Groq LLM – Answer generation & evaluation

⸻

📂 Project Structure
Homework-AI/
│── app.py                # Main Streamlit app
│── requirements.txt      # Python dependencies
│── README.md             # Documentation

⚙️ Installation
	1.	Clone this repository
     git clone git@github.com:Riruru612/Homework-Creator-And-Evaluator.git

  2.	Create a virtual environment (recommended)
      python -m venv venv
      source venv/bin/activate  
  3.	Install dependencies
      pip install -r requirements.txt
  4.	Setup environment variables
      Create a .env file in the root directory:
    	HF_TOKEN=your_huggingface_token
      GROQ_API=your_groq_api_key

▶️ Usage

Run the Streamlit app:
streamlit run app.py
   
