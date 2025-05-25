# 📚 Semantic Book Recommender

## 🖼️ App Screenshot
![image](https://github.com/user-attachments/assets/8a652305-6998-4e37-9d83-5571ed819414)


This project is a semantic search-based book recommendation system built using [LangChain](https://www.langchain.com/), [Hugging Face Transformers](https://huggingface.co/), and [Gradio](https://gradio.app/). It allows users to input a book description and receive recommendations for similar books based on vector similarity.

> 🔗 Based on the [FreeCodeCamp tutorial](https://www.youtube.com/watch?v=i_23KUAEtUM) but modified to use Hugging Face embeddings for cost-free inference.

## 💡 Key Features

- **Semantic Search** using sentence-transformer embeddings  
- **Embeddings**: Uses `sentence-transformers/all-MiniLM-L6-v2` from Hugging Face  
- **Local Embedding Indexing** with FAISS  
- **Simple and Clean UI** built using Gradio  
- **No OpenAI API Key Needed** (unlike the original project)

## 🚀 How It Works

1. Loads a dataset of books with their descriptions.
2. Converts book descriptions into vector embeddings using Hugging Face models.
3. User inputs a book description or theme.
4. The system finds and recommends semantically similar books.

## 🛠 Tech Stack

- Python  
- [LangChain](https://www.langchain.com/)  
- [Hugging Face Transformers](https://huggingface.co/)  
- [FAISS](https://github.com/facebookresearch/faiss)  
- [Gradio](https://gradio.app/)

## 📝 Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/semantic-book-recommender
   cd semantic-book-recommender
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app**
   ```bash
   python gradio_dashboard.py
   ```

## 🧠 Model Used

- [`sentence-transformers/all-MiniLM-]()


