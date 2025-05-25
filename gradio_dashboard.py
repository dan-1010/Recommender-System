import pandas as pd
import numpy as np
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


import gradio as gr

load_dotenv()






# Choose a high-quality Hugging Face embedding model
model_name = 'sentence-transformers/all-MiniLM-L6-v2'  # You can also try 'BAAI/bge-small-en-v1.5' for even better performance

# Initialize the embedding model
hf_embeddings = HuggingFaceEmbeddings(model_name=model_name)


books = pd.read_csv("books_with_emotions.csv")
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover-not-found.jpg",
    books["large_thumbnail"],
)


raw_documents = TextLoader("tagged_description.txt",encoding="utf-8").load()
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=0, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)
db_books = Chroma.from_documents(documents, hf_embeddings)


def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16,
) -> pd.DataFrame:
    
    """
        Retrieve book recommendations based on semantic similarity, category, and tone preferences.
        Args:
            query (str): The search query or description to find semantically similar books.
            category (str, optional): The book category to filter recommendations. Defaults to None. If "All", no category filtering is applied.
            tone (str, optional): The desired emotional tone of the recommendations. Can be one of "Happy", "Surprising", "Angry", "Suspenseful", or "Sad". Defaults to None.
            initial_top_k (int, optional): The number of top similar books to retrieve initially. Defaults to 50.
            final_top_k (int, optional): The final number of recommendations to return. Defaults to 16.
        Returns:
            pd.DataFrame: A DataFrame containing the recommended books filtered and sorted according to the specified criteria.
        """
    
    

    recs = db_books.similarity_search(query, k=initial_top_k)
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    book_recs = books[books["isbn13"].isin(books_list)].head(initial_top_k)

    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    if tone == "Happy":
        book_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprising":
        book_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":
        book_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        book_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        book_recs.sort_values(by="sadness", ascending=False, inplace=True)

    return book_recs


def recommend_books(
        query: str,
        category: str,
        tone: str
):
    """
    Generates a list of recommended books based on a user query, category, and tone.
    Args:
        query (str): The user's search query or interest.
        category (str): The book category or genre to filter recommendations.
        tone (str): The desired tone or mood for the recommendations.
    Returns:
        list of tuple: A list of tuples, each containing:
            - large_thumbnail (str): URL or path to the book's large thumbnail image.
            - caption (str): A formatted string with the book's title, authors, and a truncated description.
    Notes:
        - The function retrieves recommendations using `retrieve_semantic_recommendations`.
        - Book descriptions are truncated to the first 30 words.
        - Author names are formatted for readability.
    """
    
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."

        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))
    return results

categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks(theme = gr.themes.Glass()) as dashboard:
    gr.Markdown("# Semantic book recommender")

    with gr.Row():
        user_query = gr.Textbox(label = "Please enter a description of a book:",
                                placeholder = "e.g., A story about forgiveness")
        category_dropdown = gr.Dropdown(choices = categories, label = "Select a category:", value = "All")
        tone_dropdown = gr.Dropdown(choices = tones, label = "Select an emotional tone:", value = "All")
        submit_button = gr.Button("Find recommendations")

    gr.Markdown("## Recommendations")
    output = gr.Gallery(label = "Recommended books", columns = 8, rows = 2)

    submit_button.click(fn = recommend_books,
                        inputs = [user_query, category_dropdown, tone_dropdown],
                        outputs = output)


if __name__ == "__main__":
    dashboard.launch()