import pickle
import streamlit as st
import numpy as np

# Set Streamlit page configuration
st.set_page_config(page_title="Book Recommender", layout="wide")

# Custom CSS for minimal design
st.markdown("""
    <style>
        .main {
            background-color: #f5f5f5;
        }
        .stTextInput>div>div>input {
            border-radius: 10px;
            padding: 5px;
            font-size: 14px;
            width: 200px;
            float: right;
        }
        .stButton>button {
            border-radius: 10px;
            padding: 10px;
            font-size: 16px;
        }
        .stImage>img {
            border-radius: 10px;
            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
        }
        .stSelectbox>div {
            font-size: 16px;
        }
        .centered {
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# Load trained model and datasets
st.title("Book Recommendation System", anchor=False)

model = pickle.load(open("model.pkl", 'rb'))
book_names = pickle.load(open("book_names.pkl", 'rb'))
Rating = pickle.load(open("Rating.pkl", 'rb'))
book_pivot = pickle.load(open("book_pivot.pkl", 'rb'))

# Function to fetch book cover images
def fetch_poster(suggestion):
    poster_url = []
    for book_id in suggestion:
        book_title = book_pivot.index[book_id]
        idx = np.where(Rating['Book-Title'].str.lower() == book_title.lower())[0][0]
        poster_url.append(Rating.iloc[idx]['Image-URL-L'])
    return poster_url

# Function to get book recommendations
def recommend_book(book_name):
    book_id = np.where(book_pivot.index.str.lower() == book_name.lower())[0][0]
    _, suggestion = model.kneighbors(book_pivot.iloc[book_id, :].values.reshape(1, -1), n_neighbors=6)
    poster_url = fetch_poster(suggestion[0])
    recommended_books = [book_pivot.index[i] for i in suggestion[0]]
    return recommended_books, poster_url

# Streamlit UI
st.write("<div class='centered'>Search for book recommendations!</div>", unsafe_allow_html=True)

# Search bar in the top right
col1, col2 = st.columns([3, 1])
with col2:
    target_book = st.text_input("", "", placeholder="Search a book...")

# Filter the books based on the search input
if target_book:
    filtered_books = [book for book in book_names if target_book.lower() in book.lower()]
else:
    filtered_books = book_names

# Dropdown select box for filtered books
selected_book = st.selectbox("Choose from the list", filtered_books)

if st.button('Show Recommendations') or target_book:
    book_to_search = selected_book
    if book_to_search:
        recommended_books, poster_url = recommend_book(book_to_search)
        cols = st.columns(5)
        for i in range(1, 6):  # Display top 5 recommended books
            with cols[i-1]:
                st.text(recommended_books[i])
                st.image(poster_url[i], use_column_width=True)
