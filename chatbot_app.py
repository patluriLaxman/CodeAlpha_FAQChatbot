import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import spacy
import string
import logging

# --- Load SpaCy Model ---
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    st.error("SpaCy model not found. Run `python -m spacy download en_core_web_sm`.")
    st.stop()

# --- Logging Setup (Optional) ---
logging.basicConfig(filename="faq_chatbot_log.txt", level=logging.INFO)

# --- Text Preprocessing ---
def preprocess_text_spacy(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_punct and not token.is_space and not token.is_stop]
    return " ".join(tokens)

# --- Load FAQ Data ---
@st.cache_data
def load_faqs():
    try:
        with open('faqs.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        for faq in data:
            faq['processed_question'] = preprocess_text_spacy(faq['question'])
        return data
    except FileNotFoundError:
        st.error("File `faqs.json` not found.")
        st.stop()

faqs_data = load_faqs()
corpus = [faq['processed_question'] for faq in faqs_data]

# --- Load Vectorizer & Vectors ---
@st.cache_resource
def load_vectorizer_and_vectors(corpus):
    vectorizer = TfidfVectorizer(max_features=5000)
    faq_vectors = vectorizer.fit_transform(corpus)
    return vectorizer, faq_vectors

vectorizer, faq_vectors = load_vectorizer_and_vectors(corpus)

# --- Match Function ---
def find_best_match(user_query, faqs_data, vectorizer, faq_vectors, threshold=0.6):
    processed_user_query = preprocess_text_spacy(user_query)

    if not processed_user_query.strip():
        return "Your question seems empty or unclear. Please rephrase.", 0.0

    user_query_vector = vectorizer.transform([processed_user_query])
    similarities = cosine_similarity(user_query_vector, faq_vectors).flatten()

    most_similar_index = similarities.argmax()
    highest_similarity = similarities[most_similar_index]

    if highest_similarity >= threshold:
        matched_question = faqs_data[most_similar_index]['question']
        matched_answer = faqs_data[most_similar_index]['answer']
        logging.info(f"Match: '{user_query}' → '{matched_question}' (Score: {highest_similarity:.2f})")
        return matched_answer, highest_similarity
    else:
        top_indices = similarities.argsort()[-3:][::-1]
        suggestions = "\n".join(f"- {faqs_data[i]['question']}" for i in top_indices)
        logging.info(f"No match: '{user_query}' → Top Suggestions (Max Score: {highest_similarity:.2f})")
        return (
            f"🤔 I couldn't find a perfect match. You could try asking:\n\n{suggestions}",
            highest_similarity
        )

# --- Streamlit UI ---
st.set_page_config(page_title="📚 FAQ Chatbot", layout="centered")
st.title("🤖 Product FAQ Chatbot")
st.markdown("Ask me anything about our product or services.")

# View all FAQs
with st.expander("📋 View All FAQs"):
    for faq in faqs_data:
        st.markdown(f"**Q:** {faq['question']}\n\n**A:** {faq['answer']}\n---")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Input
if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Finding the best answer..."):
            response, similarity = find_best_match(prompt, faqs_data, vectorizer, faq_vectors)
            st.markdown(response)
            st.caption(f"🧠 Confidence Score: `{similarity:.2f}`")

    st.session_state.messages.append({"role": "assistant", "content": response})
