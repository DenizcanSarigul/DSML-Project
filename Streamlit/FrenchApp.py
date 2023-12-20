# app.py

import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Load the Camembert model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained('model')
tokenizer = AutoTokenizer.from_pretrained("camembert-base")

# Streamlit app layout configuration
st.set_page_config(
    page_title="Text Difficulty Analyser and Reading Suggestions",
    page_icon="📚",
    layout="wide",
)

# Custom HTML for background image
background_html = """
<style>
body {
    background-image: url("https://media.istockphoto.com/id/1209873889/de/foto/weinberg-saint-emilion-unter-der-herbstsonne.jpg?s=1024x1024&w=is&k=20&c=5CRoVOrksviM3yWCB4EfBem0bRXALKWK8VRNa0fWx4I=");
    background-size: cover;
    background-repeat: no-repeat;
}
</style>
"""

# Streamlit app layout
def main():
    # Inject custom HTML for background image
    st.markdown(background_html, unsafe_allow_html=True)

    st.title("Text Difficulty Analyser and Reading Suggestions")

    # User input: text area with dynamic height
    user_text = st.text_area("Enter your text here:", height=get_text_area_height())

    # Button to trigger prediction
    if st.button("Analyze Text"):
        # Get model prediction
        difficulty = analyze_text(user_text, model, tokenizer)
        st.write(f"The difficulty of the text is: {difficulty}")

        # Display reading suggestions based on difficulty level
        display_reading_suggestions(difficulty)

def get_text_area_height():
    # Get the length of the input text and set a dynamic height
    text_length = len(st.session_state.user_text) if hasattr(st.session_state, 'user_text') else 0
    max_height = 400  # Set a maximum height
    return min(max(100, text_length * 10), max_height)  # Adjust the multiplier as needed

def analyze_text(text, model, tokenizer):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256)

    # Make the prediction
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the predicted difficulty level
    logits = outputs.logits
    softmax = torch.nn.Softmax(dim=-1)
    probabilities = softmax(logits.squeeze().cpu())
    predicted_class = torch.argmax(probabilities).item()

    # Map the predicted class to difficulty level
    label_mapping = {0: 'A1', 1: 'A2', 2: 'B1', 3: 'B2', 4: 'C1', 5: 'C2'}
    difficulty = label_mapping[predicted_class]

    return difficulty

def display_reading_suggestions(difficulty):
    st.subheader("Reading Suggestions")

    # Placeholder for reading suggestions
    if difficulty == 'A1':
        st.write("Reading suggestions for A1 level:")
    elif difficulty == 'B1':
        st.write("Reading suggestions for B1 level:")
        # Display B1 reading suggestions
    elif difficulty == 'B2':
        st.write("Reading suggestions for B2 level:")
        # Display B2 reading suggestions
    elif difficulty == 'C1':
        st.write("Reading suggestions for C1 level:")
        # Display C1 reading suggestions
    elif difficulty == 'C2':
        st.write("Reading suggestions for C2 level:")
        # Display C2 reading suggestions

if __name__ == "__main__":
    main()
