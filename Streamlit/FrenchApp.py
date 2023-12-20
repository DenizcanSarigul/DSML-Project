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


# Streamlit app layout
def main():

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

        book_links_A1 = {
            "Merde, It's Not Easy to Learn French": "https://www.amazon.com/Merde-Its-Easy-Learn-French/dp/1795292733?crid=BBJ0F0HCMF82&keywords=merde+its+not+easy+to+learn+french&qid=1639658757&sprefix=merde+it%27s+not+easy,aps,224&sr=8-3&linkCode=sl1&tag=relearnalangu-20&linkId=46371e44e1932ab005474112eb8b2496&language=en_US&ref_=as_li_ss_tl",
            "La planète grise": "https://www.thecibookshop.com/en/la-planete-grise.html",
            "Semi Professionel": "https://www.thecibookshop.com/en/semi-professionnel-tome-1.html"
        }

        for title, link in book_links_A1.items():
            st.markdown(f"[{title}]({link})")
            
    elif difficulty == 'A2':
        st.write("Reading suggestions for A2 level:")
        
        book_links_A2 = {
            "Mystère à Nice" : "https://www.blackcat-cideb.com/en/books/mystere-a-nice-en",
            "Jeanne D'Arc": "https://www.blackcat-cideb.com/en/books/jeanne-d-arc-en",
            "Aventure dans les Pyrénées": "https://www.blackcat-cideb.com/en/books/aventure-dans-les-pyrenees-en"
        }

        for title, link in book_links_A2.items():
            st.markdown(f"[{title}]({link})")

    elif difficulty == 'B1':
        st.write("Reading suggestions for B1 level:")
        
        book_links_B1 = {"Le tour du monde en 80 jours" : "https://www.blackcat-cideb.com/en/books/le-tour-du-monde-en-80-jours-5",
                        "Voyage au centre de la Terre" : "https://www.blackcat-cideb.com/en/books/voyage-au-centre-de-la-terre-en",
                        "Enquête à Saint-Malo": "https://www.blackcat-cideb.com/en/books/enquete-a-saint-malo-en"
        }

        for title, link in book_links_B1.items():
            st.markdown(f"[{title}]({link})")

    elif difficulty == 'B2':
        st.write("Reading suggestions for B2 level:")
        
        book_links_B2 = {
            "Vengeance à La Réunion" : "https://www.blackcat-cideb.com/en/books/vengeance-a-la-reunion-en",
            "Double assassinat dans la Rue Morgue et La lettre volée" : "https://www.blackcat-cideb.com/en/books/double-assassinat-dans-la-rue-morgue-et-la-lettre-volee-en",
            "Le Mystère de la Chambre Jaune" : "https://www.blackcat-cideb.com/en/books/mystere-de-la-chambre-jaune-le-en"
        }

        for title, link in book_links_B2.items():
            st.markdown(f"[{title}]({link})")

    elif difficulty == 'C1':
        st.write("Reading suggestions for C1 level:")

        book_links_C1 = {"L'Île mystérieuse" : "https://anylang.net/en/books/fr/mysterious-island",
                         "Les Trois Mousquetaires" : "https://anylang.net/en/books/fr/three-musketeers",
                         "Pilote de guerre": "https://anylang.net/en/books/fr/flight-arras"}
        
        for title, link in book_links_C1.items():
            st.markdown(f"[{title}]({link})")

    elif difficulty == 'C2':
        st.write("Reading suggestions for C2 level:")

        book_links_C2 = {"Cyrano de Bergerac" : "https://anylang.net/en/books/fr/cyrano-de-bergerac",
                         "A rebours" : "«Against Nature» in French with a Parallel Translation",
                         "Les Fleurs du Mal" :  "https://anylang.net/en/books/fr/flowers-evil"

        }

        for title, link in book_links_C2.items():
            st.markdown(f"[{title}]({link})")

if __name__ == "__main__":
    main()

    