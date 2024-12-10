import streamlit as st
import asyncio
from segmenter import Segmenter  
import matplotlib.pyplot as plt
import numpy as np

async def process_text(corpus):
    segmenter = Segmenter(corpus)
    await segmenter.initialize()
    return segmenter

def plot_embedding(segmenter):
    median = segmenter._embed_corpus()
    points = segmenter.points

    plt.figure(figsize=(10, 10))
    plt.style.use('ggplot')
    quadrant_colors = {
        'top_left': '#FFB3BA',      # Soft Pink (Liberal)
        'top_right': '#BAFFC9',     # Soft Green (Libertarian)
        'bottom_left': '#BAE1FF',   # Soft Blue (Socialist)
        'bottom_right': '#FFE9BA'   # Soft Yellow (Conservative)
    }

    # Fill quadrants with colors
    plt.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    plt.axvline(x=0, color='gray', linestyle='--', linewidth=0.5)

    plt.fill_between([-10, 0], 0, 10, color=quadrant_colors['top_left'], alpha=0.3)
    plt.fill_between([0, 10], 0, 10, color=quadrant_colors['top_right'], alpha=0.3)
    plt.fill_between([-10, 0], -10, 0, color=quadrant_colors['bottom_left'], alpha=0.3)
    plt.fill_between([0, 10], -10, 0, color=quadrant_colors['bottom_right'], alpha=0.3)

    # Add labels to quadrants
    plt.text(-5, 9, 'AuthLeft', horizontalalignment='center', fontsize=10)
    plt.text(5, 9, 'AuthRight', horizontalalignment='center', fontsize=10)
    plt.text(-5, -1, 'LibLeft', horizontalalignment='center', fontsize=10)
    plt.text(5, -1, 'LibRight', horizontalalignment='center', fontsize=10)

    # Plot individual points
    econ_points = [point[0] for point in points]
    social_points = [point[1] for point in points]
    plt.scatter(econ_points, social_points, c='blue', alpha=0.6, edgecolors='black', linewidth=0.5)

    # Plot geometric median
    if median is not None:
        mx, my = median
        plt.scatter(mx, my, c='red', s=200, marker='*', edgecolors='black', linewidth=1, label='Corpus Median')

    plt.title('Political Ideology Embedding', fontsize=15)
    plt.xlabel('Economic Axis (Left ← → Right)', fontsize=12)
    plt.ylabel('Social Axis (Liberal ↑ → Conservative ↓)', fontsize=12)
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()

    # Return the figure to render in Streamlit
    return plt.gcf()

# Prefilled text examples with contextual surroundings
examples = {
    "Communist": {
        "text": "The means of production must be collectively owned.",
        "context": (
            "In a society dominated by capitalist structures, inequality has reached an unbearable peak. "
            "The means of production must be collectively owned. Only through collective ownership can workers "
            "ensure their freedom and well-being."
        )
    },
    "Fascist": {
        "text": "The state must enforce strict social order and unity.",
        "context": (
            "Amid growing societal chaos and disintegration, there is only one way forward. "
            "The state must enforce strict social order and unity. This will ensure stability and national greatness."
        )
    },
    "Extreme Example": {
        "text": "We must eradicate all enemies of the state.",
        "context": (
            "In times of war, compromise is not an option. The survival of our nation depends on radical actions. "
            "We must eradicate all enemies of the state. Only then can our society achieve purity and strength."
        )
    }
}

def get_quadrant_description(econ_score, social_score):
    """Determine the political quadrant based on economic and social scores."""
    if econ_score >= 0 and social_score >= 0:
        return "AuthRight", "Conservative, free-market oriented"
    elif econ_score < 0 and social_score >= 0:
        return "AuthLeft", "Statist, collectivist"
    elif econ_score < 0 and social_score < 0:
        return "LibLeft", "Libertarian, socially progressive"
    else:
        return "LibRight", "Libertarian, free-market oriented"

def create_sentence_summary_card(sentence, econ_score, social_score, probs):
    """Create a detailed summary card for a sentence."""
    quadrant, description = get_quadrant_description(econ_score, social_score)
    
    # Determine the most likely category
    top_category = max(probs, key=lambda x: x[1])[0] if probs else "Uncategorized"
    
    # Color-code the economic and social scores
    econ_color = 'green' if econ_score >= 0 else 'red'
    social_color = 'blue' if social_score >= 0 else 'orange'
    
    # Create the card
    st.markdown(f"""
    <div style="
        background-color: #f0f2f6;
        border-left: 5px solid {econ_color};
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
        color: #333333;
    ">
    <h4 style="margin-top: 0; color: #333333;">📝 Sentence Analysis</h4>
    <p style="color: #333333;"><strong style="color: #333333;">Sentence:</strong> {sentence}</p>
    <div style="display: flex; justify-content: space-between; color: #333333;">
        <div>
            <p><strong style="color: {econ_color};">Economic Score:</strong> {econ_score:.2f}</p>
            <p><strong style="color: {social_color};">Social Score:</strong> {social_score:.2f}</p>
        </div>
        <div>
            <p><strong style="color: #333333;">Quadrant:</strong> {quadrant}</p>
            <p><small style="color: #333333;">{description}</small></p>
        </div>
    </div>
    <div style="background-color: #e9ecef; padding: 10px; border-radius: 5px; margin-top: 10px; color: #333333;">
        <strong style="color: #333333;">Categorization Probabilities:</strong>
        {''.join(f'<br>• <span style="color: #333333;">{cat}:</span> {prob:.2f}' for cat, prob in probs)}
    </div>
    </div>
    """, unsafe_allow_html=True)

# Main Streamlit app
def main():
    st.set_page_config(page_title="Political Ideology Analyzer", page_icon="🌎", layout="wide")
    
    st.title("🌎 Political Ideology Analyzer")
    st.write("Analyze the economic and social ideological leanings of your text.")

    # Initialize session state for the selected example and custom text
    if "example_choice" not in st.session_state:
        st.session_state.example_choice = ""
    if "custom_text" not in st.session_state:
        st.session_state.custom_text = ""

    # Sidebar with example buttons
    with st.sidebar:
        st.header("Examples")
        st.write("Click a button to fill the text box with an example:")
        if st.button("Communist Example"):
            st.session_state.example_choice = "Communist"
            st.session_state.custom_text = examples["Communist"]["text"]
        if st.button("Fascist Example"):
            st.session_state.example_choice = "Fascist"
            st.session_state.custom_text = examples["Fascist"]["text"]
        if st.button("Extreme Example"):
            st.session_state.example_choice = "Extreme Example"
            st.session_state.custom_text = examples["Extreme Example"]["text"]

    # Input text box and context display
    example = examples.get(st.session_state.example_choice, {"text": "", "context": ""})
    
    # Allow editing of the input text
    st.session_state.custom_text = st.text_area(
        "Enter text for analysis (limit to ~10 sentences):",
        value=st.session_state.custom_text or example["text"],
        height=100,
        max_chars=2000,
        key="input_text",
        help="Enter a text of approximately 10 sentences for analysis or select an example from the sidebar.",
    )

    # Display context if an example is selected
    if example["context"]:
        st.text_area("Context for the text:", value=example["context"], height=100, disabled=True)

    # Process text when the button is clicked
    if st.button("Analyze"):
        input_text = st.session_state.custom_text
        if not input_text.strip():
            st.error("Please enter some text for analysis.")
        else:
            with st.spinner("Processing..."):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                context = example["context"] if st.session_state.example_choice else ""
                segmenter = loop.run_until_complete(process_text(input_text + "\n" + context))
            
            # Create two columns for results
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Display economic and social scores for each sentence
                st.subheader("📊 Sentence-Level Ideology Analysis")
                scores = []
                for i, data in enumerate(segmenter.context):
                    sentence = data["sentence"]
                    econ_score, social_score, probs = Segmenter._embed(
                        sentence=sentence,
                        context=data["similarity_context"],
                    )
                    scores.append((sentence, econ_score, social_score))
                    create_sentence_summary_card(sentence, econ_score, social_score, probs)

            with col2:
                # Display overall ideological embedding visualization
                st.subheader("🗺️ Corpus Embedding")
                fig = plot_embedding(segmenter)
                st.pyplot(fig)

                # Calculate and display overall corpus ideology
                if scores:
                    avg_econ = np.mean([s[1] for s in scores])
                    avg_social = np.mean([s[2] for s in scores])
                    quadrant, description = get_quadrant_description(avg_econ, avg_social)
                    
                    st.markdown(f"""
                    <div style="
                        background-color: #f0f2f6;
                        border-left: 5px solid #4a4a4a;
                        padding: 15px;
                        border-radius: 10px;
                        margin-top: 10px;
                        color: #333333;
                    ">
                    <h4 style="margin-top: 0; color: #333333;">📍 Corpus Ideology Overview</h4>
                    <p style="color: #333333;"><strong style="color: #333333;">Average Economic Score:</strong> {avg_econ:.2f}</p>
                    <p style="color: #333333;"><strong style="color: #333333;">Average Social Score:</strong> {avg_social:.2f}</p>
                    <p style="color: #333333;"><strong style="color: #333333;">Dominant Quadrant:</strong> {quadrant}</p>
                    <p style="color: #333333;"><small>{description}</small></p>
                    </div>
                    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()