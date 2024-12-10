import streamlit as st
import asyncio
from segmenter import Segmenter  
import matplotlib.pyplot as plt

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

# Main Streamlit app
def main():
    st.title("🌎 Political Ideology Analyzer")
    st.write("Analyze the economic and social ideological leanings of your text.")

    # Initialize session state for the selected example
    if "example_choice" not in st.session_state:
        st.session_state.example_choice = ""

    # Sidebar with example buttons
    with st.sidebar:
        st.header("Examples")
        st.write("Click a button to fill the text box with an example:")
        if st.button("Communist Example"):
            st.session_state.example_choice = "Communist"
        if st.button("Fascist Example"):
            st.session_state.example_choice = "Fascist"
        if st.button("Extreme Example"):
            st.session_state.example_choice = "Extreme Example"

    # Input text box and context display
    example = examples.get(st.session_state.example_choice, {"text": "", "context": ""})
    input_text = st.text_area(
        "Enter text for analysis (limit to ~10 sentences):",
        value=example["text"],
        height=100,
        max_chars=2000,
        help="Enter a text of approximately 10 sentences for analysis or select an example from the sidebar.",
    )
    if example["context"]:
        st.text_area("Context for the text:", value=example["context"], height=100, disabled=True)

    # Process text when the button is clicked
    if st.button("Analyze"):
        if not input_text.strip():
            st.error("Please enter some text for analysis.")
        else:
            with st.spinner("Processing..."):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                segmenter = loop.run_until_complete(process_text(example["context"]))
            
            # Display the results
            st.success("Analysis Complete!")

            # Display economic and social scores for each sentence
            st.subheader("Sentence Analysis")
            scores = []
            for i, data in enumerate(segmenter.context):
                sentence = data["sentence"]
                econ_score, social_score, probs = Segmenter._embed(
                    sentence=sentence,
                    context=data["similarity_context"],
                )
                scores.append((sentence, econ_score, social_score))
                st.write(f"**Sentence {i + 1}:** {sentence}")
                st.write(f"  - **Economic Score:** {econ_score:.2f}")
                st.write(f"  - **Social Score:** {social_score:.2f}")
                for cat, prob in probs:
                    st.write(f"  - **{cat}:** {prob:.2f}")

            # Display overall ideological embedding visualization
            st.subheader("Ideological Embedding Visualization")
            fig = plot_embedding(segmenter)
            st.pyplot(fig)

if __name__ == "__main__":
    main()
