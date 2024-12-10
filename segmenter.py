import spacy
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from constants import category_ideology_mapping
from openai import  AsyncOpenAI
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import asyncio
from dotenv import load_dotenv

load_dotenv()  # This will load the environment variables from the .env file

model = AutoModelForSequenceClassification.from_pretrained("manifesto-project/manifestoberta-xlm-roberta-56policy-topics-context-2024-1-1", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")

class Segmenter:
    def __init__(self, corpus, language_model="en_core_web_sm"):
        self.corpus = corpus
        self.nlp = spacy.load(language_model)
        self.client = AsyncOpenAI()
        self._load_sentences()
       
        
    async def initialize(self):
        await self._calculate_openai_embeddings()
        self._construct_similarity_context()

    async def _calculate_openai_embeddings(self):
        self.sentence_embeddings = []
        tasks = [
            self._fetch_embedding(sentence) for sentence in self.sentences
        ]
        results = await asyncio.gather(*tasks)
        self.sentence_embeddings.extend(results)

    async def _fetch_embedding(self, sentence):
        """Helper function to fetch embedding for a single sentence."""
        response = await self.client.embeddings.create(
            model="text-embedding-3-small",
            input=sentence
        )
        return response.data[0].embedding

    def _construct_similarity_context(self):
        """Construct context by finding closest sentences in embedding space."""
        similarity_context_data = []
        num_sentences = len(self.sentences)

        # Compute cosine similarity matrix
        similarity_matrix = cosine_similarity(self.sentence_embeddings)

        for i, sentence in enumerate(self.sentences):
            # Find indices of sentences closest to the current sentence
            similar_indices = similarity_matrix[i].argsort()[::-1]  # Sort by similarity descending
            context = ""
            char_count = len(sentence)
            
            for j in similar_indices:
                if i == j or char_count + len(self.sentences[j]) > 300:
                    continue
                context += self.sentences[j] + " "
                char_count += len(self.sentences[j])
                if char_count >= 300:
                    break
            
            similarity_context_data.append({
                'sentence': sentence,
                'similarity_context': context.strip()
            })
        
        self.context = similarity_context_data
        
    def _load_sentences(self):
        doc = self.nlp(self.corpus)
        self.sentences = [sent.text.strip() for sent in doc.sents]
        #self.filter_sentences()
             
    def _construct_greedy_context(self):
        greedy_context_data = []
        num_sentences = len(self.sentences)

        for i, sentence in enumerate(self.sentences):
            # Initialize context
            greedy_context = ""
            char_count = 0

            # Try to get the next sentences
            for j in range(i + 1, num_sentences):
                if char_count + len(self.sentences[j]) <= 300:
                    greedy_context += self.sentences[j] + " "
                    char_count += len(self.sentences[j])
                else:
                    break

            # If no sentences in the next context, try to get the previous sentences
            if not greedy_context.strip():
                char_count = 0
                for j in range(i - 1, -1, -1):
                    if char_count + len(self.sentences[j]) <= 300:
                        greedy_context = self.sentences[j] + " " + greedy_context
                        char_count += len(self.sentences[j])
                    else:
                        break

            greedy_context_data.append({
                'sentence': sentence,
                'greedy_context': greedy_context.strip()
            })
            
        self.context = greedy_context_data   
    
    @classmethod
    def _infer(cls, sentence, context, top_k=3):
        inputs = tokenizer(sentence, context, return_tensors="pt", max_length=300, padding="max_length", truncation=True)
        logits = model(**inputs).logits
        probabilities = torch.softmax(logits, dim=1).tolist()[0]
        class_probs = {model.config.id2label[i]: probabilities[i]*100 for i in range(len(probabilities))}
        class_probs = dict(sorted(class_probs.items(), key=lambda item: item[1], reverse=True))
        return list(class_probs.items())[:top_k]
    
    @classmethod    
    def _embed(cls, sentence, context, top_k=3):
        probs = Segmenter._infer(
            sentence=sentence,
            context=context,
            top_k=top_k
        )
        econ_score_total, social_score_total = 0, 0
        for cat, prob in probs:
            econ_score, soc_score = category_ideology_mapping[cat] 
            
            econ_score_total += (econ_score *2) * (prob / 100)
            social_score_total += (soc_score *2) * (prob / 100)
        return econ_score_total, social_score_total, probs
    
    
    @classmethod    
    def _embed_no_probs(cls, sentence, context, top_k=3):
        probs = Segmenter._infer(
            sentence=sentence,
            context=context,
            top_k=top_k
        )
        econ_score_total, social_score_total = 0, 0
        for cat, prob in probs:
            econ_score, soc_score = category_ideology_mapping[cat] 
            
            econ_score_total += (econ_score *2) * (prob / 100)
            social_score_total += (soc_score *2) * (prob / 100)
        return econ_score_total, social_score_total
    
    
    
    def _embed_corpus(self, top_k=3):
        self.points = ([
            Segmenter._embed_no_probs(
                sentence=datum.get('sentence'),
                context=datum.get('greedy_context'),
                top_k=top_k
            )
            for datum in self.context
        ])
        
        return Segmenter.geometric_median(
            np.array(self.points)
        )
         
    @classmethod
    def geometric_median(cls, X, eps=1e-5):
        y = np.mean(X, axis=0)
        while True:
            D = np.sqrt(((X - y)**2).sum(axis=1))
            nonzeros = (D != 0)
            if not np.any(nonzeros):
                return y
            w = 1 / D[nonzeros]
            T = (X[nonzeros] * w[:, np.newaxis]).sum(axis=0) / w.sum()
            
            if np.linalg.norm(y - T) < eps:
                return T
            y = T
        
    def plot_sentence(self, at_index=0):
        Segmenter._plot(
            Segmenter._embed(
                sentence=self.context[at_index].get('sentence'),
                context=self.context[at_index].get('greedy_context'),
                top_k=3
            )
        )
    
    def show(self):
        median = self._embed_corpus()
        
        Segmenter._plot(
            self.points,
            median
        )
    
    @classmethod
    def _plot(cls, points, median_point=None):
        plt.figure(figsize=(10, 10))
        plt.style.use('ggplot')  # Switched from 'seaborn' to 'ggplot' for a change of scenery
        # Custom color palette
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
        if isinstance(points, list):
            econ_points = [point[0] for point in points]
            social_points = [point[1] for point in points]
            plt.scatter(econ_points, social_points, c='blue', alpha=0.6, edgecolors='black', linewidth=0.5)
        else:
            plt.scatter(points[0], points[1], c='blue', alpha=0.6, edgecolors='black', linewidth=0.5)

        # Plot geometric median
        if median_point is not None:
            mx, my = median_point
            plt.scatter(mx, my, c='red', s=200, marker='*', edgecolors='black', linewidth=1, label='Corpus Median')

        plt.title('Political Ideology Embedding', fontsize=15)
        plt.xlabel('Economic Axis (Left ← → Right)', fontsize=12)
        plt.ylabel('Social Axis (Liberal ↑ → Conservative ↓)', fontsize=12)
        plt.xlim(-10, 10)
        plt.ylim(-10, 10)
        plt.grid(True, linestyle='--', linewidth=0.5)
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        
async def main():
    corpus = (
        "We must kill all jews!"
    )

    segmenter = Segmenter(corpus)
    await segmenter.initialize()
    segmenter.show()

if __name__ == "__main__":
    asyncio.run(main())
    