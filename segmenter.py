import spacy
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from constants import category_ideology_mapping
from openai import OpenAI
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import numpy as np

model = AutoModelForSequenceClassification.from_pretrained("manifesto-project/manifestoberta-xlm-roberta-56policy-topics-context-2024-1-1", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")

class Segmenter:
    def __init__(self, corpus, language_model="en_core_web_sm"):
        self.corpus = corpus
        self.nlp = spacy.load(language_model)
        self._load_sentences()
        self._construct_greedy_context()
        
        

  

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
            econ_score_total += econ_score * (prob / 100)
            social_score_total += soc_score * (prob / 100)
        return econ_score_total, social_score_total
    
    def _embed_corpus(self, top_k=3):
        self.points = ([
            Segmenter._embed(
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
        plt.text(-5, 9, 'Liberal\nLeft', horizontalalignment='center', fontsize=10)
        plt.text(5, 9, 'Libertarian\nRight', horizontalalignment='center', fontsize=10)
        plt.text(-5, -1, 'Socialist', horizontalalignment='center', fontsize=10)
        plt.text(5, -1, 'Conservative', horizontalalignment='center', fontsize=10)

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

# Example usage
if __name__ == "__main__":
    corpus = (
    "Today, Bharat is making yet another significant stride towards women empowerment. This day holds special significance for several other reasons as well. Today is the 9th- a number that holds immense auspiciousness in our scriptures. The number 9 is linked to the nine powers of Nav Durga, and during Navratri, we dedicate nine days to the worship of Shakti. This day, too, is dedicated to honouring women. "
    "Friends, "
    "On this very day, 9th December, the first meeting of the Constituent Assembly was held. As the nation celebrates 75 years of the Constitution, this date serves as a reminder to uphold the principles of equality and inclusive development. "
    "Friends, "
    "It is indeed a privilege to be here on this revered land that imparted the knowledge of ethics and religion to the world. The International Gita Jayanti Mahotsav is also taking place in Kurukshetra right now. I bow to this sacred land of the Gita and greet the entire state of Haryana and its patriotic people with a warm Ram Ram. The way Haryana embraced the mantra, 'Ek hai to Safe Hai' (If we are together, we are safe) has set a remarkable example for the entire nation. "
    "Friends, "
    "My bond with Haryana and my affection for this land are no secret. Your immense support and blessings have led to the BJP forming the government here for the third consecutive term. For this, I offer my heartfelt gratitude to every family in Haryana. The new government under Saini Ji has been in office for only a few weeks, yet it is being lauded across the nation. The entire country has witnessed how thousands of young people secured permanent jobs immediately after the formation of this government, without incurring any expenses or requiring any recommendations. The double-engine government here is now working at double the speed. "
    "Friends, "
    "During the elections, the women of Haryana raised the slogan, 'Mhara Haryana, Non-Stop Haryana.' We have adopted this slogan as our resolution. It is with this commitment that I am here today to connect with all of you. Looking around, I see an overwhelming presence of mothers and sisters, which is truly heartening. "
    "Friends, "
    "The Bima Sakhi scheme has just been launched here to create employment opportunities for the women and daughters of the nation. Certificates under the Bima Sakhi initiative have been distributed to daughters here today. I extend my heartfelt congratulations to all the women across the country. "
    "Friends, "
    "A few years ago, I had the honour of launching the ‘Beti Bachao, Beti Padhao’ campaign from Panipat. Its positive impact has been felt not only in Haryana but across the entire country. In Haryana alone, thousands of daughters’ lives have been saved over the past decade. Now, 10 years later, the Bima Sakhi Yojana for our sisters and daughters has been inaugurated from this same land of Panipat. In many ways, Panipat has become a symbol of women empowerment. "
    "Friends, "
    "Bharat is now advancing with the resolve to become a developed nation by 2047. Since 1947, the collective energy of every community and every region has brought Bharat to its current heights. However, to achieve the goal of a Viksit Bharat (developed India) by 2047, we must tap into many new sources of energy. One such source is Eastern India, including the North East. Another critical source of energy is the Nari Shakti—the women power—of our nation. To build a developed Bharat, we require the additional strength of our countless mothers and sisters, whose contributions will be our greatest source of inspiration. Today, women-led self-help groups, Bima Sakhi, Bank Sakhi, and Krishi Sakhi are emerging as vital pillars of a developed Bharat. "
    "Friends, "
    "To empower women, it is crucial to ensure they have ample opportunities to progress and that every obstacle is removed from their path. When women are given opportunities to advance, they, in turn, unlock new opportunities for the nation. For years, there were many professions in our country where women were barred from working. Our BJP government resolved to eliminate every barrier hindering our daughters. Today, you can see that women are being deployed on the front lines of the army. Our daughters are also becoming fighter pilots in significant numbers. Many women are now joining the police force. Moreover, our daughters are leading prominent companies. Across the country, there are 1,200 producer associations or cooperative societies of farmers and cattle rearers that are being led by women. Whether in sports or education, our daughters are excelling in every field. Additionally, lakhs of women have benefited from the extension of maternity leave to 26 weeks. "
    "Friends, "
    "Often, when we see a sportsperson proudly displaying a medal or triumphantly walking with the tricolour atop Mount Everest, we fail to realise the years of dedication and relentless effort that went into achieving that success. The foundation of the Bima Sakhi programme, launched here today, is similarly built upon years of perseverance and tireless effort. Even after 60–65 years of independence, most women in Bharat did not have their personal bank accounts. This meant that women were largely excluded from the formal banking system. Recognising this gap, our government prioritised opening Jan Dhan bank accounts for mothers and sisters. Today, I am proud to say that more than 30 crore women and daughters now have Jan Dhan accounts. Imagine what the situation would have been like without these Jan Dhan accounts. Without them, gas subsidy amounts would not have been directly deposited into your accounts. During the COVID-19 pandemic, financial assistance could not have been provided to you. The funds from the Kisan Kalyan Nidhi would not have reached the accounts of women, nor would daughters have benefitted from the higher interest offered under the Sukanya Samriddhi Yojana. Money for building homes would not have been directly transferred to women’s accounts. Moreover, sisters establishing small businesses would not have been able to access banks, and it would have been nearly impossible for crores of women to secure collateral-free loans under the Mudra Yojana. Because women now have their personal bank accounts, they are able to avail themselves of Mudra loans and, for the first time, start businesses and ventures of their choosing. "
    "Friends, "
    "Our sisters have played a pivotal role in expanding banking services to every village. It is remarkable to see women who once lacked bank accounts themselves now connecting others to the banking system as Bank Sakhis. These mothers and sisters are teaching people how to save money, how to take loans, and how to utilise banking facilities effectively. Today, lakhs of Bank Sakhis are providing essential services in rural areas. "
    "Friends, "
    "Just as women were once excluded from banking, they were also not part of the insurance ecosystem. Today, a campaign has been launched to make lakhs of women insurance agents, or Bima Sakhis. This initiative enables women, who were once denied access to insurance services, to become key players in connecting others to these services. In doing so, they will also lead the expansion of the insurance sector. Under the Bima Sakhi Yojana, we aim to provide employment opportunities to 2 lakh women. Sisters and daughters who have passed the 10th standard will receive specialised training, financial assistance for three years, and allowances. According to industry data, an LIC agent earns an average monthly income of Rs 15,000. This means that our Bima Sakhis can expect to earn more than Rs 1.75 lakh annually. This income will provide much-needed financial support to their families. "
    "Friends, "
    "The significance of the work being undertaken by Bima Sakhis extends beyond their monthly earnings. Their role will be instrumental in achieving our nation’s goal of 'Insurance for All'. This mission is vital for enhancing social security and eradicating poverty at its roots. The role that you are playing today as Bima Sakhi will strengthen the mission of Insurance for All. "
    "Friends, "
    "We have seen clear examples of how insurance empowers individuals and transforms lives. The government has launched the 'Pradhan Mantri Jeevan Jyoti Bima Yojana' and the 'Pradhan Mantri Suraksha Bima Yojana', which provide insurance coverage of up to Rs 2 lakh each at highly affordable premiums. Over 20 crore people in the country, many of whom had never imagined having insurance, are now insured under these schemes. To date, approximately Rs 20,000 crore in claim settlements has been provided. Just imagine the impact—if someone suffered an accident or lost a loved one, how crucial that Rs 2 lakh would be during such a challenging time. This means the Bima Sakhis are not just offering insurance; they are providing a vital social security net to countless families and performing a deeply virtuous service. "
    "Friends, "
    "The revolutionary policies and decisions implemented over the last 10 years for rural women in Bharat deserve recognition and study. Names like Bima Sakhi, Bank Sakhi, Krishi Sakhi, Pashu Sakhi, Drone Didi, and Lakhpati Didi may seem simple and ordinary, but these women are reshaping the destiny of India. The Self-Help Group (SHG) movement, in particular, is a remarkable story of women’s empowerment that will be celebrated in history. We have transformed SHGs into powerful tools for revolutionising the rural economy. Today, 10 crore women across the country are associated with SHGs, earning livelihoods through their efforts. In the past decade, the government has provided financial assistance exceeding RS 8 lakh crore to SHGs, significantly boosting their contributions. "
    "Friends, "
    "To all the women associated with SHGs nationwide, I want to emphasise how extraordinary your role is and how immense your contribution is. You are driving Bharat towards becoming the world’s third-largest economic power. Women from all sections of society, every class, and every family are part of this movement, ensuring inclusivity for everyone. This movement of SHGs is not only uplifting rural economies but also fostering social harmony and justice. It is often said in our country that when one daughter is educated, two families benefit. Similarly, SHGs not only increase the income of one woman but also boost the confidence of her entire family and village. Your work is immense and invaluable. "
    "Friends, "
    "From the ramparts of the Red Fort, I also announced the goal of creating 3 crore Lakhpati Didis. So far, more than 1 crore 15 lakh Lakhpati Didis have emerged across the country, with each earning over Rs 1 lakh annually. The Lakhpati Didi initiative is further bolstered by the government’s NaMo Drone Didi scheme, which is receiving widespread acclaim in Haryana. During the Haryana elections, I came across interviews with some sisters. One sister shared her journey of becoming a trained drone pilot, how her group acquired a drone, and how she secured work spraying crops during the last Kharif season. She sprayed pesticides over approximately 800 acres of farmland using the drone. Do you know how much she earned? She made Rs 3 lakh in just one season. This initiative is not only transforming agriculture but also revolutionising the lives of women, enabling them to achieve financial independence and prosperity. "
    "Friends, "
    "Today, thousands of Krishi Sakhis are being trained to raise awareness about modern farming practices, natural farming, and sustainable agricultural methods across the country. So far, nearly 70,000 Krishi Sakhis have received their certifications. These Krishi Sakhis also have the potential to earn over Rs 60,000 annually. Similarly, more than 1.25 lakh Pashu Sakhis are actively participating in awareness campaigns on animal husbandry. The roles of Krishi Sakhis and Pashu Sakhis extend far beyond employment; they are providing invaluable service to humanity. Just as nurses play a crucial role in saving lives and providing care, Krishi Sakhis are safeguarding Mother Earth for future generations. By promoting organic farming, they are serving the soil, our farmers, and the planet itself. Likewise, Pashu Sakhis are contributing significantly by caring for animals, thereby performing an equally noble service to humanity. "
    "Friends, "
    "There are those who view everything through the lens of politics and vote banks, and they are visibly perplexed and troubled these days. They cannot comprehend why the blessings of mothers, sisters, and daughters continue to grow in Modi’s favour, election after election. Those who treated women merely as a vote bank and indulged in token announcements during election seasons cannot grasp this deep and genuine bond. "
    "To understand the immense love and affection I receive from mothers and sisters, one must look back at the last 10 years. A decade ago, crores of women lacked access to basic sanitation. Today, more than 12 crore toilets have been built across the country. Ten years ago, crores of women did not have gas connections. Through the Ujjwala Yojana, free connections were provided, and cylinder prices were made more affordable. Many homes lacked water taps; we have initiated the provision of tap water to every household. In the past, women rarely owned property. Now, crores of women are proud owners of pucca houses. For decades, women have demanded 33% reservation in the Lok Sabha and state assemblies. With your blessings, we had the privilege of fulfilling this long-standing demand. When sincere efforts are made with pure intentions, they earn the heartfelt blessings of mothers and sisters. "
    "Friends, "
    "Our double-engine government is working with full sincerity for the welfare of farmers. During the first two terms, Haryana’s farmers received more than Rs 1.25 lakh crore as Minimum Support Price (MSP). In this third term, Rs 14,000 crore has already been provided as MSP to paddy, millet, and moong farmers. Additionally, over Rs 800 crore has been allocated to assist farmers affected by drought. We all recognise the pivotal role played by Chaudhary Charan Singh University in establishing Haryana as a leader of the Green Revolution. Now, in the 21st century, Maharana Pratap Horticulture University will play a key role in making Haryana a leader in fruit and vegetable production. Today, the foundation stone for the new campus of Maharana Pratap Horticulture University has been laid, which will provide modern facilities for the youth pursuing studies in this field. "
    "Friends, "
    "Today, I once again assure all of you, especially the sisters of Haryana, that the state will witness rapid development. The double-engine government, in its third term, will work with triple the speed. The role of women empowerment in this progress will continue to expand and flourish. May your love and blessings remain with us always. With this hope, I extend my heartfelt congratulations and best wishes to everyone once again. "
    "Say with me— "
    "Bharat Mata Ki Jai! "
    "Bharat Mata Ki Jai! "
    "Bharat Mata Ki Jai! "
    "Thank you very much! "
)

    segmenter = Segmenter(corpus)
    segmenter.show()