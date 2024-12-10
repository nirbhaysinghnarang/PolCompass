import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained("manifesto-project/manifestoberta-xlm-roberta-56policy-topics-context-2024-1-1", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")

political_sentences_m = [
    "On this very day, 9th December, the first meeting of the Constituent Assembly was held. As the nation celebrates 75 years of the Constitution, this date serves as a reminder to uphold the principles of equality and inclusive development.",
    "The way Haryana embraced the mantra, 'Ek hai to Safe Hai' (If we are together, we are safe), has set a remarkable example for the entire nation.",
    "The double-engine government here is now working at double the speed.",
    "During the elections, the women of Haryana raised the slogan, 'Mhara Haryana, Non-Stop Haryana.' We have adopted this slogan as our resolution.",
    "The Bima Sakhi scheme has just been launched here to create employment opportunities for the women and daughters of the nation.",
    "Bharat is now advancing with the resolve to become a developed nation by 2047.",
    "Since 1947, the collective energy of every community and every region has brought Bharat to its current heights. However, to achieve the goal of a Viksit Bharat (developed India) by 2047, we must tap into many new sources of energy.",
    "One such source is Eastern India, including the North East. Another critical source of energy is the Nari Shakti—the women power—of our nation.",
    "To empower women, it is crucial to ensure they have ample opportunities to progress and that every obstacle is removed from their path.",
    "The revolutionary policies and decisions implemented over the last 10 years for rural women in Bharat deserve recognition and study.",
    "From the ramparts of the Red Fort, I also announced the goal of creating 3 crore Lakhpati Didis.",
    "Today, the foundation stone for the new campus of Maharana Pratap Horticulture University has been laid, which will provide modern facilities for the youth pursuing studies in this field.",
    "Our double-engine government is working with full sincerity for the welfare of farmers."
]

political_sentences = [
    "For years, Americans have watched as our country has been stripped of our jobs and stripped of our wealth. We've watched our companies get sold off to foreign countries. But with my plan for the American economy, this will stop immediately. When I am president, we will begin to take other countries' jobs and factories, bringing businesses and trillions of dollars back to the United States.",
    "Under my plan, American workers will no longer be worried about losing their jobs to foreign nations. Instead, foreign nations will be worried about losing their jobs to Americans. German car companies can become American car companies. We can beat China in electronic production. Manufacturers that have left us will come sprinting back to our shores.",
    "Here's the deal that I will be offering to companies and manufacturers around the planet: The United States will give you the lowest taxes, the lowest energy costs, the lowest regulatory burdens, and free access to the best and biggest market on the planet—but only if you make your products here in America and hire American workers for the job.",
    "If these companies don't take the deal, they'll pay a tariff when they send their products—made in another country—to us. And we will use the hundreds of billions of tariff dollars to benefit American citizens.",
    "This New American Industrialism will create millions of jobs, massively raise wages for American workers, and make the United States a manufacturing powerhouse once again.",
    "By contrast, Vice President Kamala Harris, the Tax Queen, plans to tax unrealized capital gains, so if a company succeeds, it must give half of its value to the government. Harris is right now shutting down power-plants across the country, causing electricity prices to soar by more than 100 percent and driving us into third world status by attacking the entire fossil fuel industry.",
    "In Kamala Harris' America, if you ship production overseas, she will give you a tax break and subsidies. With her tie-breaking vote on what I call the 'Inflation Creation Act,' she sent billions to Chinese battery factories, Chinese solar factories, and Chinese electronics factories. It's no wonder that under Kamala Harris, we lost 24,000 manufacturing jobs in August 2024 alone.",
    "But this horrific nightmare for American workers ends the day I take the oath of office.",
    "The centerpiece of my plan for a manufacturing renaissance will be a 15 percent Made in America Corporate Tax Rate, cutting the business tax from 21 percent to 15 percent—but again, only for those who make their product in America. U.S.-based manufacturers will also be rewarded with expanded research and development tax credits to help build the sprawling, state-of-the art plants our country needs to be an industrial superpower in the modern world.",
    "We will cut energy and electricity prices in half within 12 months—not just for businesses but for all Americans and their families, and we will quickly double our electricity capacity, which we need to do to compete with China and other countries on Artificial Intelligence. With the lowest energy prices on earth, we will attract energy-hungry industries from all over the planet and millions of blue-collar jobs.",
    "We will also set up special zones on federal land with ultra-low taxes and regulations for American producers, to entice the relocation of entire industries from other countries. And we will seriously expedite environmental approvals so we can use the resources we have right here on American soil.",
    "And I have pledged to remove 10 old and burdensome regulations from the books for every new regulation. In my first term as president, I was the biggest regulation cutter in history!",
    "To help ensure our pro-manufacturing policies reach their full potential, I will also appoint a Manufacturing Ambassador, whose sole task will be to go around the world and convince major manufacturers to pack up and move their production to America.",
    "And we will also remove millions of illegal immigrants from this country. These migrants are taking jobs from American workers and driving down their wages, especially African American and Hispanic workers. Our agenda will always prioritize American workers making good wages.",
    "The same can't be said for Vice President Kamala Harris, who oversaw an influx of millions upon millions of illegal immigrants into the United States, presiding over the worst border crisis in American history.",
    "If Vice President Harris gets four more years, this time as president, she will de-industrialize the United States and destroy our country. We will become a Banana Republic. There will be no car industry, no steel industry, no significant manufacturing of any kind—and we will be at risk of military defeat.",
    "Here's a very simple fact: If you don't have steel, you don't have a military. That's one of the reasons I'm not going to allow Japan to buy U.S. Steel, which 70 years ago was the greatest company on earth.",
    "My industrial policy is only one piece of an economic vision for revitalizing this country for the hardest working Americans. It will join my 2017 tax cuts for working Americans—the largest in U.S. history, which I will make permanent—and my new proposal to ban all taxes on overtime hours, tips for service workers, and Social Security benefits for our great seniors. And we will deport millions of illegal immigrants and begin to control our own borders again, restoring wonderful jobs in America to legal American workers.",
    "This is how we will make America affordable again.",
    "Vote Trump, and you will see a mass exodus of manufacturing from China to Pennsylvania, from South Korea to North Carolina, and from Germany to Georgia.",
    "We will bring back the American dream, bigger, better, and stronger than ever before."
]
context = ""
political_count = len(political_sentences)

# Initialize a dictionary to accumulate scores for each category
category_scores = {}

for sentence in political_sentences:
    print("\nProcessing political sentence:")
    inputs = tokenizer(sentence, context, return_tensors="pt", max_length=300, padding="max_length", truncation=True)
    logits = model(**inputs).logits
    probabilities = torch.softmax(logits, dim=1).tolist()[0]

    # Convert to {label_name: prob%}
    class_probs = {model.config.id2label[i]: probabilities[i]*100 for i in range(len(probabilities))}
    # Sort by probability descending
    class_probs = dict(sorted(class_probs.items(), key=lambda item: item[1], reverse=True))

    # Print top 5 predicted classes
    top_5 = list(class_probs.items())[:5]
    for predicted_class, probability in top_5:
        print(f"Class: {predicted_class} with probability {probability:.2f}%")

    # Update category scores
    for predicted_class, probability in top_5:
        # Accumulate the probability for each category
        if predicted_class not in category_scores:
            category_scores[predicted_class] = 0.0
        category_scores[predicted_class] += probability

# After processing all sentences, we have cumulative category scores.
print("\nCumulative Category Scores:")
for category, score in category_scores.items():
    print(f"{category}: {score:.2f}%")

# Sort categories by cumulative score to see which categories dominate.
sorted_scores = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)
print("\nTop categories by cumulative score:")
for category, score in sorted_scores[:10]:
    print(f"{category}: {score:.2f}%")



# ---------------------------
# Political Compass Integration
# ---------------------------


def calculate_econ_soc_from_probs(class_probs, category_map):
    """Calculate econ and social score for a single sentence from class probabilities."""
    econ_total = 0.0
    soc_total = 0.0
    for cat, prob in class_probs.items():
        if cat in category_map:
            econ_score, soc_score = category_map[cat]
            # prob is in percentage, divide by 100 to get fraction if needed
            econ_total += econ_score * (prob / 100.0)
            soc_total += soc_score * (prob / 100.0)
    return econ_total, soc_total

sentence_econ_scores = []
sentence_soc_scores = []

for sentence in political_sentences:
    print("\nProcessing political sentence:")
    inputs = tokenizer(sentence, context, return_tensors="pt", max_length=300, padding="max_length", truncation=True)
    logits = model(**inputs).logits
    probabilities = torch.softmax(logits, dim=1).tolist()[0]

    # Convert to {label_name: prob%}
    class_probs = {model.config.id2label[i]: probabilities[i]*100 for i in range(len(probabilities))}
    # Sort by probability descending
    class_probs = dict(sorted(class_probs.items(), key=lambda item: item[1], reverse=True))

    # Print top 5 predicted classes
    top_5 = list(class_probs.items())[:5]
    for predicted_class, probability in top_5:
        print(f"Class: {predicted_class} with probability {probability:.2f}%")

    # Compute econ and social scores for this sentence
    # First, we sum the top 5 probabilities to normalize
    top_5_total_prob = sum(prob for _, prob in top_5)
    # Normalize probabilities so each sentence is on a comparable scale
    normalized_top_5 = {cat: (prob / top_5_total_prob) * 100 for cat, prob in top_5}

    # Calculate sentence-level econ and social scores
    econ_score_sentence, soc_score_sentence = calculate_econ_soc_from_probs(normalized_top_5, category_ideology_mapping)
    sentence_econ_scores.append(econ_score_sentence)
    sentence_soc_scores.append(soc_score_sentence)

if sentence_econ_scores and sentence_soc_scores:
    final_econ = sum(sentence_econ_scores) / len(sentence_econ_scores)
    final_soc = sum(sentence_soc_scores) / len(sentence_soc_scores)
else:
    final_econ = 0.0
    final_soc = 0.0

print("\nFinal Normalized Political Compass Scores:")
print(f"Economic axis score: {final_econ:.2f} (negative=left, positive=right)")
print(f"Social axis score: {final_soc:.2f} (negative=authoritarian, positive=libertarian)")