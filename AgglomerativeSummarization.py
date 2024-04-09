import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util
import scipy.cluster.hierarchy as sch

# Sample sentences (you can replace this with your document's sentences)
document_sentences = [
    
    "Cardiff University students said they had received first class grades for essays written using the AI chatbot.",
    "ChatGPT is an AI program capable of producing human-like responses and academic pieces of work.",
    "Cardiff University said it was reviewing its policies and would issue new university-wide guidance shortly.",
    "Tom, not his real name, is one of the students who conducted his own experiment using ChatGPT.",
    "Tom, who averages a 2.1 grade, submitted two 2,500 essays in January, one with the help of the chatbot and one without.",
    "For the essay he wrote with the help of AI, Tom received a first - the highest mark he has ever had at university.",
    "In comparison, he received a low 2.1 on the essay he wrote without the software.",
    "I did not copy everything word for word, but I would prompt it with questions that gave me access to information much quicker than usual, said Tom.",
    "He also admitted that he would most likely continue to use ChatGPT for the planning and framing of his essays.",
    "A recent Freedom of Information request to Cardiff University revealed that during the January 2023 assessment period, there were 14,443 visits to the ChatGPT site on the university's own wi-fi networks.",
    "One month before, there were zero recorded visits.",
    "Despite the increase in visits during Januarys assessment period, the university believes there is nothing to suggest that the visits were for illegitimate purposes.",
    "Most visits have been identified as coming from our research network - our School of Computer Science and Informatics, for example, has an academic interest in the research and teaching of artificial intelligence, said Cardiff University.",
    "John, not his real name, is another student at the university who admitted using the software to help him with assignments.",
    "I have used it quite a few times since December. I think I have used it at least a little bit for every assessment I have had, he said.",
    "It is basically just become part of my work process, and will probably continue to be until I can not access it anymore.",
    "When I first started using it, I asked it to write stuff like compare this niche theory with this other niche theory in an academic way and it just aced it.",
    "Although ChatGPT does not insert references, John said he had no issue filling those in himself.",
    "I have also used it to summarise concepts from my course that I do not think the lecturers have been great at explaining, he said.",
    "It is a really good tool for cutting out the waffle that some lecturers go into for theories which you do not actually need to talk about in essays.",
    "It probably cuts about 20% of the effort I would need to put into an essay.",
    "Both students said they do not use ChatGPT to write their essays, but to generate content they can tweak and adapt themselves.",
    "As for being caught, John is certain that the AI influence in his work is undetectable.",
    "I see no way that anyone could distinguish between work completely my own and work which was aided by AI, he said.",
    "However, John is concerned about being caught in the future. He said if transcripts of his communication with the AI network were ever found, he fears his degree could be taken away from him.",
    "I am glad I used it when I did, in the final year of my degree, because I feel like a big change is coming to universities when it comes to coursework because it's way too easy to cheat with the help of AI, he said.",
    "I like to think that I have avoided this, whilst reaping the benefits of GPT in my most important year.",
    "Cardiff University said it took allegations of academic misconduct, including plagiarism, extremely seriously.",
    "Although not specifically referenced, the improper use of AI would be covered by our existing academic integrity policy, a spokesman said.",
    "We are aware of the potential impact of AI programmes, like ChatGPT, on our assessments and coursework.",
    "Maintaining academic integrity is our main priority and we actively discourage any student from academic misconduct in its many forms."

]


# Initialize the RoBERTa-based Sentence Transformer model
#model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
model = SentenceTransformer('bert-base-nli-mean-tokens')


# Sample reference summary sentences (you can replace this with your summary)
reference_summary_sentences = [
    "Students at Cardiff University have reported receiving high grades for essays produced with the assistance of an AI chatbot.",
    "A recent Freedom of Information request submitted to the university disclosed that, during the January 2023 assessment period, there were a total of 14,443 visits to the ChatGPT site from the university's own Wi-Fi networks.",
    "Both students have clarified that they employ ChatGPT not to compose their essays but to generate content that they can subsequently modify and customize to their needs.",
    "Cardiff University emphasized its strong commitment to addressing allegations of academic misconduct, such as plagiarism."
]


# Compute embeddings for reference summary sentences
reference_embeddings = model.encode(reference_summary_sentences, convert_to_tensor=True)

# Calculate cosine similarity matrix between document sentences and reference summary sentences
cosine_similarity_matrix = []
for doc_sentence in document_sentences:
    doc_sentence = doc_sentence.strip()
    if not doc_sentence:
        continue
    sentence_embedding = model.encode(doc_sentence, convert_to_tensor=True)
    similarity_scores = util.pytorch_cos_sim(sentence_embedding.unsqueeze(0), reference_embeddings)
    cosine_similarity_matrix.append(similarity_scores[0].tolist())

# Perform hierarchical clustering
linkage_matrix = sch.linkage(cosine_similarity_matrix, method='ward')

# Plot dendrogram with white background
fig, ax = plt.subplots(figsize=(8, 10))
d = sch.dendrogram(linkage_matrix, labels=document_sentences, orientation="left", leaf_font_size=8)


# Find the indices of the 4 most similar sentences
cosine_similarity_scores = np.max(cosine_similarity_matrix, axis=1)
most_similar_indices = np.argsort(cosine_similarity_scores)[-4:]

# Highlight the 4 most important sentences in red
for i, sentence_index in enumerate(d['leaves']):
    if sentence_index in most_similar_indices:
        plt.gca().get_yticklabels()[i].set_color('red')

#plt.title('Hierarchical Clustering Dendrogram with Highlighted of Most Important Sentences')
plt.title('Semantic Similaity Score')
#plt.xlabel('Semantic Similaity Score')

# Set the background color to white
ax.set_facecolor('white')

plt.show()

# Print the 4 most important sentences
print("The 4 most important sentences are:")
for i, index in enumerate(most_similar_indices):
    print(f"{i + 1}. {document_sentences[index]}")
