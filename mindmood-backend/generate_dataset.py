import random
import pandas as pd

# Keywords extracted from your original index1.html
keywords = {
    'amused': ["amused", "funny", "hilarious", "laugh", "laughing", "laughter", "joke", "comic", "entertaining", "fun", "playful", "silly", "witty", "humorous", "cheerful", "delighted", "tickled", "giddy", "giggly", "chuckling", "smiling", "happy", "joyful", "enjoying", "pleased", "content", "light-hearted", "carefree"],
    'angry': ["angry", "furious", "rage", "mad", "livid", "irritated", "annoyed", "frustrated", "upset", "outraged", "enraged", "seething", "fuming", "bitter", "resentful", "hostile", "aggressive", "cross", "irate", "indignant", "wrathful", "incensed", "provoked", "exasperated", "vexed", "infuriated", "antagonistic"],
    'disgusted': ["disgusted", "disgust", "repulsed", "revolted", "appalled", "sickened", "nauseated", "repugnant", "abhorrent", "loathsome", "vile", "gross", "yuck", "ew", "eww", "revolting", "offensive", "distasteful", "unpleasant", "repellent", "abominable", "detestable", "contemptible", "despicable", "foul", "putrid", "disgusting"],
    'neutral': ["neutral", "okay", "fine", "alright", "normal", "regular", "usual", "average", "meh", "so-so", "neither", "neither good nor bad", "mediocre", "ordinary", "standard", "typical", "unremarkable", "indifferent", "balanced", "stable", "steady", "consistent", "unchanged", "same", "moderate", "fair", "plain"],
    'sleepy': ["sleepy", "tired", "exhausted", "fatigued", "drowsy", "weary", "worn out", "drained", "lethargic", "sluggish", "groggy", "yawning", "dozing", "napping", "sleepless", "insomnia", "rest", "sleep", "nap", "bed", "tired out", "beat", "knackered", "wiped out", "zonked", "dead tired", "bushed", "pooped"]
}

templates = [
    "I feel very {keyword}.",
    "Today I am feeling {keyword}.",
    "This makes me {keyword}.",
    "I'm so {keyword} right now.",
    "Everything feels {keyword}.",
    "My mood is {keyword}.",
    "That was really {keyword}!",
    "I can't stop feeling {keyword}."
]

data = []
labels = []

for mood, kw_list in keywords.items():
    for kw in kw_list:
        for _ in range(8):  # Generate 8 variations per keyword
            template = random.choice(templates)
            sentence = template.format(keyword=kw)
            data.append(sentence)
            labels.append(mood)

# Shuffle and save
combined = list(zip(data, labels))
random.shuffle(combined)
data[:], labels[:] = zip(*combined)

df = pd.DataFrame({'text': data, 'label': labels})
df.to_csv('mood_dataset.csv', index=False)
print(f"Dataset generated: {len(df)} samples saved to mood_dataset.csv")