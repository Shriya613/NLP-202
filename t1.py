import numpy as np
from collections import defaultdict
from sklearn.metrics import classification_report

# --- Load CoNLL-2003 Data ---
def load_data(path):
    data = []
    sentence = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                parts = line.strip().split()
                word = parts[0]  # First column is the word
                pos = parts[1]  # Second column is POS tag
                ner_tag = parts[-1]  # Last column is the NER tag
                sentence.append((word, pos, ner_tag))
            else:
                if sentence:
                    data.append(sentence)
                sentence = []
    return data

# --- Feature Extraction ---
def extract_features(sentence, i, prev_tag, tag, gazetteer):
    word, pos, ner = sentence[i]
    features = {}
    
    features[f"W={word}+T={tag}"] = 1.0
    features[f"O={word.lower()}+T={tag}"] = 1.0
    features[f"P={pos}+T={tag}"] = 1.0
    features[f"T-1={prev_tag}+T={tag}"] = 1.0
    
    if word in gazetteer:
        features[f"GAZ=True+T={tag}"] = 1.0
    
    return features

# --- Viterbi Decoder ---
TAGS = ['O', 'B-LOC', 'I-LOC', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-MISC', 'I-MISC']

def viterbi_decode(sentence, weights, gazetteer):
    n = len(sentence)
    dp = [{} for _ in range(n+1)]
    back = [{} for _ in range(n+1)]
    dp[0]['<START>'] = 0.0

    for i in range(n):
        for prev_tag in dp[i]:
            for tag in TAGS:
                feats = extract_features(sentence, i, prev_tag, tag, gazetteer)
                s = dp[i][prev_tag] + sum(weights.get(f, 0.0) * v for f, v in feats.items())
                if tag not in dp[i+1] or s > dp[i+1][tag]:
                    dp[i+1][tag] = s
                    back[i+1][tag] = prev_tag

    # Backtrack
    tags = []
    curr = max(dp[n], key=dp[n].get)
    for i in reversed(range(1, n+1)):
        tags.append(curr)
        curr = back[i][curr]
    return list(reversed(tags))

# --- Training with Structured Perceptron ---
def train_ssgd(train_data, dev_data, gazetteer, epochs=5):
    weights = defaultdict(float)
    for epoch in range(epochs):
        for sentence in train_data:
            gold_tags = [token[2] for token in sentence]
            pred_tags = viterbi_decode(sentence, weights, gazetteer)
            if gold_tags != pred_tags:
                f_gold = extract_features(sentence, 0, '<START>', gold_tags[0], gazetteer)
                f_pred = extract_features(sentence, 0, '<START>', pred_tags[0], gazetteer)
                for f in set(f_gold.keys()).union(f_pred.keys()):
                    weights[f] += f_gold.get(f, 0) - f_pred.get(f, 0)
        print(f"Epoch {epoch+1} done")
    return weights

# --- Evaluation ---
def evaluate(pred_tags, data):
    gold = [tag[2] for sent in data for tag in sent]
    pred = [tag for tags in pred_tags for tag in tags]
    print(classification_report(gold, pred, zero_division=0))

# --- Load Gazetteer ---
def load_gazetteer(filename):
    gazetteer = set()
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            gazetteer.add(line.strip())
    return gazetteer

# --- Run Training & Evaluation ---
if __name__ == "__main__":
    dataset_path = "/Users/yshriyasravani/Downloads/NLP-202-A1-Code/"  # Update this path if needed

    train_data = load_data(dataset_path + "ner.train")
    dev_data = load_data(dataset_path + "ner.dev")
    test_data = load_data(dataset_path + "ner.test")
    gazetteer = load_gazetteer(dataset_path + "gazetteer.txt")
    
    weights = train_ssgd(train_data, dev_data, gazetteer)
    
    dev_preds = [viterbi_decode(sent, weights, gazetteer) for sent in dev_data]
    test_preds = [viterbi_decode(sent, weights, gazetteer) for sent in test_data]
    
    print("Dev Set Performance:")
    evaluate(dev_preds, dev_data)
    print("Test Set Performance:")
    evaluate(test_preds, test_data)
