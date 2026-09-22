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

# --- Basic Feature Extraction (Parts 1-3) ---
def extract_basic_features(sentence, i, prev_tag, tag):
    word, pos, ner = sentence[i]
    features = {}
    
    # Basic features (1-4)
    features[f"W={word}+T={tag}"] = 1.0
    features[f"T-1={prev_tag}+T={tag}"] = 1.0
    features[f"O={word.lower()}+T={tag}"] = 1.0
    features[f"P={pos}+T={tag}"] = 1.0
    
    return features

# --- Viterbi Decoder ---
TAGS = ['O', 'B-LOC', 'I-LOC', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-MISC', 'I-MISC']

def viterbi_decode(sentence, weights):
    n = len(sentence)
    dp = [{} for _ in range(n+1)]
    back = [{} for _ in range(n+1)]
    dp[0]['<START>'] = 0.0

    for i in range(n):
        for prev_tag in dp[i]:
            for tag in TAGS:
                feats = extract_basic_features(sentence, i, prev_tag, tag)
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
def train_ssgd(train_data, dev_data, epochs=5):
    weights = defaultdict(float)
    best_weights = None
    best_f1 = 0.0
    
    for epoch in range(epochs):
        # Shuffle training data
        np.random.shuffle(train_data)
        
        for sentence in train_data:
            gold_tags = [token[2] for token in sentence]
            pred_tags = viterbi_decode(sentence, weights)
            if gold_tags != pred_tags:
                f_gold = extract_basic_features(sentence, 0, '<START>', gold_tags[0])
                f_pred = extract_basic_features(sentence, 0, '<START>', pred_tags[0])
                for f in set(f_gold.keys()).union(f_pred.keys()):
                    weights[f] += f_gold.get(f, 0) - f_pred.get(f, 0)
        
        # Evaluate on dev set
        dev_preds = [viterbi_decode(sent, weights) for sent in dev_data]
        gold = [tag[2] for sent in dev_data for tag in sent]
        pred = [tag for tags in dev_preds for tag in tags]
        report = classification_report(gold, pred, output_dict=True, zero_division=0)
        current_f1 = report['weighted avg']['f1-score']
        
        print(f"Epoch {epoch+1} done, Dev F1: {current_f1:.4f}", flush=True)
        
        # Save best weights
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_weights = weights.copy()
    
    return best_weights

# --- Evaluation ---
def evaluate(pred_tags, data):
    gold = [tag[2] for sent in data for tag in sent]
    pred = [tag for tags in pred_tags for tag in tags]
    print(classification_report(gold, pred, zero_division=0))

# --- Generate CoNLL Format Output ---
def generate_conll_output(sentences, pred_tags_list, output_file):
    with open(output_file, 'a', encoding='utf-8') as f:
        # Write section header for Parts 1-3
        f.write("\n=== Parts 1-3 Output (Basic Features) ===\n\n")
        
        for i, (sentence, pred_tags) in enumerate(zip(sentences, pred_tags_list)):
            for (word, pos, gold_tag), pred_tag in zip(sentence, pred_tags):
                f.write(f"{word}\t{pos}\t{gold_tag}\t{pred_tag}\n")
            if i < len(sentences) - 1:
                f.write("\n")

# --- Run Training & Evaluation ---
if __name__ == "__main__":
    dataset_path = "/home/sy4/nlp202/a3/"  # Update this path if needed

    # Load data
    train_data = load_data(dataset_path + "ner.train")
    dev_data = load_data(dataset_path + "ner.dev")
    test_data = load_data(dataset_path + "ner.test")
    
    # Train the model
    print("Training model with basic features...")
    weights = train_ssgd(train_data, dev_data, epochs=20)
    
    # Generate predictions
    print("\nGenerating predictions...")
    dev_preds = [viterbi_decode(sent, weights) for sent in dev_data]
    test_preds = [viterbi_decode(sent, weights) for sent in test_data]
    
    # Print evaluation metrics
    print("\nDev Set Performance:")
    evaluate(dev_preds, dev_data)
    
    print("\nTest Set Performance:")
    evaluate(test_preds, test_data)
    
    # Generate CoNLL format output
    print("\nGenerating CoNLL format output...")
    generate_conll_output(test_data, test_preds, "all_outputs.txt")
    print("Done!", flush=True) 