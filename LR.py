#part 1 - Logistic Regression
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from collections import Counter
import random
import spacy
import numpy as np
import os
import tarfile
import json
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score

print("PyTorch version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
print("CUDA Device Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")

# Ensure dataset is extracted
dataset_path = "aclImdb"
if not os.path.exists(dataset_path):
    print("Extracting dataset...")
    with tarfile.open("aclImdb_v1.tar.gz", "r:gz") as tar:
        tar.extractall()
    print("Extraction complete.")

# Set random seed for reproducibility
SEED = 1234
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

# Ensure spaCy model is available
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading 'en_core_web_sm' model...")
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Define padding and unknown token indices
PAD_IDX = 0
UNK_IDX = 1

# Tokenization function using spaCy
def tokenize(text):
    return [token.text.lower() for token in nlp(text)]

# Save and Load Vocabulary
VOCAB_FILE = "vocab.json"

def build_vocab(texts, max_vocab_size=25_000):
    if os.path.exists(VOCAB_FILE):
        print("Loading existing vocabulary...")
        with open(VOCAB_FILE, "r") as f:
            return json.load(f)
    print("Building vocabulary...")
    counter = Counter(token for text in texts for token in tokenize(text))
    vocab = {word: idx + 2 for idx, (word, _) in enumerate(counter.most_common(max_vocab_size))}
    vocab["<pad>"] = PAD_IDX
    vocab["<unk>"] = UNK_IDX

    # Save vocabulary
    with open(VOCAB_FILE, "w") as f:
        json.dump(vocab, f)
    return vocab

def numericalize(texts, vocab):
    return [[vocab.get(token, UNK_IDX) for token in tokenize(text)] for text in texts]

# Load IMDB dataset
def load_imdb_data(data_dir):
    texts, labels = [], []
    for label_type in ["pos", "neg"]:
        folder = os.path.join(data_dir, label_type)
        if not os.path.exists(folder):
            raise FileNotFoundError(f"Error: {folder} does not exist. Verify dataset extraction.")
        for file in os.listdir(folder):
            with open(os.path.join(folder, file), "r", encoding="utf-8") as f:
                texts.append(f.read())
                labels.append(1 if label_type == "pos" else 0)
    return texts, labels

# Load dataset and shuffle
train_texts, train_labels = load_imdb_data("aclImdb/train")
test_texts, test_labels = load_imdb_data("aclImdb/test")

# Shuffle data before splitting
combined = list(zip(train_texts, train_labels))
random.shuffle(combined)
train_texts, train_labels = zip(*combined)

# Split data into train and validation sets
train_texts, valid_texts = train_texts[:20000], train_texts[20000:]
train_labels, valid_labels = train_labels[:20000], train_labels[20000:]

# Build and load vocabulary
vocab = build_vocab(train_texts)

# Dataset class
class IMDBDataset(Dataset):
    def __init__(self, texts, labels, vocab):
        self.texts = numericalize(texts, vocab)
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return torch.tensor(self.texts[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.float)

def collate_fn(batch):
    texts, labels = zip(*batch)
    lengths = torch.tensor([len(text) for text in texts])
    padded_texts = pad_sequence(texts, batch_first=True, padding_value=PAD_IDX)
    labels = torch.tensor(labels, dtype=torch.float)
    return padded_texts, labels, lengths

# Create dataloaders
train_dataset = IMDBDataset(train_texts, train_labels, vocab)
valid_dataset = IMDBDataset(valid_texts, valid_labels, vocab)
test_dataset = IMDBDataset(test_texts, test_labels, vocab)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Logistic Regression Model
class LogisticRegression(nn.Module):
    def __init__(self, vocab_size, embed_dim, dropout=0.3):
        super(LogisticRegression, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, 1)

    def forward(self, x):
        embedded = self.embedding(x)
        pooled = embedded.mean(dim=1)
        dropped = self.dropout(pooled)
        return self.fc(dropped).squeeze(1)

# Training function
def train_model(model, dataloader, optimizer, criterion, device, epoch):
    model.train()
    epoch_loss = 0
    start_time = time.time()
    for texts, labels, _ in tqdm(dataloader, desc=f"Training Epoch {epoch+1}"):
        texts, labels = texts.to(device), labels.to(device)
        optimizer.zero_grad()
        predictions = model(texts)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    end_time = time.time()
    elapsed_time = end_time - start_time
    return epoch_loss / len(dataloader), elapsed_time

# Evaluation function
def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for texts, labels, _ in tqdm(dataloader, desc="Evaluating"):
            texts, labels = texts.to(device), labels.to(device)
            predictions = model(texts)
            loss = criterion(predictions, labels)
            epoch_loss += loss.item()
            all_preds.extend(torch.round(torch.sigmoid(predictions)).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    return epoch_loss / len(dataloader), accuracy

# Hyperparameter tuning
batch_sizes = [1, 16, 32, 64, 128]
learning_rates = [1e-3, 1e-4]
dropout_rates = [0.3, 0.5]
num_epochs = [10, 20]
results = []
log_file = "training_log_lr.txt"

best_valid_acc = 0
best_params = {}

for batch_size in batch_sizes:
    for lr in learning_rates:
        for dropout in dropout_rates:
            for epochs in num_epochs:
                print(f"Running experiment with Batch Size={batch_size}, Learning Rate={lr}, Dropout={dropout}, Epochs={epochs}")

                # Update DataLoader
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
                valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

                # Initialize model, optimizer, and loss function
                model = LogisticRegression(len(vocab), embed_dim=100, dropout=dropout).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)  # L2 regularization
                criterion = nn.BCEWithLogitsLoss()

                # Train and evaluate model
                total_train_time = 0
                for epoch in range(epochs):
                    train_loss, elapsed_time = train_model(model, train_loader, optimizer, criterion, device, epoch)
                    total_train_time += elapsed_time

                valid_loss, valid_acc = evaluate_model(model, valid_loader, criterion, device)

                # Log results
                with open(log_file, "a") as f:
                    f.write(f"Batch Size={batch_size}, Learning Rate={lr}, Dropout={dropout}, Epochs={epochs}, Train Time={total_train_time:.2f}s, Valid Loss={valid_loss:.4f}, Valid Accuracy={valid_acc:.4f}\n")

                # Track best parameters
                if valid_acc > best_valid_acc:
                    best_valid_acc = valid_acc
                    best_params = {
                        "batch_size": batch_size,
                        "learning_rate": lr,
                        "dropout": dropout,
                        "epochs": epochs
                    }

                results.append((batch_size, lr, dropout, epochs, valid_acc, total_train_time))

# Evaluate on the test set with the best hyperparameters
print(f"Best Parameters: {best_params}")
with open(log_file, "a") as f:
    f.write(f"Best Parameters: {best_params}\n")

test_loader = DataLoader(test_dataset, batch_size=best_params['batch_size'], shuffle=False, collate_fn=collate_fn)

final_model = LogisticRegression(len(vocab), embed_dim=100, dropout=best_params['dropout']).to(device)
final_optimizer = torch.optim.Adam(final_model.parameters(), lr=best_params['learning_rate'], weight_decay=1e-5)
final_criterion = nn.BCEWithLogitsLoss()

# Train final model
total_final_train_time = 0
for epoch in range(best_params['epochs']):
    _, elapsed_time = train_model(final_model, train_loader, final_optimizer, final_criterion, device, epoch)
    total_final_train_time += elapsed_time

# Evaluate on test set
test_loss, test_acc = evaluate_model(final_model, test_loader, final_criterion, device)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

with open(log_file, "a") as f:
    f.write(f"Final Test Loss: {test_loss:.4f}, Final Test Accuracy: {test_acc:.4f}, Total Final Train Time: {total_final_train_time:.2f}s\n")

# Generate plots
batch_sizes, valid_accuracies = zip(*[(res[0], res[4]) for res in results if res[1] == 1e-3])
plt.figure(figsize=(10, 5))
plt.plot(batch_sizes, valid_accuracies, marker='o', label='Validation Accuracy')
plt.xlabel
