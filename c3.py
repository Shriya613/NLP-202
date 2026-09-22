import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict

torch.manual_seed(1)

# --------------------------- Data Loading --------------------------- #
def load_data(file_path):
    """Loads sentences and labels from the dataset files."""
    sentences = []
    labels = []
    with open(file_path, 'r', encoding='utf-8') as f:
        sentence = []
        label = []
        for line in f:
            if line.strip():
                parts = line.strip().split()
                if len(parts) == 2:
                    word, tag = parts
                    sentence.append(word)
                    label.append(tag)
            else:
                if sentence:
                    sentences.append(sentence)
                    labels.append(label)
                    sentence = []
                    label = []
        if sentence:
            sentences.append(sentence)
            labels.append(label)
    return sentences, labels

# Load datasets
train_sentences, train_labels = load_data("A2-data_2/train")
dev_sentences, dev_labels = load_data("A2-data_2/dev")
test_sentences, _ = load_data("A2-data_2/test")  # Test set has no labels

# --------------------------- Vocabulary Creation --------------------------- #
def create_vocab(sentences):
    """Creates a vocabulary from the training sentences."""
    word_to_ix = defaultdict(lambda: len(word_to_ix))
    word_to_ix["<PAD>"] = 0  # Padding token
    for sentence in sentences:
        for word in sentence:
            word_to_ix[word]  # Assign index
    return dict(word_to_ix)

word_to_ix = create_vocab(train_sentences)
tag_to_ix = {"B-DNA": 0, "I-DNA": 1, "B-RNA": 2, "I-RNA": 3, "B-protein": 4, "I-protein": 5,
             "B-cell_line": 6, "I-cell_line": 7, "B-cell_type": 8, "I-cell_type": 9, "O": 10,
             "<START>": 11, "<STOP>": 12, "<PAD>": 13}  # Added "<PAD>"

# --------------------------- Data Preparation --------------------------- #
def prepare_sequence(seq, to_ix):
    """Converts a sequence of words or tags to a tensor of indices."""
    return torch.tensor([to_ix.get(w, to_ix["<PAD>"]) for w in seq], dtype=torch.long)

def pad_collate(batch):
    """Custom collate function for DataLoader, handling padding."""
    sentences, labels = zip(*batch)
    sentence_tensors = [prepare_sequence(s, word_to_ix) for s in sentences]
    label_tensors = [prepare_sequence(l, tag_to_ix) for l in labels]
    
    sentence_tensors = pad_sequence(sentence_tensors, batch_first=True, padding_value=word_to_ix["<PAD>"])
    label_tensors = pad_sequence(label_tensors, batch_first=True, padding_value=tag_to_ix["<PAD>"])
    
    return sentence_tensors, label_tensors

# Create DataLoaders for Minibatching
train_data = list(zip(train_sentences, train_labels))
train_loader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=pad_collate)

# --------------------------- BiLSTM-CRF Model --------------------------- #
class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim, padding_idx=word_to_ix["<PAD>"])
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True, batch_first=True)

        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # CRF Transition Matrix
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))
        self.transitions.data[tag_to_ix["<START>"], :] = -10000
        self.transitions.data[:, tag_to_ix["<STOP>"]] = -10000

    def _get_lstm_features(self, sentences):
        embeds = self.word_embeds(sentences)
        lstm_out, _ = self.lstm(embeds)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _forward_alg(self, feats):
        """Computes the partition function using the forward algorithm."""
        batch_size, seq_len, _ = feats.shape

        # Initialize alpha values (forward variables)
        init_alphas = torch.full((batch_size, self.tagset_size), -10000., device=feats.device)
        init_alphas[:, self.tag_to_ix["<START>"]] = 0.  # Set start tag score

        forward_var = init_alphas

        for i in range(seq_len):
            alphas_t = []
            for next_tag in range(self.tagset_size):
                emit_score = feats[:, i, next_tag]  # Emission scores
                trans_score = self.transitions[next_tag].unsqueeze(0)  # Broadcast transition scores
                next_tag_var = forward_var + trans_score + emit_score.unsqueeze(1)
                alphas_t.append(torch.logsumexp(next_tag_var, dim=1))  # Log-sum-exp trick
            forward_var = torch.stack(alphas_t, dim=1)  # Update forward variables

        terminal_var = forward_var + self.transitions[self.tag_to_ix["<STOP>"]].unsqueeze(0)
        return torch.logsumexp(terminal_var, dim=1)  # Compute final log-sum-exp

    def _score_sentence(self, feats, tags):
        """Computes the score of a given tag sequence for a batch."""
        batch_size, seq_len, _ = feats.shape

        # Create a tensor of <START> tags for the batch
        start_tags = torch.full((batch_size, 1), self.tag_to_ix["<START>"], dtype=torch.long, device=feats.device)

        # Concatenate <START> token to the beginning of each tag sequence
        tags = torch.cat([start_tags, tags], dim=1)  # Shape: (batch_size, seq_len + 1)

        score = torch.zeros(batch_size, device=feats.device)

        for i in range(seq_len):
            transition_score = self.transitions[tags[:, i + 1], tags[:, i]]
            emission_score = feats[torch.arange(batch_size), i, tags[:, i + 1]]
            score += transition_score + emission_score

        score += self.transitions[self.tag_to_ix["<STOP>"], tags[:, -1]]
    
        return score


    def neg_log_likelihood(self, sentences, tags):
        feats = self._get_lstm_features(sentences)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)

        # Ensure loss is positive
        return (forward_score - gold_score).mean()

    def forward(self, sentences):
        feats = self._get_lstm_features(sentences)
        return feats.argmax(dim=2)  # Predict the most likely tags

# --------------------------- Training --------------------------- #
EMBEDDING_DIM = 128
HIDDEN_DIM = 256

model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(20):
    for sentences, labels in train_loader:
        model.zero_grad()
        loss = model.neg_log_likelihood(sentences, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# --------------------------- Inference on Test Set --------------------------- #
with torch.no_grad():
    test_outputs = []
    for sentence in test_sentences:
        sentence_tensor = prepare_sequence(sentence, word_to_ix).unsqueeze(0)
        predicted_tags = model(sentence_tensor)
        predicted_labels = [list(tag_to_ix.keys())[i] for i in predicted_tags[0]]
        test_outputs.append(predicted_labels)

# Save predictions
with open("test.output", "w") as f:
    for sentence, labels in zip(test_sentences, test_outputs):
        for word, label in zip(sentence, labels):
            f.write(f"{word}\t{label}\n")
        f.write("\n")  # Blank line between sentences

# --------------------------- Evaluation --------------------------- #
import os
os.system("perl evalIOB2.pl dev.answers test.output")
