# ----- Part 1 and Part 2 ------ #

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
import time
import os
import logging

torch.manual_seed(1)

# --------------------------- Setup Logging --------------------------- #
logging.basicConfig(
    filename="training_p1.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    filemode="w"  # Overwrite log file each run; use "a" to append
)

# --------------------------- Data Loading --------------------------- #
def load_data(file_path):
    sentences, labels = [], []
    with open(file_path, 'r', encoding='utf-8') as f:
        sentence, label = [], []
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
                    sentence, label = [], []
        if sentence:
            sentences.append(sentence)
            labels.append(label)
    return sentences, labels

train_sentences, train_labels = load_data("A2-data_2/train")[:1000]
dev_sentences, dev_labels = load_data("A2-data_2/dev")
test_sentences, _ = load_data("A2-data_2/test")

# --------------------------- Vocabulary Creation --------------------------- #
def create_vocab(sentences):
    word_to_ix = defaultdict(lambda: len(word_to_ix))
    word_to_ix["<PAD>"] = 0
    for sentence in sentences:
        for word in sentence:
            word_to_ix[word]
    return dict(word_to_ix)

word_to_ix = create_vocab(train_sentences)
tag_to_ix = {"B-DNA": 0, "I-DNA": 1, "B-RNA": 2, "I-RNA": 3, "B-protein": 4, "I-protein": 5,
             "B-cell_line": 6, "I-cell_line": 7, "B-cell_type": 8, "I-cell_type": 9, "O": 10,
             "<START>": 11, "<STOP>": 12, "<PAD>": 13}
ix_to_tag = {v: k for k, v in tag_to_ix.items()}

# --------------------------- Data Preparation --------------------------- #
class NERDataset(Dataset):
    def __init__(self, sentences, labels):
        self.sentences = sentences
        self.labels = labels

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx], self.labels[idx]

def prepare_sequence(seq, to_ix):
    return torch.tensor([to_ix.get(w, to_ix["<PAD>"]) for w in seq], dtype=torch.long)

def pad_collate(batch):
    sentences, labels = zip(*batch)
    sentence_tensors = [prepare_sequence(s, word_to_ix) for s in sentences]
    label_tensors = [prepare_sequence(l, tag_to_ix) for l in labels]
    lengths = torch.tensor([len(s) for s in sentences], dtype=torch.long)
    
    sentence_tensors = pad_sequence(sentence_tensors, batch_first=True, padding_value=word_to_ix["<PAD>"])
    label_tensors = pad_sequence(label_tensors, batch_first=True, padding_value=tag_to_ix["<PAD>"])
    
    return sentence_tensors, label_tensors, lengths

train_data = NERDataset(train_sentences, train_labels)
dev_data = NERDataset(dev_sentences, dev_labels)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=pad_collate)

# --------------------------- BiLSTM-CRF Model (Part 1) --------------------------- #
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

        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))
        self.transitions.data[tag_to_ix["<START>"], :] = -10000
        self.transitions.data[:, tag_to_ix["<STOP>"]] = -10000

    def _get_lstm_features(self, sentences, lengths):
        embeds = self.word_embeds(sentences)
        lstm_out, _ = self.lstm(embeds)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _forward_alg(self, feats):
        batch_size, seq_len, _ = feats.shape
        init_alphas = torch.full((batch_size, self.tagset_size), -10000., device=feats.device)
        init_alphas[:, self.tag_to_ix["<START>"]] = 0.
        forward_var = init_alphas

        for i in range(seq_len):
            alphas_t = []
            for next_tag in range(self.tagset_size):
                emit_score = feats[:, i, next_tag]
                trans_score = self.transitions[next_tag].unsqueeze(0)
                next_tag_var = forward_var + trans_score + emit_score.unsqueeze(1)
                alphas_t.append(torch.logsumexp(next_tag_var, dim=1))
            forward_var = torch.stack(alphas_t, dim=1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix["<STOP>"]].unsqueeze(0)
        return torch.logsumexp(terminal_var, dim=1)

    def _score_sentence(self, feats, tags):
        batch_size, seq_len = tags.shape
        start_tags = torch.full((batch_size, 1), self.tag_to_ix["<START>"], dtype=torch.long, device=feats.device)
        tags = torch.cat([start_tags, tags], dim=1)
        score = torch.zeros(batch_size, device=feats.device)
        for i in range(seq_len):
            score += self.transitions[tags[:, i + 1], tags[:, i]] + feats[torch.arange(batch_size), i, tags[:, i + 1]]
        score += self.transitions[self.tag_to_ix["<STOP>"], tags[:, -1]]
        return score

    def neg_log_likelihood(self, sentences, tags, lengths):
        feats = self._get_lstm_features(sentences, lengths)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return (forward_score - gold_score).mean()

    # --------------------------- Viterbi Decoder (Part 2) --------------------------- #
    def _viterbi_decode(self, feats):
        batch_size, seq_len, _ = feats.shape
        print(f"Feats shape: {feats.shape}")  # Debug
        backpointers = []
        init_vvars = torch.full((batch_size, self.tagset_size), -10000., device=feats.device)
        init_vvars[:, self.tag_to_ix["<START>"]] = 0
        forward_var = init_vvars

        for i in range(seq_len):
            bptrs_t = []
            vvars_t = []
            for next_tag in range(self.tagset_size):
                next_tag_var = forward_var + self.transitions[next_tag].unsqueeze(0)
                best_tag_id = next_tag_var.argmax(dim=1)
                bptrs_t.append(best_tag_id)
                vvars_t.append(next_tag_var[torch.arange(batch_size), best_tag_id])
            forward_var = (torch.stack(vvars_t, dim=1) + feats[:, i])
            backpointers.append(torch.stack(bptrs_t, dim=1))

        terminal_var = forward_var + self.transitions[self.tag_to_ix["<STOP>"]].unsqueeze(0)
        best_tag_id = terminal_var.argmax(dim=1)
        best_score = terminal_var[torch.arange(batch_size), best_tag_id]

        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[torch.arange(batch_size), best_tag_id]
            best_path.append(best_tag_id)
        best_path = torch.stack(best_path[::-1][1:], dim=1)
        print(f"Best path shape: {best_path.shape}, Tags: {best_path[0].tolist()}")  # Debug
        return best_score, best_path

    def forward(self, sentences, lengths):
        feats = self._get_lstm_features(sentences, lengths)
        _, tag_seq = self._viterbi_decode(feats)
        return tag_seq

# --------------------------- Training (Part 1) --------------------------- #
EMBEDDING_DIM, HIDDEN_DIM, EPOCHS = 128, 256, 10
model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(EPOCHS):
    start_time = time.time()
    model.train()
    total_loss = 0
    for sentences, labels, lengths in train_loader:
        model.zero_grad()
        loss = model.neg_log_likelihood(sentences, labels, lengths)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    epoch_time = time.time() - start_time
    log_message = f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}, Time: {epoch_time:.2f}s"
    print(log_message)
    logging.info(log_message)

# Saving model
torch.save(model.state_dict(), "bilstm_crf_base.pth")
print("Base model saved to bilstm_crf_base.pth")

# --------------------------- Inference and Evaluation (Part 2) --------------------------- #

def evaluate_model(model, sentences, gold_labels, file_name):
    model.eval()
    outputs = []
    with torch.no_grad():
        for sentence, gold in zip(sentences, gold_labels):
            sentence_tensor = prepare_sequence(sentence, word_to_ix).unsqueeze(0)
            lengths = torch.tensor([len(sentence)], dtype=torch.long)
            tags = model(sentence_tensor, lengths)[0].tolist()
            predicted_labels = [ix_to_tag[i] for i in tags if i != tag_to_ix["<PAD>"]]
            print(f"Sentence: {sentence}, Predicted Tags: {predicted_labels}")  # Debug
            outputs.append(predicted_labels[:len(sentence)])

    with open(file_name, "w") as f:
        for sentence, labels in zip(sentences, outputs):
            for word, label in zip(sentence, labels):
                f.write(f"{word}\t{label}\n")
            f.write("\n")
    print(f"Wrote {len(outputs)} sentences to {file_name}")  # Debug
    return outputs

evaluate_model(model, dev_sentences, dev_labels, "bilstm_crfp1_dev.output")
evaluate_model(model, test_sentences, [[] for _ in test_sentences], "bilstm_crfp1_test.output")

# Hypothetical evaluation (run manually with perl script)
# os.system("perl A2-data_2/evalIOB2.pl A2-data_2/dev.answers bilstm_crf1_dev.output")