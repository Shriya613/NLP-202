#character cnn

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict

# --------------------------- Character CNN Embedding --------------------------- #
class CharCNN(nn.Module):
    def __init__(self, char_vocab_size, char_embedding_dim, out_channels, kernel_size, max_word_len):
        super(CharCNN, self).__init__()
        self.char_embeds = nn.Embedding(char_vocab_size, char_embedding_dim, padding_idx=0)
        self.conv = nn.Conv1d(in_channels=char_embedding_dim, 
                              out_channels=out_channels, 
                              kernel_size=kernel_size, 
                              padding=kernel_size//2)
        self.maxpool = nn.MaxPool1d(kernel_size=max_word_len - kernel_size + 1)
    
    def forward(self, char_sequences):
        batch_size, word_count, max_word_len = char_sequences.shape  # (B, W, L)
        char_embeds = self.char_embeds(char_sequences.view(-1, max_word_len))  # (B*W, L, E)
        char_embeds = char_embeds.permute(0, 2, 1)  # (B*W, E, L)
        conv_out = torch.relu(self.conv(char_embeds))  # (B*W, out_channels, L)
        pooled = self.maxpool(conv_out).squeeze(-1)  # (B*W, out_channels)
        return pooled.view(batch_size, word_count, -1)  # (B, W, out_channels)

# --------------------------- BiLSTM-CRF Model with CharCNN --------------------------- #
class BiLSTM_CRF_Char(nn.Module):
    def __init__(self, vocab_size, char_vocab_size, tag_to_ix, embedding_dim, char_embedding_dim, char_out_channels, hidden_dim, max_word_len):
        super(BiLSTM_CRF_Char, self).__init__()
        self.embedding_dim = embedding_dim
        self.char_out_channels = char_out_channels
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        
        # Word Embedding Layer
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Character CNN
        self.char_cnn = CharCNN(char_vocab_size, char_embedding_dim, char_out_channels, kernel_size=3, max_word_len=max_word_len)
        
        # BiLSTM Layer (Word Embedding + Character Embedding)
        self.lstm = nn.LSTM(embedding_dim + char_out_channels, hidden_dim // 2, num_layers=1, bidirectional=True, batch_first=True)
        
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)
        
        # CRF Transition Matrix
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))
        self.transitions.data[tag_to_ix["<START>"], :] = -10000
        self.transitions.data[:, tag_to_ix["<STOP>"]] = -10000
    
    def _get_lstm_features(self, sentences, char_sequences):
        word_embeds = self.word_embeds(sentences)  # (B, W, E)
        char_embeds = self.char_cnn(char_sequences)  # (B, W, char_out_channels)
        combined_embeds = torch.cat((word_embeds, char_embeds), dim=2)  # (B, W, E + char_out_channels)
        
        lstm_out, _ = self.lstm(combined_embeds)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats
    
    def neg_log_likelihood(self, sentences, char_sequences, tags):
        feats = self._get_lstm_features(sentences, char_sequences)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return (forward_score - gold_score).mean()
    
    def forward(self, sentences, char_sequences):
        feats = self._get_lstm_features(sentences, char_sequences)
        return feats.argmax(dim=2)  # Predict the most likely tags

# --------------------------- Training Configuration --------------------------- #
EMBEDDING_DIM = 128
CHAR_EMBEDDING_DIM = 30
CHAR_OUT_CHANNELS = 50
HIDDEN_DIM = 256
MAX_WORD_LEN = 10  # Max character length per word

model = BiLSTM_CRF_Char(len(word_to_ix), len(char_to_ix), tag_to_ix, EMBEDDING_DIM, CHAR_EMBEDDING_DIM, CHAR_OUT_CHANNELS, HIDDEN_DIM, MAX_WORD_LEN)
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(20):
    for sentences, char_sequences, labels in train_loader:
        model.zero_grad()
        loss = model.neg_log_likelihood(sentences, char_sequences, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# --------------------------- Inference --------------------------- #
with torch.no_grad():
    test_outputs = []
    for sentence, char_sequence in zip(test_sentences, test_char_sequences):
        sentence_tensor = prepare_sequence(sentence, word_to_ix).unsqueeze(0)
        char_tensor = prepare_char_sequence(char_sequence, char_to_ix, MAX_WORD_LEN).unsqueeze(0)
        predicted_tags = model(sentence_tensor, char_tensor)
        predicted_labels = [list(tag_to_ix.keys())[i] for i in predicted_tags[0]]
        test_outputs.append(predicted_labels)

# Save predictions
with open("test.output", "w") as f:
    for sentence, labels in zip(test_sentences, test_outputs):
        for word, label in zip(sentence, labels):
            f.write(f"{word}\t{label}\n")
        f.write("\n")
