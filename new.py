import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_data(file_path):
    """Load data from file where each line contains either a token or a token and its tag separated by whitespace"""
    sentences, labels = [], []
    sentence, label = [], []

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()

            if not line:
                if sentence:
                    sentences.append(sentence)
                    labels.append(label)
                    sentence, label = [], []
            else:
                parts = line.split()
                if len(parts) == 2:
                    # Line has token and tag
                    token, tag = parts
                    sentence.append(token)
                    label.append(tag)
                elif len(parts) == 1:
                    # Line has token only, tag is "O" (outside)
                    token = parts[0]
                    sentence.append(token)
                    label.append("O")
                else:
                    print(f"Skipping invalid line: {line}")
                    continue
        
        if sentence:
            sentences.append(sentence)
            labels.append(label)

    print(f"Total sentences loaded: {len(sentences)}")
    return sentences, labels

def build_vocab(sentences, labels):
    word_to_idx = {'<PAD>': 0, '<UNK>': 1}
    char_to_idx = {'<PAD>': 0, '<UNK>': 1}
    label_to_idx = {'<PAD>': 0}
    
    for sentence in sentences:
        for word in sentence:
            if word not in word_to_idx:
                word_to_idx[word] = len(word_to_idx)
            for char in word:
                if char not in char_to_idx:
                    char_to_idx[char] = len(char_to_idx)
                    
    for sentence_labels in labels:
        for tag in sentence_labels:
            if tag not in label_to_idx:
                label_to_idx[tag] = len(label_to_idx)

    idx_to_tag = {i: t for t, i in label_to_idx.items()}
    return word_to_idx, char_to_idx, label_to_idx, idx_to_tag

class NERDataset(Dataset):
    def __init__(self, sentences, labels, word_to_idx, char_to_idx, label_to_idx):
        self.sentences = sentences
        self.labels = labels
        self.word_to_idx = word_to_idx
        self.char_to_idx = char_to_idx
        self.label_to_idx = label_to_idx

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        words = self.sentences[idx]
        word_indices = [self.word_to_idx.get(w, 1) for w in words]
        
        char_indices = []
        for word in words:
            chars = [self.char_to_idx.get(c, 1) for c in word]
            char_indices.append(chars)
        
        labels = self.labels[idx]
        label_indices = [self.label_to_idx.get(l, 0) for l in labels]
        
        return {
            'word_ids': torch.tensor(word_indices, dtype=torch.long),
            'char_ids': char_indices,
            'label_ids': torch.tensor(label_indices, dtype=torch.long),
            'length': len(word_indices)
        }

def collate_fn(batch):
    batch = sorted(batch, key=lambda x: x['length'], reverse=True)
    
    word_ids = [item['word_ids'] for item in batch]
    char_ids = [item['char_ids'] for item in batch]
    label_ids = [item['label_ids'] for item in batch]
    lengths = [item['length'] for item in batch]
    
    word_ids = pad_sequence(word_ids, batch_first=True)
    label_ids = pad_sequence(label_ids, batch_first=True)
    
    max_word_len = max([len(word) for words in char_ids for word in words])
    batch_size = len(char_ids)
    max_sent_len = max([len(words) for words in char_ids])
    
    char_tensor = torch.zeros(batch_size, max_sent_len, max_word_len, dtype=torch.long)
    
    for batch_idx, words in enumerate(char_ids):
        for word_idx, chars in enumerate(words):
            char_len = len(chars)
            char_tensor[batch_idx, word_idx, :char_len] = torch.tensor(chars, dtype=torch.long)
    
    return {
        'word_ids': word_ids.to(device),
        'char_ids': char_tensor.to(device),
        'label_ids': label_ids.to(device),
        'lengths': torch.tensor(lengths, dtype=torch.long).to(device)
    }

class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, char_size, tagset_size, 
                 embedding_dim=100, char_embedding_dim=30, hidden_dim=256):
        super(BiLSTM_CRF, self).__init__()
        
        # Embeddings
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.char_embedding = nn.Embedding(char_size, char_embedding_dim, padding_idx=0)
        
        # Character CNN
        self.char_cnn = nn.Conv1d(char_embedding_dim, 50, kernel_size=3, padding=1)
        
        # BiLSTM
        self.lstm = nn.LSTM(embedding_dim + 50, hidden_dim // 2, 
                           num_layers=2, bidirectional=True, 
                           batch_first=True, dropout=0.5)
        
        # Output layer
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        
        # CRF parameters
        self.transitions = nn.Parameter(torch.randn(tagset_size, tagset_size))
        self.transitions.data[0, :] = -10000  # No transitions from PAD
        self.transitions.data[:, 0] = -10000  # No transitions to PAD
        
        self.start_transitions = nn.Parameter(torch.randn(tagset_size))
        self.end_transitions = nn.Parameter(torch.randn(tagset_size))
        self.start_transitions.data[0] = -10000
        self.end_transitions.data[0] = -10000
    
    def _get_char_features(self, char_ids):
        batch_size, max_seq_len, max_word_len = char_ids.shape
        
        # Process each word
        char_ids = char_ids.view(batch_size * max_seq_len, max_word_len)
        char_embeds = self.char_embedding(char_ids)
        
        # Transpose for convolution
        char_embeds = char_embeds.transpose(1, 2)
        
        # Apply CNN and max pooling
        char_features = self.char_cnn(char_embeds)
        char_features = torch.max(char_features, dim=2)[0]
        
        # Reshape back
        return char_features.view(batch_size, max_seq_len, -1)
    
    def _get_lstm_features(self, word_ids, char_ids, lengths):
        # Word embeddings
        word_embeds = self.word_embedding(word_ids)
        
        # Character features
        char_features = self._get_char_features(char_ids)
        
        # Combine embeddings
        embeds = torch.cat([word_embeds, char_features], dim=2)
        
        # Pack for LSTM
        packed_embeds = pack_padded_sequence(embeds, lengths.cpu(), batch_first=True)
        
        # LSTM forward pass
        packed_lstm_out, _ = self.lstm(packed_embeds)
        lstm_out, _ = pad_packed_sequence(packed_lstm_out, batch_first=True)
        
        # Project to tag space
        emissions = self.hidden2tag(lstm_out)
        
        return emissions
    
    def _score_sentence(self, emissions, tags, mask):
        batch_size, seq_len, num_tags = emissions.shape
        
        # Start transition scores
        score = self.start_transitions[tags[:, 0]]
        
        # Add emission scores
        score += emissions[torch.arange(batch_size).unsqueeze(1), torch.arange(seq_len), tags].sum(dim=1)
        
        # Add transition scores
        for i in range(seq_len - 1):
            valid_transitions = mask[:, i+1]
            score += self.transitions[tags[:, i], tags[:, i+1]] * valid_transitions
        
        # Add end transition scores
        seq_ends = mask.sum(dim=1) - 1
        last_tags = tags[torch.arange(batch_size), seq_ends]
        score += self.end_transitions[last_tags]
        
        return score
    
    def _forward_algorithm(self, emissions, mask):
        batch_size, seq_len, num_tags = emissions.shape
        
        # Initialize alphas with start transitions
        alphas = self.start_transitions.unsqueeze(0) + emissions[:, 0]
        
        for i in range(1, seq_len):
            alpha_t = alphas.unsqueeze(2)
            trans_t = self.transitions.unsqueeze(0)
            emit_t = emissions[:, i].unsqueeze(1)
            
            scores = alpha_t + trans_t + emit_t
            scores = torch.logsumexp(scores, dim=1)
            
            mask_t = mask[:, i].unsqueeze(1)
            alphas = alphas * (~mask_t) + scores * mask_t
        
        # Add end transitions
        end_scores = alphas + self.end_transitions.unsqueeze(0)
        log_partition = torch.logsumexp(end_scores, dim=1)
        
        return log_partition
    
    def neg_log_likelihood(self, word_ids, char_ids, tags, lengths):
        emissions = self._get_lstm_features(word_ids, char_ids, lengths)
        mask = torch.zeros_like(word_ids, dtype=torch.bool)

        for i, length in enumerate(lengths):
            mask[i, :length] = 1

        gold_score = self._score_sentence(emissions, tags, mask)
        log_norm = self._forward_algorithm(emissions, mask)

        # Compute Hamming loss
        pred_tags = self._viterbi_decode(emissions, mask)
        hamming_cost = sum(sum(gold != pred for gold, pred in zip(g.tolist(), p.tolist())) 
                            for g, p in zip(tags, pred_tags)) * 0.1  # Hamming factor

        return (log_norm - gold_score + hamming_cost).mean()

    
    def _viterbi_decode(self, emissions, mask):
        batch_size, seq_len, num_tags = emissions.shape
        
        # Initialize scores
        scores = self.start_transitions + emissions[:, 0]
        history = []
        
        for i in range(1, seq_len):
            broadcast_scores = scores.unsqueeze(2)
            broadcast_trans = self.transitions.unsqueeze(0)
            
            next_scores = broadcast_scores + broadcast_trans + emissions[:, i].unsqueeze(1)
            best_scores, best_tags = next_scores.max(dim=1)
            
            mask_i = mask[:, i].unsqueeze(1)
            scores = scores * (~mask_i) + best_scores * mask_i
            history.append(best_tags)
        
        # Add end transitions
        scores += self.end_transitions
        _, best_last_tags = scores.max(dim=1)
        
        # Follow backpointers
        best_tags = []
        for idx in range(batch_size):
            best_tag_seq = [best_last_tags[idx].item()]
            seq_len_i = int(mask[idx].sum().item())
            
            for hist_idx in reversed(range(seq_len_i - 1)):
                best_tag = history[hist_idx][idx][best_tag_seq[-1]]
                best_tag_seq.append(best_tag.item())
            
            best_tag_seq.reverse()
            best_tags.append(best_tag_seq)
        
        return best_tags
    
    def forward(self, word_ids, char_ids, lengths):
        # For prediction
        emissions = self._get_lstm_features(word_ids, char_ids, lengths)
        
        # Create mask
        mask = torch.zeros_like(word_ids, dtype=torch.bool)
        for i, length in enumerate(lengths):
            mask[i, :length] = 1
        
        # Viterbi decoding
        return self._viterbi_decode(emissions, mask)

def train(model, train_loader, optimizer, epochs=10, patience=3):
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        start_time = time.time()
        
        for batch in train_loader:
            word_ids = batch['word_ids']
            char_ids = batch['char_ids']
            tags = batch['label_ids']
            lengths = batch['lengths']
            
            optimizer.zero_grad()
            loss = model.neg_log_likelihood(word_ids, char_ids, tags, lengths)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Time: {time.time()-start_time:.2f}s")
        
        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'best_model.pt')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping!")
                break
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pt'))
    return model

def evaluate(model, data_loader, idx_to_tag):
    model.eval()
    all_preds = []
    all_true = []
    
    with torch.no_grad():
        for batch in data_loader:
            word_ids = batch['word_ids']
            char_ids = batch['char_ids']
            tags = batch['label_ids']
            lengths = batch['lengths']
            
            # Get predictions
            pred_tags = model(word_ids, char_ids, lengths)
            
            # Collect true and predicted labels
            for i, length in enumerate(lengths):
                length = length.item()
                true_seq = tags[i, :length].cpu().tolist()
                pred_seq = pred_tags[i][:length]
                
                all_true.extend(true_seq)
                all_preds.extend(pred_seq)
    
    # Skip padding tags
    valid_indices = [i for i, tag in enumerate(all_true) if tag != 0]
    filtered_true = [all_true[i] for i in valid_indices]
    filtered_preds = [all_preds[i] for i in valid_indices]
    
    # Calculate metrics
    p, r, f1, _ = precision_recall_fscore_support(
        filtered_true, filtered_preds, average='weighted')
    
    print(f"Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}")
    return p, r, f1

def predict_and_save(model, data_loader, idx_to_tag, file_path, original_sentences=None):
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for batch in data_loader:
            word_ids = batch['word_ids']
            char_ids = batch['char_ids']
            lengths = batch['lengths']
            
            pred_tags = model(word_ids, char_ids, lengths)
            
            for i, pred_seq in enumerate(pred_tags):
                length = lengths[i].item()
                tags = [idx_to_tag[tag] for tag in pred_seq[:length]]
                predictions.append(tags)
    
    # Write predictions
    with open(file_path, 'w') as f:
        if original_sentences:
            for i, (sentence, pred_tags) in enumerate(zip(original_sentences, predictions)):
                for token, tag in zip(sentence, pred_tags):
                    f.write(f"{token}\t{tag}\n")
                f.write("\n")
        else:
            for pred_tags in predictions:
                for tag in pred_tags:
                    f.write(f"{tag}\n")
                f.write("\n")
    
    print(f"Predictions saved to {file_path}")

def compare_batch_sizes(model, train_sentences, train_labels, word_to_idx, 
                         char_to_idx, label_to_idx, batch_sizes=[32, 64]):
    results = {}
    
    # Use a small subset of data
    subset_size = min(500, len(train_sentences))
    subset_sentences = train_sentences[:subset_size]
    subset_labels = train_labels[:subset_size]
    
    for batch_size in batch_sizes:
        # Create dataloader with current batch size
        loader = create_data_loader(
            subset_sentences, subset_labels, word_to_idx, 
            char_to_idx, label_to_idx, batch_size)
        
        # Time one epoch
        model.train()
        start_time = time.time()
        
        for batch in loader:
            word_ids = batch['word_ids']
            char_ids = batch['char_ids']
            tags = batch['label_ids']
            lengths = batch['lengths']
            
            # Forward pass only
            loss = model.neg_log_likelihood(word_ids, char_ids, tags, lengths)
        
        elapsed = time.time() - start_time
        results[batch_size] = elapsed
        print(f"Batch size {batch_size}: {elapsed:.2f} seconds")
    
    return results

def create_data_loader(sentences, labels, word_to_idx, char_to_idx, label_to_idx, batch_size=32):
    dataset = NERDataset(sentences, labels, word_to_idx, char_to_idx, label_to_idx)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

def main():
    # Load data
    train_path = 'A2-data_2/train'  # Update with your path
    dev_path = 'A2-data_2/dev'
    test_path = 'A2-data_2/test'
    
    print("Loading data...")
    train_sentences, train_labels = load_data(train_path)
    dev_sentences, dev_labels = load_data(dev_path)
    test_sentences, test_labels = load_data(test_path)
    
    # Check if data is loaded correctly
    print(f"Loaded {len(train_sentences)} training sentences")
    print(f"Loaded {len(dev_sentences)} dev sentences")
    print(f"Loaded {len(test_sentences)} test sentences")
    
    if not train_sentences or not dev_sentences or not test_sentences:
        print("Error: One or more datasets are empty. Please check the file paths and data files.")
        return
    
    # Use a subset for faster training if needed
    subset_size = min(1000, len(train_sentences))
    train_sentences = train_sentences[:subset_size]
    train_labels = train_labels[:subset_size]
    
    # Build vocabulary
    word_to_idx, char_to_idx, label_to_idx, idx_to_tag = build_vocab(
        train_sentences, train_labels)
    
    # Create dataloaders
    batch_size = 32
    train_loader = create_data_loader(
        train_sentences, train_labels, word_to_idx, char_to_idx, label_to_idx, batch_size)
    
    dev_loader = create_data_loader(
        dev_sentences, dev_labels, word_to_idx, char_to_idx, label_to_idx, batch_size)
    
    test_loader = create_data_loader(
        test_sentences, test_labels, word_to_idx, char_to_idx, label_to_idx, batch_size)
    
    # Initialize model
    model = BiLSTM_CRF(
        vocab_size=len(word_to_idx),
        char_size=len(char_to_idx),
        tagset_size=len(label_to_idx),
        embedding_dim=100,
        char_embedding_dim=30,
        hidden_dim=256
    ).to(device)
    
    # Compare batch sizes for minibatching demonstration
    print("\nComparing training times for different batch sizes...")
    batch_results = compare_batch_sizes(
        model, train_sentences, train_labels, word_to_idx, 
        char_to_idx, label_to_idx, batch_sizes=[16, 32, 64])
    
    # Train the model
    print("\nTraining model...")
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model = train(model, train_loader, optimizer, epochs=15, patience=3)
    
    # Evaluate
    print("\nEvaluating on dev set:")
    dev_p, dev_r, dev_f1 = evaluate(model, dev_loader, idx_to_tag)
    
    print("\nEvaluating on test set:")
    test_p, test_r, test_f1 = evaluate(model, test_loader, idx_to_tag)
    
    # Save predictions
    print("\nSaving predictions...")
    predict_and_save(model, dev_loader, idx_to_tag, 'dev.output', dev_sentences)
    predict_and_save(model, test_loader, idx_to_tag, 'test.output', test_sentences)
    
    # Print results
    print("\nFinal Results:")
    print(f"Dev - P: {dev_p:.4f}, R: {dev_r:.4f}, F1: {dev_f1:.4f}")
    print(f"Test - P: {test_p:.4f}, R: {test_r:.4f}, F1: {test_f1:.4f}")
    
    # Print batch size comparison
    print("\nBatch Size Training Time Comparison:")
    for size, time_taken in batch_results.items():
        print(f"Batch size {size}: {time_taken:.2f} seconds")

if __name__ == "__main__":
    main()