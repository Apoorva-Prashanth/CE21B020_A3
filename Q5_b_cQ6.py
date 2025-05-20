import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, HTML
import zipfile
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Char-level vocabulary
class CharVocab:
    """
    A character-level vocabulary class for encoding and decoding sequences of characters.

    Attributes:
    -----------
    char2idx : dict
        Mapping from characters to integer indices. Includes special tokens:
        '<pad>' (0), '<sos>' (1), '<eos>' (2), and '<unk>' (3).
        
    idx2char : dict
        Reverse mapping from indices to characters.

    pad_idx : int
        Index of the padding token ('<pad>').

    sos_idx : int
        Index of the start-of-sequence token ('<sos>').

    eos_idx : int
        Index of the end-of-sequence token ('<eos>').

    Methods:
    --------
    encode(word: str) -> List[int]
        Converts a string into a list of indices, including <sos> at the start and <eos> at the end.
        Unknown characters are mapped to the <unk> index.

    decode(ids: List[int]) -> str
        Converts a list of indices back into a string, ignoring <sos> and <pad>, and stopping at <eos>.

    __len__() -> int
        Returns the size of the vocabulary (i.e., number of unique tokens including special tokens).
    """
    def __init__(self, words):
        chars = sorted(set("".join(words)))
        self.char2idx = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
        for c in chars:
            self.char2idx[c] = len(self.char2idx)
        self.idx2char = {i: c for c, i in self.char2idx.items()}
        self.pad_idx = self.char2idx['<pad>']
        self.sos_idx = self.char2idx['<sos>']
        self.eos_idx = self.char2idx['<eos>']

    def encode(self, word):
        return [self.sos_idx] + [self.char2idx.get(c, self.char2idx['<unk>']) for c in word] + [self.eos_idx]

    def decode(self, ids):
        chars = []
        for idx in ids:
            if idx == self.eos_idx:
                break
            if idx not in (self.sos_idx, self.pad_idx):
                chars.append(self.idx2char.get(idx, ''))
        return ''.join(chars)

    def __len__(self):
        return len(self.char2idx)

def read_file(path):
    with open(path, encoding='utf-8') as f:
        lines = f.read().strip().split('\n')
    return [(line.split('\t')[0], line.split('\t')[1]) for line in lines if len(line.split('\t')) >= 2]

train_pairs = read_file('/kaggle/input/malayalam/ml.translit.sampled.train.tsv')
dev_pairs = read_file('/kaggle/input/malayalam/ml.translit.sampled.dev.tsv')
test_pairs = read_file('/kaggle/input/malayalam/ml.translit.sampled.test.tsv')

src_vocab = CharVocab([src for _, src in train_pairs])
tgt_vocab = CharVocab([tgt for tgt, _ in train_pairs])

class TransliterationDataset(Dataset):
    def __init__(self, pairs, src_vocab, tgt_vocab):
        self.data = pairs
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tgt, src = self.data[idx]
        return torch.tensor(self.src_vocab.encode(src)), torch.tensor(self.tgt_vocab.encode(tgt))

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_pad = nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=src_vocab.pad_idx)
    tgt_pad = nn.utils.rnn.pad_sequence(tgt_batch, batch_first=True, padding_value=tgt_vocab.pad_idx)
    return src_pad, tgt_pad

train_loader = DataLoader(TransliterationDataset(train_pairs, src_vocab, tgt_vocab),
                          batch_size=64, shuffle=True, collate_fn=collate_fn)
dev_loader = DataLoader(TransliterationDataset(dev_pairs, src_vocab, tgt_vocab),
                        batch_size=32, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(TransliterationDataset(test_pairs, src_vocab, tgt_vocab),
                        batch_size=32, shuffle=False, collate_fn=collate_fn)

class Attention(nn.Module):
    def __init__(self, enc_hidden_dim, dec_hidden_dim):
        super().__init__()
        self.attn = nn.Linear(enc_hidden_dim + dec_hidden_dim, dec_hidden_dim)
        self.v = nn.Parameter(torch.rand(dec_hidden_dim))

    def forward(self, hidden, encoder_outputs, mask=None):
        src_len = encoder_outputs.shape[1]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        energy = energy.permute(0, 2, 1)
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)
        attention = torch.bmm(v, energy).squeeze(1)
        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e10)
        return F.softmax(attention, dim=1)

class Seq2Seq(nn.Module):
    def __init__(self, config, input_vocab_size, output_vocab_size):
        super().__init__()
        self.embedding_dim = config.embed_size
        self.hidden_size = config.hidden_size
        self.num_enc_layers = config.enc_layers
        self.num_dec_layers = config.dec_layers
        self.cell_type = config.cell
        self.device = device
        self.dropout = nn.Dropout(config.dropout)
        self.max_len = 30
        self.attention_weights = []  # Store attention weights for visualization

        self.encoder_embedding = nn.Embedding(input_vocab_size, self.embedding_dim)
        self.decoder_embedding = nn.Embedding(output_vocab_size, self.embedding_dim)

        RNN = {'RNN': nn.RNN, 'GRU': nn.GRU, 'LSTM': nn.LSTM}[self.cell_type]
        self.encoder = RNN(self.embedding_dim, self.hidden_size, num_layers=self.num_enc_layers,
                           batch_first=True, bidirectional=True)
        self.decoder = RNN(self.embedding_dim + self.hidden_size * 2, self.hidden_size * 2,
                           num_layers=self.num_dec_layers, batch_first=True)

        self.attention = Attention(self.hidden_size * 2, self.hidden_size * 2)
        self.fc = nn.Linear(self.hidden_size * 4, output_vocab_size)

        self.sos_idx = tgt_vocab.sos_idx
        self.eos_idx = tgt_vocab.eos_idx
        self.pad_idx = tgt_vocab.pad_idx

    def encode(self, src):
        embedded = self.dropout(self.encoder_embedding(src))
        outputs, h_n = self.encoder(embedded)
        if self.cell_type == 'LSTM':
            h, c = h_n
            h_cat = torch.cat((h[-2], h[-1]), dim=1).unsqueeze(0)
            c_cat = torch.cat((c[-2], c[-1]), dim=1).unsqueeze(0)
            return outputs, (h_cat, c_cat)
        else:
            h_cat = torch.cat((h_n[-2], h_n[-1]), dim=1).unsqueeze(0)
            return outputs, h_cat

    def decode_step(self, input_token, hidden, encoder_outputs):
        embedded = self.dropout(self.decoder_embedding(input_token))
        if self.cell_type == 'LSTM':
            h_t = hidden[0][-1]
        else:
            h_t = hidden[-1]
            
        attn_weights = self.attention(h_t, encoder_outputs)
        self.attention_weights.append(attn_weights)  # Store for visualization
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        rnn_input = torch.cat((embedded, context), dim=2)
        output, hidden = self.decoder(rnn_input, hidden)
        logits = self.fc(torch.cat((output.squeeze(1), context.squeeze(1)), dim=1))
        return logits, hidden, attn_weights

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        self.attention_weights = []  # Reset attention weights storage
        batch_size, tgt_len = tgt.shape
        encoder_outputs, hidden = self.encode(src)
        input_token = tgt[:, 0].unsqueeze(1)
        outputs = []

        for t in range(1, tgt_len):
            output, hidden, attn_weights = self.decode_step(input_token, hidden, encoder_outputs)
            outputs.append(output.unsqueeze(1))
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1).unsqueeze(1)
            input_token = tgt[:, t].unsqueeze(1) if teacher_force else top1
        return torch.cat(outputs, dim=1)

def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)
        optimizer.zero_grad()
        output = model(src, tgt)
        loss = criterion(output.view(-1, output.size(-1)), tgt[:, 1:].reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def beam_decode(model, src, beam_size):
    model.eval()
    with torch.no_grad():
        encoder_outputs, hidden = model.encode(src)
        batch_size = src.size(0)
        final_outputs = []

        for b in range(batch_size):
            h_b = (hidden[0][:, b:b+1, :].contiguous(), hidden[1][:, b:b+1, :].contiguous()) if model.cell_type == 'LSTM' else hidden[:, b:b+1, :].contiguous()
            enc_out_b = encoder_outputs[b:b+1]
            beams = [([model.sos_idx], 0.0, h_b)]
            
            for _ in range(model.max_len):
                new_beams = []
                for seq, score, h in beams:
                    if seq[-1] == model.eos_idx:
                        new_beams.append((seq, score, h))
                        continue
                    input_token = torch.tensor([[seq[-1]]], device=device)
                    out, h_new, _ = model.decode_step(input_token, h, enc_out_b)
                    log_probs = F.log_softmax(out, dim=1)
                    topk_probs, topk_idxs = torch.topk(log_probs, beam_size, dim=1)
                    for i in range(beam_size):
                        next_seq = seq + [topk_idxs[0][i].item()]
                        new_score = score + topk_probs[0][i].item()
                        new_beams.append((next_seq, new_score, h_new))
                beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]
            final_outputs.append(beams[0][0])
        return final_outputs

def plot_attention_heatmaps(model, test_samples, num_samples=9):
    model.eval()
    plt.figure(figsize=(15, 15))
    samples = test_samples[:num_samples]
    
    for i, (src, tgt) in enumerate(samples):
        plt.subplot(3, 3, i+1)
        with torch.no_grad():
            src_tensor = torch.tensor([src_vocab.encode(src)], device=device)
            tgt_tensor = torch.tensor([tgt_vocab.encode(tgt)], device=device)
            model(src_tensor, tgt_tensor)  # This populates attention_weights
            
            # Get attention weights and convert to numpy
            attn_weights = torch.cat(model.attention_weights).squeeze().cpu().numpy()
            
            # Create heatmap
            sns.heatmap(attn_weights, cmap="YlGnBu", 
                        xticklabels=list(src),
                        yticklabels=list(tgt_vocab.decode(tgt_tensor[0][1:-1])))
            plt.title(f"Input: {src}\nOutput: {tgt_vocab.decode(tgt_tensor[0][1:-1])}")
            plt.xlabel("Source Characters")
            plt.ylabel("Target Characters")
    
    plt.tight_layout()
    return plt

def evaluate_beam(model, dataloader, beam_size):
    model.eval()  # Set model to evaluation mode (disables dropout, etc.)
    total_seq, correct_seq = 0, 0  # Track full sequence accuracy
    total_tokens, correct_tokens = 0, 0  # Track token-level accuracy
    all_predictions = []  # Store predictions for analysis/visualization

    with torch.no_grad():  # Disable gradient computation for faster evaluation
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)  # Move tensors to device (CPU/GPU)

            preds = beam_decode(model, src, beam_size)  # Perform beam search decoding

            for i, (pred, true) in enumerate(zip(preds, tgt)):
                # Remove special tokens (pad/eos) from predictions and targets
                pred_trimmed = [tok for tok in pred[1:] if tok != model.pad_idx and tok != model.eos_idx]
                true_trimmed = [tok.item() for tok in true[1:] if tok.item() != model.pad_idx and tok.item() != model.eos_idx]

                # Check if entire sequence matches exactly
                if pred_trimmed == true_trimmed:
                    correct_seq += 1
                total_seq += 1

                # Compare token-by-token for token-level accuracy
                for p, t in zip(pred_trimmed, true_trimmed):
                    if p == t:
                        correct_tokens += 1
                total_tokens += len(true_trimmed)

                # Decode input, prediction, and ground truth for storage
                src_word = src_vocab.decode([x.item() for x in src[i] if x.item() not in (src_vocab.sos_idx, src_vocab.eos_idx, src_vocab.pad_idx)])
                pred_word = tgt_vocab.decode(pred)
                true_word = tgt_vocab.decode([x.item() for x in true if x.item() not in (tgt_vocab.pad_idx, tgt_vocab.eos_idx)])

                # Store triplet: (input word, predicted word, true word)
                all_predictions.append((src_word, pred_word, true_word))

    # Compute accuracy scores
    seq_accuracy = correct_seq / total_seq if total_seq > 0 else 0.0
    token_accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0.0

    return seq_accuracy, token_accuracy, all_predictions  # Return both accuracy metrics and all decoded predictions

def visualize_predictions(predictions, num_samples=10, log_to_wandb=False):
    """
    Visualizes a subset of model predictions by highlighting character-level differences 
    between the predicted and true output.

    Parameters:
    -----------
    predictions : list of tuples or lists
        Each element should be a (input, predicted, true) triple representing the input string,
        the model's prediction, and the ground truth string.

    num_samples : int, optional (default=10)
        The number of prediction samples to display.

    log_to_wandb : bool, optional (default=False)
        If True, logs the raw predictions to Weights & Biases using wandb.Table.

    Returns:
    --------
    styled_df

    Notes:
    ------
    - This function is useful for qualitative evaluation of sequence prediction models.
    - It uses HTML and pandas styling to visually compare predictions.
    """
    df = pd.DataFrame(predictions[:num_samples], columns=['Input', 'Predicted', 'True'])
    
    def highlight_diff(row):
        pred, true = row['Predicted'], row['True']
        diff = []
        for p, t in zip(pred, true):
            if p == t:
                diff.append(p)
            else:
                diff.append(f'<b style="color:red">{p}</b>')
        return ''.join(diff)
    
    df['Difference'] = df.apply(lambda row: highlight_diff(row), axis=1)
    
    def row_style(row):
        color = 'lightgreen' if row['Predicted'] == row['True'] else 'lightpink'
        return [f'background-color: {color}' for _ in row]
    
    styled_df = df.style.apply(row_style, axis=1).set_properties(**{'text-align': 'left'})
    display(HTML(styled_df.to_html(escape=False)))
    
    if log_to_wandb:
        wandb.log({"predictions": wandb.Table(dataframe=df)})
    
    return styled_df

def save_predictions(predictions, filename):
    os.makedirs('predictions_attention', exist_ok=True)
    df = pd.DataFrame(predictions, columns=['Input', 'Predicted', 'True'])
    df.to_csv(f'predictions_attention/{filename}', index=False)
    print(f"Saved to predictions_attention/{filename}")

def sweep_train(config=None):
    with wandb.init(config=config):
        config = wandb.config
        run_name = f"embed{config.embed_size}_hid{config.hidden_size}_enc{config.enc_layers}_dec{config.dec_layers}_{config.cell}_drop{config.dropout}_beam{config.beam_size}_attn"
        wandb.run.name = run_name

        model = Seq2Seq(config, len(src_vocab), len(tgt_vocab)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab.pad_idx)

        for epoch in range(20):  # Increased epochs for better training
            train_loss = train(model, train_loader, optimizer, criterion)
            acc, token_acc, val_preds = evaluate_beam(model, dev_loader, beam_size=config.beam_size)
            test_acc, test_token_acc, test_preds = evaluate_beam(model, test_loader, beam_size=config.beam_size)

            # Log attention heatmaps at the end of training
            if epoch == 19:
                # Get first 9 test samples properly
                test_samples = []
                for i in range(9):
                    src, tgt = test_loader.dataset[i]
                    src_word = src_vocab.decode([x.item() for x in src if x.item() not in (src_vocab.sos_idx, src_vocab.eos_idx, src_vocab.pad_idx)])
                    tgt_word = tgt_vocab.decode([x.item() for x in tgt if x.item() not in (tgt_vocab.pad_idx, tgt_vocab.eos_idx)])
                    test_samples.append((src_word, tgt_word))
                
                attention_plot = plot_attention_heatmaps(model, test_samples)
                wandb.log({"attention_heatmaps": wandb.Image(attention_plot)})
                plt.close()

            wandb.log({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_accuracy': acc,
                'val_token_accuracy': token_acc,
                'test_accuracy': test_acc,
                'test_token_accuracy': test_token_acc,
                'used_attention': True
            })

            print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Acc = {acc:.4f}, Test Acc = {test_acc:.4f}")
        visualize_predictions(test_preds, num_samples=15, log_to_wandb=True)
        save_predictions(test_preds, f'test_predictions_{run_name}.csv')

# Sweep Config
sweep_config = {
    'method': 'bayes',
    'metric': {'name': 'val_accuracy', 'goal': 'maximize'},
    'parameters': {
        'embed_size': {'values': [256]},
        'hidden_size': {'values': [128]},
        'enc_layers': {'values': [2]},
        'dec_layers': {'values': [1]},
        'dropout': {'values': [0.30]},
        'cell': {'values': ['LSTM']},
        'beam_size': {'values': [5]}
    }
}

sweep_id = wandb.sweep(sweep_config, project="A3_attention_ce21b020")
wandb.agent(sweep_id, function=sweep_train, count=1)

def create_prediction_zip():
    with zipfile.ZipFile('predictions_attention.zip', 'w') as zipf:
        for root, dirs, files in os.walk('predictions_attention'):
            for file in files:
                zipf.write(os.path.join(root, file))
    print("Zip created: predictions_attention.zip")

create_prediction_zip()

