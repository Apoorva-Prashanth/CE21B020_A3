import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random

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

src_vocab = CharVocab([src for src, _ in train_pairs])
tgt_vocab = CharVocab([tgt for _, tgt in train_pairs])

class TransliterationDataset(Dataset):
    """
This code defines a PyTorch dataset and dataloaders for a character-level transliteration task. It converts input-output string pairs into sequences of token indices using source and target vocabularies, and pads them for batch processing. The TransliterationDataset encodes each word pair, while the collate_fn ensures proper padding during batching. Dataloaders are created for training and validation with appropriate batch sizes.
"""
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

# Attention
class Attention(nn.Module):
    """
    A simple attention mechanism for the Seq2Seq model.
    Args:
        enc_hidden_dim (int): The hidden dimension of the encoder.
        dec_hidden_dim (int): The hidden dimension of the decoder.
    """
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
        attn_weights = torch.bmm(v, energy).squeeze(1)
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, -1e10)
        return F.softmax(attn_weights, dim=1)

# Seq2Seq with Attention
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
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        rnn_input = torch.cat((embedded, context), dim=2)
        output, hidden = self.decoder(rnn_input, hidden)
        logits = self.fc(torch.cat((output.squeeze(1), context.squeeze(1)), dim=1))
        return logits, hidden

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        batch_size, tgt_len = tgt.shape
        encoder_outputs, hidden = self.encode(src)
        input_token = tgt[:, 0].unsqueeze(1)
        outputs = []

        for t in range(1, tgt_len):
            output, hidden = self.decode_step(input_token, hidden, encoder_outputs)
            outputs.append(output.unsqueeze(1))
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1).unsqueeze(1)
            input_token = tgt[:, t].unsqueeze(1) if teacher_force else top1
        return torch.cat(outputs, dim=1)

# Training
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

# Beam Decoding
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
                    out, h_new = model.decode_step(input_token, h, enc_out_b)
                    log_probs = F.log_softmax(out, dim=1)
                    topk_probs, topk_idxs = torch.topk(log_probs, beam_size, dim=1)
                    for i in range(beam_size):
                        next_seq = seq + [topk_idxs[0][i].item()]
                        new_score = score + topk_probs[0][i].item()
                        new_beams.append((next_seq, new_score, h_new))
                beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]
            final_outputs.append(beams[0][0])
        return final_outputs

def evaluate_beam(model, dataloader, beam_size):
    model.eval()
    total_seq, correct_seq = 0, 0
    total_tokens, correct_tokens = 0, 0

    with torch.no_grad():
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)
            preds = beam_decode(model, src, beam_size)
            for pred, true in zip(preds, tgt):
                pred_trimmed = [tok for tok in pred[1:] if tok != model.pad_idx and tok != model.eos_idx]
                true_trimmed = [tok.item() for tok in true[1:] if tok.item() != model.pad_idx and tok.item() != model.eos_idx]

                if pred_trimmed == true_trimmed:
                    correct_seq += 1
                total_seq += 1

                for p, t in zip(pred_trimmed, true_trimmed):
                    if p == t:
                        correct_tokens += 1
                total_tokens += len(true_trimmed)

    return correct_seq / total_seq, correct_tokens / total_tokens

# W&B Sweep Training
def sweep_train(config=None):
    with wandb.init(config=config):
        config = wandb.config
        run_name = f"embed{config.embed_size}_hid{config.hidden_size}_enc{config.enc_layers}_dec{config.dec_layers}_{config.cell}_drop{config.dropout}_beam{config.beam_size}"
        wandb.run.name = run_name

        model = Seq2Seq(config, len(src_vocab), len(tgt_vocab)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab.pad_idx)

        for epoch in range(10):
            train_loss = train(model, train_loader, optimizer, criterion)
            acc, token_acc = evaluate_beam(model, dev_loader, beam_size=config.beam_size)

            wandb.log({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_accuracy': acc,
                'val_token_accuracy': token_acc,
                'used_attention':config.used_attention
            })
            print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Seq Acc = {acc:.4f}, Token Acc = {token_acc:.4f}")

# Sweep Config
sweep_config = {
    'method': 'bayes',
    'metric': {'name': 'val_accuracy', 'goal': 'maximize'},
    'parameters': {
        'embed_size': {'values': [128, 256]},
        'hidden_size': {'values': [128]},
        'enc_layers': {'values': [1, 2]},
        'dec_layers': {'values': [1]},
        'dropout': {'values': [0.25, 0.3, 0.35]},
        'cell': {'values': ['GRU','LSTM']},
        'beam_size': {'values': [3, 5]},
        'used_attention':{'values':[True]}
    }
}

sweep_id = wandb.sweep(sweep_config, project="A3_ce21b020")
wandb.agent(sweep_id, function=sweep_train, count=10)
