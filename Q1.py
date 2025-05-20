import argparse
import torch
import torch.nn as nn

class Seq2Seq(nn.Module):
    def __init__(self,
                 input_vocab_size,
                 target_vocab_size,
                 embedding_dim=256,
                 hidden_dim=512,
                 num_layers=1,
                 rnn_type='LSTM'):
        super(Seq2Seq, self).__init__()

        self.rnn_type = rnn_type.upper()
        assert self.rnn_type in ['RNN', 'LSTM', 'GRU'], "Invalid RNN type"

        self.embedding = nn.Embedding(input_vocab_size, embedding_dim)
        self.target_embedding = nn.Embedding(target_vocab_size, embedding_dim)

        self.encoder_rnn = getattr(nn, self.rnn_type)(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        self.decoder_rnn = getattr(nn, self.rnn_type)(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        self.output_projection = nn.Linear(hidden_dim, target_vocab_size)

    def forward(self, src, tgt):
        embedded_src = self.embedding(src)
        encoder_outputs, hidden = self.encoder_rnn(embedded_src)

        embedded_tgt = self.target_embedding(tgt)
        decoder_outputs, _ = self.decoder_rnn(embedded_tgt, hidden)

        logits = self.output_projection(decoder_outputs)
        return logits

def parse_args():
    parser = argparse.ArgumentParser(description="Train a Seq2Seq model for Latin to Devanagari character translation")
    parser.add_argument("--embedding_dim", type=int, default=256, help="Dimension of character embeddings")
    parser.add_argument("--hidden_dim", type=int, default=512, help="Dimension of RNN hidden states")
    parser.add_argument("--num_layers", type=int, default=1, help="Number of layers in encoder and decoder")
    parser.add_argument("--rnn_type", type=str, default="LSTM", choices=["RNN", "LSTM", "GRU"], help="Type of RNN cell")
    parser.add_argument("--input_vocab_size", type=int, default=50, help="Vocabulary size for input characters (Latin)")
    parser.add_argument("--target_vocab_size", type=int, default=60, help="Vocabulary size for output characters (Devanagari)")
    return parser.parse_args()

def main():
    args = parse_args()

    # Dummy input
    batch_size = 4
    src_seq_len = 10
    tgt_seq_len = 12

    src = torch.randint(0, args.input_vocab_size, (batch_size, src_seq_len))
    tgt = torch.randint(0, args.target_vocab_size, (batch_size, tgt_seq_len))

    # Instantiate model
    model = Seq2Seq(
        input_vocab_size=args.input_vocab_size,
        target_vocab_size=args.target_vocab_size,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers, 
        rnn_type=args.rnn_type
    )

    # Forward pass
    output = model(src, tgt)
    print("Output shape:", output.shape)

if __name__ == "__main__":
    main()
