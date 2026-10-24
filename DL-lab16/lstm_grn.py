#Input (English) → [Encoder] → Context Vector → [Decoder] → Output (Hindi)
from langsmith.schemas import Dataset

# Split the problem:
# 1. Load Dataset
# 2. Clean / Tokensize sentence
# 3. Build vocabulary classes
# 4. Build dataset and data loader classes
# 5. Build Encoder, Decoder, Seq2seq classes
# 6. train LSTM model
# 7. Train GRu model
# 8. Compare LSTM and GRU model
# 9. Demonstrate vanishing gradient Solution
import os
import re
import time
import random
from collections import Counter
from dataclasses import dataclass

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
@dataclass
class Config:
    CSV_PATH: str = "Hindi_English_Truncated_Corpus.csv"
    # Start small. If code works, increase SUBSET_SIZE and EPOCHS.
    SUBSET_SIZE: int = 6000
    MAX_LEN: int = 15
    TRAIN_RATIO: float = 0.9
    MIN_FREQ: int = 2
    MAX_VOCAB_SIZE: int = 6000
    BATCH_SIZE: int = 32
    EMB_DIM: int = 128
    HID_DIM: int = 256
    N_LAYERS: int = 1
    DROPOUT: float = 0.2
    EPOCHS: int = 5
    LR: float = 0.001
    CLIP: float = 1.0
    TEACHER_FORCING: float = 0.5
    SEED: int = 42
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"

SPECIAL_TOKENS = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_base_dir():
    return os.path.dirname(os.path.abspath(__file__))

def resolve_path(path):
    """
    Finds CSV whether PyCharm runs from project root or current file folder.
    """
    if os.path.exists(path):
        return path
    path2 = os.path.join(get_base_dir(), path)
    if os.path.exists(path2):
        return path2
    raise FileNotFoundError(f"Could not find file: {path}")

def output_file(name):
    return os.path.join(get_base_dir(), name)

def clean_text(text):
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text

def tokenize(text):
    """
    Simple tokenizer for English and Hindi.
    Splits punctuation also.
    """
    text = str(text).lower().strip()
    text = re.sub(r"([?.!,।])", r" \1 ", text)
    text = re.sub(r"\s+", " ", text)
    return text.split()
def load_translation_dataframe(cfg):
    csv_path = resolve_path(cfg.CSV_PATH)
    print(f"Loading CSV from: {csv_path}")
    df = pd.read_csv(csv_path, encoding="utf-8", on_bad_lines="skip")
    df.columns = [c.strip() for c in df.columns]
    print("CSV columns:", list(df.columns))
    lower_cols = {c.lower(): c for c in df.columns}
    if "english_sentence" in lower_cols and "hindi_sentence" in lower_cols:
        en_col = lower_cols["english_sentence"]
        hi_col = lower_cols["hindi_sentence"]
    elif "english" in lower_cols and "hindi" in lower_cols:
        en_col = lower_cols["english"]
        hi_col = lower_cols["hindi"]
    else:
        # fallback: use last two text columns
        object_cols = [c for c in df.columns if df[c].dtype == "object"]
        if len(object_cols) < 2:
            raise ValueError("Could not identify English and Hindi columns.")
        en_col, hi_col = object_cols[-2], object_cols[-1]
    df = df[[en_col, hi_col]].copy()
    df.columns = ["en", "hi"]
    df["en"] = df["en"].apply(clean_text)
    df["hi"] = df["hi"].apply(clean_text)
    df = df.dropna()
    df = df[(df["en"] != "") & (df["hi"] != "")]
    df = df.drop_duplicates(subset=["en", "hi"])
    df["en_len"] = df["en"].apply(lambda x: len(tokenize(x)))
    df["hi_len"] = df["hi"].apply(lambda x: len(tokenize(x)))
    df = df[
        (df["en_len"] >= 1) &
        (df["hi_len"] >= 1) &
        (df["en_len"] <= cfg.MAX_LEN) &
        (df["hi_len"] <= cfg.MAX_LEN)
    ]
    if cfg.SUBSET_SIZE is not None and len(df) > cfg.SUBSET_SIZE:
        df = df.sample(n=cfg.SUBSET_SIZE, random_state=cfg.SEED)
    df = df.reset_index(drop=True)

    print(f"Usable sentence pairs: {len(df)}")
    print(df[["en", "hi"]].head())

    return df
class Vocab:
    def __init__(self, token_lists, min_freq=2, max_size=6000):
        self.freqs = Counter()
        for tokens in token_lists:
            self.freqs.update(tokens)
        self.itos = list(SPECIAL_TOKENS)
        words = [
            (word, freq)
            for word, freq in self.freqs.items()
            if freq >= min_freq and word not in SPECIAL_TOKENS
        ]
        words = sorted(words, key=lambda x: (-x[1], x[0]))
        if max_size is not None:
            words = words[:max_size - len(self.itos)]
        self.itos.extend([word for word, freq in words])
        self.stoi = {word: idx for idx, word in enumerate(self.itos)}
    def __len__(self):
        return len(self.itos)
    @property
    def pad_idx(self):
        return self.stoi[PAD_TOKEN]
    @property
    def sos_idx(self):
        return self.stoi[SOS_TOKEN]
    @property
    def eos_idx(self):
        return self.stoi[EOS_TOKEN]
    @property
    def unk_idx(self):
        return self.stoi[UNK_TOKEN]
    def encode(self, tokens, add_sos=False, add_eos=False):
        ids = []
        if add_sos:
            ids.append(self.sos_idx)

        for token in tokens:
            ids.append(self.stoi.get(token, self.unk_idx))

        if add_eos:
            ids.append(self.eos_idx)

        return ids
    def decode(self, ids, skip_special=True):
        words = []

        for idx in ids:
            idx = int(idx)
            token = self.itos[idx]

            if token == EOS_TOKEN:
                break

            if skip_special and token in SPECIAL_TOKENS:
                continue

            words.append(token)

        return " ".join(words)

class TranslationDataset(Dataset):
    def __init__(self, dataframe, src_vocab, trg_vocab):
        self.src_sentences = []
        self.trg_sentences = []
        for _, row in dataframe.iterrows():
            en_tokens = tokenize(row["en"])
            hi_tokens = tokenize(row["hi"])

            # English input: sentence + <eos>
            src_ids = src_vocab.encode(
                en_tokens,
                add_sos=False,
                add_eos=True
            )
            # Hindi output: <sos> + sentence + <eos>
            trg_ids = trg_vocab.encode(
                hi_tokens,
                add_sos=True,
                add_eos=True
            )
            self.src_sentences.append(torch.tensor(src_ids, dtype=torch.long))
            self.trg_sentences.append(torch.tensor(trg_ids, dtype=torch.long))

    def __len__(self):
        return len(self.src_sentences)

    def __getitem__(self, idx):
        return self.src_sentences[idx], self.trg_sentences[idx]


def translation_collate(batch, src_pad_idx, trg_pad_idx):
    src_batch, trg_batch = zip(*batch)

    src_batch = pad_sequence(
        src_batch,
        batch_first=True,
        padding_value=src_pad_idx
    )

    trg_batch = pad_sequence(
        trg_batch,
        batch_first=True,
        padding_value=trg_pad_idx
    )

    return src_batch, trg_batch


def prepare_data(cfg):
    df = load_translation_dataframe(cfg)

    df = df.sample(frac=1, random_state=cfg.SEED).reset_index(drop=True)

    train_size = int(len(df) * cfg.TRAIN_RATIO)

    train_df = df.iloc[:train_size].copy()
    val_df = df.iloc[train_size:].copy()

    src_token_lists = [tokenize(sentence) for sentence in train_df["en"]]
    trg_token_lists = [tokenize(sentence) for sentence in train_df["hi"]]

    src_vocab = Vocab(
        src_token_lists,
        min_freq=cfg.MIN_FREQ,
        max_size=cfg.MAX_VOCAB_SIZE
    )

    trg_vocab = Vocab(
        trg_token_lists,
        min_freq=cfg.MIN_FREQ,
        max_size=cfg.MAX_VOCAB_SIZE
    )

    print(f"English vocab size: {len(src_vocab)}")
    print(f"Hindi vocab size: {len(trg_vocab)}")

    train_dataset = TranslationDataset(train_df, src_vocab, trg_vocab)
    val_dataset = TranslationDataset(val_df, src_vocab, trg_vocab)

    collate_fn = lambda batch: translation_collate(
        batch,
        src_vocab.pad_idx,
        trg_vocab.pad_idx
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn
    )

    return train_loader, val_loader, src_vocab, trg_vocab, train_df, val_df

class Encoder(nn.Module):
    def __init__(
        self,
        input_dim,
        emb_dim,
        hid_dim,
        n_layers,
        dropout,
        rnn_type,
        pad_idx
    ):
        super().__init__()

        self.rnn_type = rnn_type.upper()

        self.embedding = nn.Embedding(
            input_dim,
            emb_dim,
            padding_idx=pad_idx
        )

        RNN = nn.LSTM if self.rnn_type == "LSTM" else nn.GRU

        self.rnn = RNN(
            emb_dim,
            hid_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))

        if self.rnn_type == "LSTM":
            outputs, (hidden, cell) = self.rnn(embedded)
            return outputs, hidden, cell
        else:
            outputs, hidden = self.rnn(embedded)
            return outputs, hidden, None

class Decoder(nn.Module):
    def __init__(
        self,
        output_dim,
        emb_dim,
        hid_dim,
        n_layers,
        dropout,
        rnn_type,
        pad_idx
    ):
        super().__init__()

        self.rnn_type = rnn_type.upper()
        self.output_dim = output_dim

        self.embedding = nn.Embedding(
            output_dim,
            emb_dim,
            padding_idx=pad_idx
        )

        RNN = nn.LSTM if self.rnn_type == "LSTM" else nn.GRU

        self.rnn = RNN(
            emb_dim,
            hid_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )

        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_token, hidden, cell):
        """
        input_token shape: [batch]
        """
        input_token = input_token.unsqueeze(1)

        embedded = self.dropout(self.embedding(input_token))

        if self.rnn_type == "LSTM":
            output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        else:
            output, hidden = self.rnn(embedded, hidden)
            cell = None

        prediction = self.fc_out(output.squeeze(1))

        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.output_dim = decoder.output_dim

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        """
        src shape: [batch, src_len]
        trg shape: [batch, trg_len]
        """

        batch_size = trg.shape[0]
        trg_len = trg.shape[1]

        outputs = torch.zeros(
            batch_size,
            trg_len,
            self.output_dim,
            device=self.device
        )

        _, hidden, cell = self.encoder(src)

        input_token = trg[:, 0]

        for t in range(1, trg_len):
            prediction, hidden, cell = self.decoder(
                input_token,
                hidden,
                cell
            )

            outputs[:, t, :] = prediction

            teacher_force = random.random() < teacher_forcing_ratio

            top1 = prediction.argmax(1)

            input_token = trg[:, t] if teacher_force else top1

        return outputs

def init_weights(module):
    for param in module.parameters(recurse=False):
        if param.dim() > 1:
            nn.init.xavier_uniform_(param)
        else:
            nn.init.zeros_(param)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(rnn_type, src_vocab, trg_vocab, cfg):
    encoder = Encoder(
        input_dim=len(src_vocab),
        emb_dim=cfg.EMB_DIM,
        hid_dim=cfg.HID_DIM,
        n_layers=cfg.N_LAYERS,
        dropout=cfg.DROPOUT,
        rnn_type=rnn_type,
        pad_idx=src_vocab.pad_idx
    )

    decoder = Decoder(
        output_dim=len(trg_vocab),
        emb_dim=cfg.EMB_DIM,
        hid_dim=cfg.HID_DIM,
        n_layers=cfg.N_LAYERS,
        dropout=cfg.DROPOUT,
        rnn_type=rnn_type,
        pad_idx=trg_vocab.pad_idx
    )

    model = Seq2Seq(encoder, decoder, cfg.DEVICE).to(cfg.DEVICE)
    model.apply(init_weights)

    return model


def train_one_epoch(model, loader, optimizer, criterion, cfg):
    model.train()

    epoch_loss = 0

    for src, trg in loader:
        src = src.to(cfg.DEVICE)
        trg = trg.to(cfg.DEVICE)

        optimizer.zero_grad()

        output = model(
            src,
            trg,
            teacher_forcing_ratio=cfg.TEACHER_FORCING
        )

        output_dim = output.shape[-1]

        output = output[:, 1:, :].reshape(-1, output_dim)
        trg = trg[:, 1:].reshape(-1)

        loss = criterion(output, trg)

        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), cfg.CLIP)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(loader)


def evaluate(model, loader, criterion, cfg):
    model.eval()

    epoch_loss = 0

    with torch.no_grad():
        for src, trg in loader:
            src = src.to(cfg.DEVICE)
            trg = trg.to(cfg.DEVICE)

            output = model(
                src,
                trg,
                teacher_forcing_ratio=0.0
            )

            output_dim = output.shape[-1]

            output = output[:, 1:, :].reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(loader)


def train_model(rnn_type, train_loader, val_loader, src_vocab, trg_vocab, cfg):
    print("\n" + "=" * 70)
    print(f"Training {rnn_type}")
    print("=" * 70)

    model = build_model(rnn_type, src_vocab, trg_vocab, cfg)

    print(f"{rnn_type} parameters: {count_parameters(model):,}")

    optimizer = optim.Adam(model.parameters(), lr=cfg.LR)

    criterion = nn.CrossEntropyLoss(ignore_index=trg_vocab.pad_idx)

    history = {
        "train": [],
        "val": [],
        "epoch_time": [],
        "params": count_parameters(model)
    }

    total_start = time.time()

    for epoch in range(1, cfg.EPOCHS + 1):
        start = time.time()

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            cfg
        )

        val_loss = evaluate(
            model,
            val_loader,
            criterion,
            cfg
        )

        end = time.time()
        epoch_seconds = end - start

        history["train"].append(train_loss)
        history["val"].append(val_loss)
        history["epoch_time"].append(epoch_seconds)

        print(
            f"Epoch {epoch:02}/{cfg.EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Time: {epoch_seconds:.1f}s"
        )

    history["total_time"] = time.time() - total_start

    checkpoint_path = output_file(f"seq2seq_{rnn_type.lower()}.pt")

    torch.save(
        {
            "model_state": model.state_dict(),
            "src_itos": src_vocab.itos,
            "trg_itos": trg_vocab.itos,
            "config": cfg.__dict__
        },
        checkpoint_path
    )

    print(f"Saved model: {checkpoint_path}")

    return model, history

def translate_sentence(sentence, model, src_vocab, trg_vocab, cfg, max_len=30):
    model.eval()

    tokens = tokenize(sentence)

    src_ids = src_vocab.encode(
        tokens,
        add_sos=False,
        add_eos=True
    )

    src_tensor = torch.tensor(
        src_ids,
        dtype=torch.long,
        device=cfg.DEVICE
    ).unsqueeze(0)

    with torch.no_grad():
        _, hidden, cell = model.encoder(src_tensor)

    input_token = torch.tensor(
        [trg_vocab.sos_idx],
        dtype=torch.long,
        device=cfg.DEVICE
    )

    generated_ids = []

    for _ in range(max_len):
        with torch.no_grad():
            prediction, hidden, cell = model.decoder(
                input_token,
                hidden,
                cell
            )

        next_id = prediction.argmax(1).item()

        if next_id == trg_vocab.eos_idx:
            break

        generated_ids.append(next_id)

        input_token = torch.tensor(
            [next_id],
            dtype=torch.long,
            device=cfg.DEVICE
        )

    return trg_vocab.decode(generated_ids)


def show_sample_translations(models, val_df, src_vocab, trg_vocab, cfg):
    print("\n" + "=" * 70)
    print("Sample Translations")
    print("=" * 70)

    samples = val_df.head(5)

    for _, row in samples.iterrows():
        en = row["en"]
        actual_hi = row["hi"]

        print(f"\nEnglish: {en}")
        print(f"Actual Hindi: {actual_hi}")

        for name, model in models.items():
            predicted = translate_sentence(
                en,
                model,
                src_vocab,
                trg_vocab,
                cfg
            )
            print(f"{name} Predicted: {predicted}")


def plot_loss_histories(histories):
    plt.figure(figsize=(10, 6))

    for model_name, hist in histories.items():
        epochs = range(1, len(hist["train"]) + 1)

        plt.plot(
            epochs,
            hist["train"],
            marker="o",
            linestyle="--",
            label=f"{model_name} Train"
        )

        plt.plot(
            epochs,
            hist["val"],
            marker="o",
            linestyle="-",
            label=f"{model_name} Val"
        )

    plt.title("LSTM vs GRU Loss Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Cross Entropy Loss")
    plt.grid(True)
    plt.legend()

    path = output_file("lab16_lstm_vs_gru_loss.png")
    plt.savefig(path, dpi=150)
    plt.close()

    print(f"\nSaved loss graph: {path}")


def print_comparison_table(histories):
    print("\n" + "=" * 70)
    print("LSTM vs GRU Comparison")
    print("=" * 70)

    print(
        f"{'Model':<10}"
        f"{'Params':>15}"
        f"{'Final Train':>15}"
        f"{'Final Val':>15}"
        f"{'Time(s)':>15}"
    )

    print("-" * 70)

    for name, hist in histories.items():
        print(
            f"{name:<10}"
            f"{hist['params']:>15,}"
            f"{hist['train'][-1]:>15.4f}"
            f"{hist['val'][-1]:>15.4f}"
            f"{hist['total_time']:>15.1f}"
        )

class ToyGradientModel(nn.Module):
    """
    Small toy model to compare gradient flow in RNN, LSTM, GRU.
    Task:
    Given a long random sequence, predict parity of first token.
    We check gradient at first token.
    """

    def __init__(self, rnn_type, vocab_size=20, emb_dim=16, hid_dim=32):
        super().__init__()

        self.rnn_type = rnn_type.upper()

        self.embedding = nn.Embedding(vocab_size, emb_dim)

        if self.rnn_type == "RNN":
            self.rnn = nn.RNN(
                emb_dim,
                hid_dim,
                batch_first=True
            )
        elif self.rnn_type == "LSTM":
            self.rnn = nn.LSTM(
                emb_dim,
                hid_dim,
                batch_first=True
            )
        else:
            self.rnn = nn.GRU(
                emb_dim,
                hid_dim,
                batch_first=True
            )

        self.fc = nn.Linear(hid_dim, 2)
    def forward(self, x, retain_emb=False):
        emb = self.embedding(x)

        if retain_emb:
            emb.retain_grad()

        outputs, _ = self.rnn(emb)

        last_output = outputs[:, -1, :]

        logits = self.fc(last_output)

        return logits, emb

def vanishing_gradient_demo(cfg):
    print("\n" + "=" * 70)
    print("Vanishing Gradient Demonstration")
    print("=" * 70)

    seq_lengths = [5, 10, 20, 30, 40, 50]
    results = {}

    criterion = nn.CrossEntropyLoss()

    for rnn_type in ["RNN", "LSTM", "GRU"]:
        model = ToyGradientModel(rnn_type).to(cfg.DEVICE)

        grad_values = []

        for length in seq_lengths:
            trial_grads = []

            for _ in range(5):
                x = torch.randint(
                    0,
                    20,
                    (64, length),
                    device=cfg.DEVICE
                )

                # target depends on first token
                y = (x[:, 0] % 2).long()

                model.zero_grad()

                logits, emb = model(x, retain_emb=True)

                loss = criterion(logits, y)

                loss.backward()

                first_token_grad = emb.grad[:, 0, :].norm(dim=1).mean().item()

                trial_grads.append(first_token_grad)
            avg_grad = sum(trial_grads) / len(trial_grads)
            grad_values.append(avg_grad)
        results[rnn_type] = grad_values
    plt.figure(figsize=(9, 6))

    for rnn_type, values in results.items():
        plt.semilogy(
            seq_lengths,
            values,
            marker="o",
            label=rnn_type
        )
    plt.title("Gradient Flow at First Token")
    plt.xlabel("Sequence Length")
    plt.ylabel("Gradient Norm at First Token")
    plt.grid(True, which="both")
    plt.legend()
    path = output_file("lab16_vanishing_gradient_demo.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print("Gradient values:")
    for name, values in results.items():
        print(name, values)
    print(f"Saved vanishing gradient graph: {path}")
    print("\nExplanation:")
    print("Vanilla RNN gradients usually become very small for long sequences.")
    print("LSTM uses input, forget, and output gates with a cell state.")
    print("GRU uses update and reset gates.")
    print("These gates help gradients flow through many time steps.")

def main():
    cfg = Config()
    set_seed(cfg.SEED)
    print(f"Using device: {cfg.DEVICE}")
    train_loader, val_loader, src_vocab, trg_vocab, train_df, val_df = prepare_data(cfg)
    histories = {}
    models = {}

    for rnn_type in ["LSTM", "GRU"]:
        model, history = train_model(
            rnn_type,
            train_loader,
            val_loader,
            src_vocab,
            trg_vocab,
            cfg
        )

        histories[rnn_type] = history
        models[rnn_type] = model
    print_comparison_table(histories)
    plot_loss_histories(histories)
    show_sample_translations(
        models,
        val_df,
        src_vocab,
        trg_vocab,
        cfg
    )
    vanishing_gradient_demo(cfg)

if __name__ == "__main__":
    main()

