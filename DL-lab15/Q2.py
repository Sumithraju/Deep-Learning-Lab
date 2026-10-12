
# IMAGE CAPTIONING USING CNN + LSTM
# PyTorch
# Dataset: Flickr8k
import os
import re
from collections import Counter
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
# 1. PATHS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(
    BASE_DIR,
    "Images"
)
CAPTION_FILE = os.path.join(
    BASE_DIR,
    "captions.txt"
)
print("Image folder:", IMAGE_DIR)
print("Caption file:", CAPTION_FILE)
print(
    "Image folder exists:",
    os.path.exists(IMAGE_DIR)
)
print(
    "Caption file exists:",
    os.path.exists(CAPTION_FILE)
)
if not os.path.exists(IMAGE_DIR):
    raise FileNotFoundError(
        f"Images folder not found: {IMAGE_DIR}"
    )
if not os.path.exists(CAPTION_FILE):
    raise FileNotFoundError(
        f"captions.txt not found: {CAPTION_FILE}"
    )
print(
    "Number of image files:",
    len(os.listdir(IMAGE_DIR))
)

df = pd.read_csv(CAPTION_FILE)#load caption file

print("\nFirst 5 rows:")
print(df.head())

print("\nColumns:")
print(df.columns.tolist())

print("\nDataset shape:")
print(df.shape)
# Check required columns

if "image" not in df.columns or "caption" not in df.columns:

    raise ValueError(
        "captions.txt must contain columns named 'image' and 'caption'. "
        f"Current columns are: {df.columns.tolist()}"
    )
image_name = df.iloc[0]["image"]
caption = df.iloc[0]["caption"]
image_path = os.path.join(
    IMAGE_DIR,
    image_name
)
print("\nFirst image:")
print(image_name)

print("\nFirst caption:")
print(caption)

print("\nImage path:")
print(image_path)


# Optional display
# Uncomment these lines if you want to display first image.
"""
import matplotlib.pyplot as plt

image = Image.open(
    image_path
).convert("RGB")

plt.imshow(image)
plt.title(caption)
plt.axis("off")
plt.show()
"""

# ============================================================
# 4. VOCABULARY
# ============================================================

class Vocabulary:

    def __init__(self, freq_threshold=5):

        self.freq_threshold = freq_threshold

        self.itos = {
            0: "<PAD>",
            1: "<SOS>",
            2: "<EOS>",
            3: "<UNK>"
        }

        self.stoi = {
            "<PAD>": 0,
            "<SOS>": 1,
            "<EOS>": 2,
            "<UNK>": 3
        }


    def __len__(self):

        return len(self.itos)


    def tokenizer(self, text):

        text = str(text).lower()

        tokens = re.findall(
            r"\w+",
            text
        )

        return tokens


    def build_vocabulary(self, sentence_list):

        frequencies = Counter()

        # Count all words
        for sentence in sentence_list:

            tokens = self.tokenizer(
                sentence
            )

            frequencies.update(tokens)


        idx = 4

        for word, frequency in frequencies.items():

            if frequency >= self.freq_threshold:

                self.stoi[word] = idx

                self.itos[idx] = word

                idx += 1


    def numericalize(self, text):

        tokenized_text = self.tokenizer(
            text
        )

        numericalized = []

        for token in tokenized_text:

            if token in self.stoi:

                numericalized.append(
                    self.stoi[token]
                )

            else:

                numericalized.append(
                    self.stoi["<UNK>"]
                )

        return numericalized


# ============================================================
# 5. BUILD VOCABULARY
# ============================================================

vocab = Vocabulary(
    freq_threshold=5
)

vocab.build_vocabulary(
    df["caption"].tolist()
)

print(
    "\nVocabulary size:",
    len(vocab)
)


# Test vocabulary

test_caption = df.iloc[0]["caption"]

print("\nOriginal caption:")
print(test_caption)

print("\nTokenized:")
print(
    vocab.tokenizer(test_caption)
)

print("\nNumericalized:")
print(
    vocab.numericalize(test_caption)
)


# ============================================================
# 6. IMAGE TRANSFORMATION
# ============================================================

transform = transforms.Compose(

    [

        transforms.Resize(
            (224, 224)
        ),

        transforms.ToTensor(),

        transforms.Normalize(

            mean=[
                0.485,
                0.456,
                0.406
            ],

            std=[
                0.229,
                0.224,
                0.225
            ]

        )

    ]

)


# ============================================================
# 7. CUSTOM DATASET
# ============================================================

class FlickrDataset(Dataset):

    def __init__(
        self,
        root_dir,
        captions_file,
        vocabulary,
        transform=None
    ):

        self.root_dir = root_dir

        self.df = pd.read_csv(
            captions_file
        )

        self.transform = transform

        self.vocab = vocabulary

        self.images = self.df[
            "image"
        ].tolist()

        self.captions = self.df[
            "caption"
        ].tolist()


    def __len__(self):

        return len(self.df)


    def __getitem__(self, index):

        caption = self.captions[index]

        image_name = self.images[index]

        image_path = os.path.join(
            self.root_dir,
            image_name
        )

        image = Image.open(
            image_path
        ).convert("RGB")


        if self.transform is not None:

            image = self.transform(
                image
            )


        # Add <SOS>

        numericalized_caption = [

            self.vocab.stoi[
                "<SOS>"
            ]

        ]


        # Convert words → numbers

        numericalized_caption += (
            self.vocab.numericalize(
                caption
            )
        )


        # Add <EOS>

        numericalized_caption.append(

            self.vocab.stoi[
                "<EOS>"
            ]

        )


        numericalized_caption = torch.tensor(
            numericalized_caption,
            dtype=torch.long
        )


        return (
            image,
            numericalized_caption
        )


# ============================================================
# 8. COLLATE FUNCTION
# ============================================================

class MyCollate:

    def __init__(
        self,
        pad_idx
    ):

        self.pad_idx = pad_idx


    def __call__(
        self,
        batch
    ):

        images = []

        captions = []


        for image, caption in batch:

            images.append(
                image
            )

            captions.append(
                caption
            )


        images = torch.stack(
            images,
            dim=0
        )


        captions = pad_sequence(

            captions,

            batch_first=True,

            padding_value=self.pad_idx

        )


        return images, captions


# ============================================================
# 9. CREATE DATASET
# ============================================================

dataset = FlickrDataset(

    root_dir=IMAGE_DIR,

    captions_file=CAPTION_FILE,

    vocabulary=vocab,

    transform=transform

)


print(
    "\nDataset length:",
    len(dataset)
)


# ============================================================
# 10. CREATE TRAIN LOADER
# ============================================================

BATCH_SIZE = 32

PAD_IDX = vocab.stoi[
    "<PAD>"
]


train_loader = DataLoader(

    dataset=dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    num_workers=0,

    collate_fn=MyCollate(
        PAD_IDX
    )

)


# ============================================================
# 11. TEST TRAIN LOADER
# ============================================================

sample_images, sample_captions = next(
    iter(train_loader)
)


print(
    "\nBatch image shape:",
    sample_images.shape
)

print(
    "Batch caption shape:",
    sample_captions.shape
)


# ============================================================
# 12. CNN ENCODER
# ============================================================

class EncoderCNN(nn.Module):

    def __init__(
        self,
        embed_size
    ):

        super(
            EncoderCNN,
            self
        ).__init__()


        # Pretrained ResNet50

        resnet = models.resnet50(

            weights=models.ResNet50_Weights.DEFAULT

        )


        # Freeze ResNet parameters

        for parameter in resnet.parameters():

            parameter.requires_grad = False


        # Remove final classification layer

        modules = list(
            resnet.children()
        )[:-1]


        self.resnet = nn.Sequential(
            *modules
        )


        # ResNet50 produces 2048 features
        # Convert 2048 → embed_size

        self.fc = nn.Linear(

            resnet.fc.in_features,

            embed_size

        )


    def forward(
        self,
        images
    ):

        # Don't train ResNet backbone

        with torch.no_grad():

            features = self.resnet(
                images
            )


        # [batch, 2048, 1, 1]
        #
        # becomes
        #
        # [batch, 2048]

        features = features.reshape(

            features.size(0),

            -1

        )


        # [batch, 2048]
        #
        # becomes
        #
        # [batch, embed_size]

        features = self.fc(
            features
        )


        return features


# ============================================================
# 13. LSTM DECODER
# ============================================================

class DecoderLSTM(nn.Module):

    def __init__(
        self,
        embed_size,
        hidden_size,
        vocab_size,
        num_layers
    ):

        super(
            DecoderLSTM,
            self
        ).__init__()


        # Convert word IDs → vectors

        self.embed = nn.Embedding(

            vocab_size,

            embed_size

        )


        self.lstm = nn.LSTM(

            input_size=embed_size,

            hidden_size=hidden_size,

            num_layers=num_layers,

            batch_first=True

        )


        # Convert hidden output → vocabulary

        self.fc = nn.Linear(

            hidden_size,

            vocab_size

        )


    def forward(
        self,
        features,
        captions
    ):

        # captions:
        #
        # [batch, sequence_length]

        embeddings = self.embed(
            captions
        )


        # embeddings:
        #
        # [batch, sequence_length, embed_size]


        # CNN feature:
        #
        # [batch, embed_size]
        #
        # becomes
        #
        # [batch, 1, embed_size]

        features = features.unsqueeze(
            1
        )


        # Put image feature BEFORE caption
        #
        # IMAGE, <SOS>, word1, word2 ...

        embeddings = torch.cat(

            (
                features,
                embeddings
            ),

            dim=1

        )


        outputs, (
            hidden,
            cell
        ) = self.lstm(
            embeddings
        )


        logits = self.fc(
            outputs
        )


        return logits


    # ========================================================
    # Generate caption
    # ========================================================

    def sample(
        self,
        features,
        states=None,
        max_len=20,
        eos_idx=None
    ):

        sampled_ids = []


        # CNN feature becomes first LSTM input

        inputs = features.unsqueeze(
            1
        )


        for i in range(max_len):

            hiddens, states = self.lstm(

                inputs,

                states

            )


            outputs = self.fc(

                hiddens.squeeze(1)

            )


            predicted = outputs.argmax(
                dim=1
            )


            word_id = predicted.item()


            sampled_ids.append(
                word_id
            )


            # Stop at <EOS>

            if (
                eos_idx is not None
                and
                word_id == eos_idx
            ):

                break


            # Previous predicted word
            # becomes next input

            inputs = self.embed(
                predicted
            )


            inputs = inputs.unsqueeze(
                1
            )


        return sampled_ids


# ============================================================
# 14. COMPLETE IMAGE CAPTIONING MODEL
# ============================================================

class ImageCaptioningModel(nn.Module):

    def __init__(
        self,
        embed_size,
        hidden_size,
        vocab_size,
        num_layers
    ):

        super(
            ImageCaptioningModel,
            self
        ).__init__()


        self.encoder = EncoderCNN(
            embed_size
        )


        self.decoder = DecoderLSTM(

            embed_size,

            hidden_size,

            vocab_size,

            num_layers

        )


    def forward(
        self,
        images,
        captions
    ):

        features = self.encoder(
            images
        )


        outputs = self.decoder(

            features,

            captions

        )


        return outputs


# ============================================================
# 15. DEVICE
# ============================================================

device = torch.device(

    "cuda"

    if torch.cuda.is_available()

    else "cpu"

)


print(
    "\nUsing device:",
    device
)


# ============================================================
# 16. HYPERPARAMETERS
# ============================================================

embed_size = 256

hidden_size = 512

vocab_size = len(vocab)

num_layers = 1

learning_rate = 0.001

num_epochs = 5


print(
    "Vocabulary size used by model:",
    vocab_size
)


# ============================================================
# 17. CREATE MODEL
# ============================================================

model = ImageCaptioningModel(

    embed_size,

    hidden_size,

    vocab_size,

    num_layers

).to(device)


# ============================================================
# 18. LOSS FUNCTION
# ============================================================

criterion = nn.CrossEntropyLoss(

    ignore_index=PAD_IDX

)


# ============================================================
# 19. OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(

    model.parameters(),

    lr=learning_rate

)


# ============================================================
# 20. TEST MODEL BEFORE TRAINING
# ============================================================

images, captions = next(
    iter(train_loader)
)


images = images.to(
    device
)

captions = captions.to(
    device
)


# We don't give <EOS> as decoder input.
#
# Example input:
#
# <SOS> a dog is running
#
# target:
#
# <SOS> a dog is running <EOS>


outputs = model(

    images,

    captions[:, :-1]

)


print(
    "\nBefore training:"
)

print(
    "Images shape:",
    images.shape
)

print(
    "Captions shape:",
    captions.shape
)

print(
    "Model output shape:",
    outputs.shape
)


# ============================================================
# 21. TRAIN MODEL
# ============================================================

print(
    "\nStarting training...\n"
)


for epoch in range(num_epochs):

    model.train()

    total_loss = 0


    for batch_idx, (
        images,
        captions
    ) in enumerate(train_loader):


        # Move data to GPU/CPU

        images = images.to(
            device
        )

        captions = captions.to(
            device
        )


        # Reset gradients

        optimizer.zero_grad()


        # ----------------------------------------------------
        # Decoder input
        # ----------------------------------------------------
        #
        # Caption:
        #
        # <SOS> a dog is running <EOS>
        #
        # captions[:, :-1]:
        #
        # <SOS> a dog is running
        #
        # CNN image is automatically added
        # at the beginning by DecoderLSTM.


        outputs = model(

            images,

            captions[:, :-1]

        )


        # Targets contain full caption

        targets = captions


        # outputs:
        #
        # [batch,
        #  sequence_length,
        #  vocabulary_size]
        #
        # CrossEntropy needs:
        #
        # [batch * sequence_length,
        #  vocabulary_size]


        outputs = outputs.reshape(

            -1,

            outputs.size(2)

        )


        targets = targets.reshape(
            -1
        )


        # Calculate loss

        loss = criterion(

            outputs,

            targets

        )


        # Backpropagation

        loss.backward()


        # Update parameters

        optimizer.step()


        total_loss += loss.item()


        # Print every 100 batches

        if batch_idx % 100 == 0:

            print(

                f"Epoch "
                f"[{epoch + 1}/{num_epochs}] "

                f"Batch "
                f"[{batch_idx}/{len(train_loader)}] "

                f"Loss: "
                f"{loss.item():.4f}"

            )


    average_loss = (

        total_loss
        /
        len(train_loader)

    )


    print(

        f"\nEpoch "
        f"[{epoch + 1}/{num_epochs}] "

        f"Average Loss: "
        f"{average_loss:.4f}\n"

    )


# ============================================================
# 22. SAVE MODEL
# ============================================================

MODEL_PATH = os.path.join(

    BASE_DIR,

    "image_caption_lstm.pth"

)


torch.save(

    model.state_dict(),

    MODEL_PATH

)


print(
    "\nModel saved to:"
)

print(
    MODEL_PATH
)


# ============================================================
# 23. CAPTION GENERATION FUNCTION
# ============================================================

def generate_caption(
    image_path,
    model,
    vocabulary,
    transform,
    device
):

    model.eval()


    image = Image.open(
        image_path
    ).convert("RGB")


    image_tensor = transform(
        image
    )


    image_tensor = image_tensor.unsqueeze(
        0
    )


    image_tensor = image_tensor.to(
        device
    )


    with torch.no_grad():

        features = model.encoder(
            image_tensor
        )


        sampled_ids = model.decoder.sample(

            features,

            eos_idx=vocabulary.stoi[
                "<EOS>"
            ]

        )


    words = []


    for word_id in sampled_ids:

        word = vocabulary.itos.get(

            word_id,

            "<UNK>"

        )


        if word == "<EOS>":

            break


        if word not in [
            "<SOS>",
            "<PAD>"
        ]:

            words.append(
                word
            )


    sentence = " ".join(
        words
    )


    return sentence


# ============================================================
# 24. TEST CAPTION GENERATION
# ============================================================

test_image_name = df.iloc[0][
    "image"
]

test_image_path = os.path.join(

    IMAGE_DIR,

    test_image_name

)


predicted_caption = generate_caption(

    test_image_path,

    model,

    vocab,

    transform,

    device

)


print(
    "\nActual caption:"
)

print(
    df.iloc[0]["caption"]
)


print(
    "\nGenerated caption:"
)

print(
    predicted_caption
)