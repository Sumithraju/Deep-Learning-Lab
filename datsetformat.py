
datasetformats.py

100%
'''
credit: ChatGPT

| File type              | Typical use                                          | Example                                 |
| ---------------------- | ---------------------------------------------------- | --------------------------------------- |
| **`.npz`**             | NumPy arrays, multiple arrays in one compressed file | `X`, `y`, masks, embeddings             |
| **`.npy`**             | A single NumPy array                                 | Image tensor, feature matrix            |
| **`.csv`**             | Tabular data                                         | Patient features + labels               |
| **`.tsv`**             | Tabular/text data with tabs as separators            | Genomic annotations                     |
| **`.parquet`**         | Large tabular datasets                               | Large-scale ML datasets                 |
| **`.h5` / `.hdf5`**    | Large hierarchical numerical datasets                | Images, genomics, neural-network data   |
| **`.hdf5` / `.h5`**    | Large multidimensional arrays                        | EEG, MRI, genomics                      |
| **`.pt` / `.pth`**     | PyTorch tensors/models                               | Preprocessed tensors, model checkpoints |
| **`.pkl`**             | Python objects                                       | Preprocessing objects, ML datasets      |
| **`.json`**            | Structured metadata/text                             | NLP datasets, annotations               |
| **`.jsonl`**           | One JSON object per line                             | LLM/NLP training data                   |
| **`.tfrecord`**        | TensorFlow training datasets                         | Image/text/audio pipelines              |
| **`.tfrecords`**       | Same as above                                        | Large TensorFlow datasets               |
| **`.parquet`**         | Efficient columnar storage                           | Large biomedical datasets               |
| **`.arrow`**           | Hugging Face / Apache Arrow datasets                 | NLP and multimodal datasets             |
| **`.fa` / `.fasta`**   | Biological sequences                                 | DNA/protein sequences                   |
| **`.fastq`**           | Sequencing reads + quality scores                    | Genomics                                |
| **`.vcf`**             | Genetic variants                                     | Human genomic variant datasets          |
| **`.bam` / `.cram`**   | Aligned sequencing reads                             | Genomics                                |
| **`.nii` / `.nii.gz`** | Medical 3D imaging                                   | MRI/fMRI                                |
| **`.edf`**             | Physiological signals                                | EEG                                     |
| **`.wav` / `.flac`**   | Audio                                                | Speech/audio DL                         |
| **`.jpg` / `.png`**    | Images                                               | Computer vision                         |
| **`.mp4` / `.avi`**    | Video                                                | Video DL                                |
'''

### npy
import numpy as np

X = np.load("features.npy")

print(X.shape)
print(X.dtype)

np.save("features.npy", X)

### .npz
import numpy as np

data = np.load("dataset.npz")

X = data["X"]
y = data["y"]

print(X.shape)
print(y.shape)

print(data.files)
np.savez_compressed("dataset.npz", X=X, y=y)

#### csv
import pandas as pd

df = pd.read_csv("dataset.csv")

print(df.head())
print(df.shape)


#### tsv
import pandas as pd

df = pd.read_csv("dataset.tsv", sep="\t")

print(df.head())

### parquet - large tabular datasets
import pandas as pd

df = pd.read_parquet("dataset.parquet")

print(df.head())
print(df.shape)

X = df.drop("label", axis=1).to_numpy()
y = df["label"].to_numpy()

### HDF5 — .h5 / .hdf5 - Useful for large multidimensional numerical datasets.
import h5py

with h5py.File("dataset.h5", "r") as f:
    print(list(f.keys()))

    X = f["X"][:]
    y = f["y"][:]

print(X.shape)
'''
dataset.h5
│
├── X
├── y
└── metadata
    ├── age
    └── sex
'''
### PyTorch — .pt - Usually used for saved tensors or other PyTorch objects.
import torch

data = torch.load("dataset.pt", weights_only=False)

print(data)
X = data["X"]
y = data["y"]
X = torch.load("X.pt", weights_only=True)

### PyTorch — .pth - .pth is commonly used for model checkpoints, although it can contain other PyTorch objects.
import torch

checkpoint = torch.load("model.pth", weights_only=False)

print(checkpoint.keys())
model.load_state_dict(checkpoint["model_state_dict"])

### Pickle — .pkl - Stores arbitrary Python objects.
import pickle

with open("dataset.pkl", "rb") as f:
    data = pickle.load(f)

print(data)

X = data["X"]
y = data["y"]

### JSON — .json - Useful for structured data and metadata.
import json

with open("dataset.json", "r") as f:
    data = json.load(f)

print(data)
'''
{
    "patient_id": "P001",
    "age": 12,
    "label": 1
}
'''
print(data["age"])
print(data["label"])

### JSON Lines — .jsonl - Very common in NLP/LLM datasets. - Each line is a separate JSON object.
import json

data = []

with open("dataset.jsonl", "r") as f:
    for line in f:
        data.append(json.loads(line))

print(data[0])

from datasets import load_dataset

dataset = load_dataset("json", data_files="dataset.jsonl")

print(dataset)

### TFRecord — .tfrecord - Common TensorFlow data format.
import tensorflow as tf

dataset = tf.data.TFRecordDataset("dataset.tfrecord")

for record in dataset.take(1):
    print(record)

def parse_example(record):
    features = {
        "label": tf.io.FixedLenFeature([], tf.int64),
        "value": tf.io.FixedLenFeature([], tf.float32)
    }

    return tf.io.parse_single_example(record, features)

dataset = dataset.map(parse_example)

### .arrow - Hugging Face datasets commonly use Apache Arrow internally. - Usually you don't directly load the .arrow file.
from datasets import load_from_disk

dataset = load_from_disk("my_dataset")

print(dataset)

###  OR load a hugging face dataset
from datasets import load_dataset

dataset = load_dataset("csv", data_files="dataset.csv")


### FASTA — .fa / .fasta - DNA/protein sequences. - Using Biopython:
from Bio import SeqIO

records = SeqIO.parse("sequences.fasta", "fasta")

for record in records:
    print(record.id)
    print(record.seq)

### FASTQ — .fastq / .fq - Contains sequencing reads plus quality scores.
from Bio import SeqIO

records = SeqIO.parse("reads.fastq", "fastq")

for record in records:
    print(record.id)
    print(record.seq)
    print(record.letter_annotations["phred_quality"])

### VCF — .vcf  - Contains genetic variants.- using cyvcf2:

from cyvcf2 import VCF

vcf = VCF("variants.vcf")

for variant in vcf:
    print(variant.CHROM)
    print(variant.POS)
    print(variant.REF)
    print(variant.ALT)


### BAM — .bam - Aligned sequencing reads. Using pysam:
import pysam

bam = pysam.AlignmentFile("sample.bam", "rb")

for read in bam.fetch():
    print(read.query_name)
    print(read.reference_name)
    print(read.reference_start)

### CRAM — .cram - Similar concept to BAM but more compressed.
import pysam

cram = pysam.AlignmentFile(
    "sample.cram",
    "rc",
    reference_filename="reference.fasta"
)

for read in cram.fetch():
    print(read.query_name)

### Medical imaging =  NIfTI — .nii / .nii.gz - Very common for MRI/fMRI. - Using NiBabel:
import nibabel as nib

img = nib.load("brain.nii.gz")

data = img.get_fdata()

print(data.shape)

### DICOM — .dcm - Common in clinical imaging. - Using pydicom:
import pydicom

ds = pydicom.dcmread("image.dcm")

print(ds.PatientID)
print(ds.Rows)
print(ds.Columns)

image = ds.pixel_array

### EEG / physiological signals -  EDF — .edf - Common for EEG. - Using MNE:
import mne

raw = mne.io.read_raw_edf(
    "recording.edf",
    preload=True
)

print(raw.info)
print(raw.get_data().shape)

### Audio -  WAV — .wav - using soundfile:
import soundfile as sf

audio, sample_rate = sf.read("speech.wav")

print(audio.shape)
print(sample_rate)

### FLAC — .flac - Same library:
import soundfile as sf

audio, sample_rate = sf.read("speech.flac")

print(audio.shape)
print(sample_rate)
Displaying datasetformats.py.