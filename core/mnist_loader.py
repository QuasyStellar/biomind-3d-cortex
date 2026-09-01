"""
Загрузка сырых IDX-файлов MNIST с нуля (без torchvision - несовместимая
версия с текущим torch в venv при попытке установить, да и незачем: формат
IDX тривиален, парсим сами через stdlib gzip/struct, не готовый ML-фреймворк,
только формат хранения данных). Файлы - официальный публичный mirror
(ossci-datasets.s3.amazonaws.com/mnist), первое реальное сенсорное
подключение в проекте (README пункт "подключить реальные сенсорные данные
хотя бы MNIST-уровня").
"""
import gzip
import struct
import torch


def _read_idx(path):
    with gzip.open(path, "rb") as f:
        magic = struct.unpack(">I", f.read(4))[0]
        ndim = magic & 0xFF
        dims = struct.unpack(">" + "I" * ndim, f.read(4 * ndim))
        data = f.read()
    arr = torch.frombuffer(bytearray(data), dtype=torch.uint8).clone()
    return arr.view(*dims)


def load_mnist(data_dir="data/mnist"):
    train_images = _read_idx(f"{data_dir}/train-images-idx3-ubyte.gz").float() / 255.0
    train_labels = _read_idx(f"{data_dir}/train-labels-idx1-ubyte.gz").long()
    test_images = _read_idx(f"{data_dir}/t10k-images-idx3-ubyte.gz").float() / 255.0
    test_labels = _read_idx(f"{data_dir}/t10k-labels-idx1-ubyte.gz").long()
    return train_images, train_labels, test_images, test_labels
