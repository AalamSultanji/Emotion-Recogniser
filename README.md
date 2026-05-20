# Facial Emotion Recognition CNN

A custom convolutional neural network built from scratch in PyTorch for classifying facial expressions across 5 emotion classes. This is **v1** — a clean, modular baseline designed to be extended with augmentation, scheduling, and other improvements in future iterations.

## Results (v1)

| Metric | Value |
|---|---|
| **Test Accuracy** | **71.45%** |
| **Macro F1** | **0.71** |
| **Best Val Accuracy** | 72.22% |
| Training Time | ~4 hours (RTX 4060, 50 epochs) |

### Per-class performance

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Angry | 0.58 | 0.81 | 0.68 | 1,572 |
| Fear | 0.70 | 0.42 | 0.53 | 1,452 |
| Happy | 0.94 | 0.72 | 0.81 | 2,786 |
| Sad | 0.57 | 0.77 | 0.65 | 1,877 |
| Surprise | 0.88 | 0.86 | 0.87 | 1,179 |

**Notable findings:** Surprise achieves the highest F1 (0.87) despite being the smallest class — class-weighted loss successfully countered the imbalance. Happy has near-perfect precision (0.94), reflecting how visually distinctive smiles are. Fear is the weakest class, frequently misclassified as Sad due to genuine perceptual overlap (furrowed brows, downturned expressions).

## Dataset

- **Source:** [Human Face Emotions](https://www.kaggle.com/datasets/samithsachidanandan/human-face-emotions) (Kaggle)
- **Size:** ~59,000 images across 5 classes
- **Classes:** Angry, Fear, Happy, Sad, Surprise
- **Distribution:** Mildly imbalanced (~2.2:1 ratio between largest and smallest class)

```
Happy:    18,400 images (31%)
Sad:      12,600 images (21%)
Angry:    10,100 images (17%)
Fear:      9,732 images (16%)
Surprise:  8,227 images (14%)
```

## Architecture

A 4-layer convolutional network with progressively increasing filter counts, BatchNorm regularization, and global average pooling before the classification head.

```
Input (1×128×128 grayscale)
│
├─ Conv2d(1→32, 3×3) → BatchNorm → ReLU
├─ Conv2d(32→64, 3×3) → BatchNorm → ReLU
├─ MaxPool(2×2)                              → 64×64×64
│
├─ Conv2d(64→128, 3×3) → BatchNorm → ReLU
├─ Conv2d(128→256, 3×3) → BatchNorm → ReLU
├─ MaxPool(2×2)                              → 256×32×32
│
├─ GlobalAvgPool                             → 256
├─ Linear(256→512) → ReLU → Dropout(0.5)
└─ Linear(512→5)                             → logits
```

**Design choices:**
- **3×3 kernels throughout** — better receptive field per parameter than larger kernels
- **`bias=False` on conv layers** — BatchNorm's shift parameter makes the conv bias redundant
- **Global Average Pooling** instead of Flatten — drastically reduces parameter count before dense layers (~134M params saved vs. naive flatten)
- **Dropout only in dense layers** — conv feature maps are too spatially correlated for dropout to be effective
- **Raw logits output** — softmax is applied implicitly by `CrossEntropyLoss` for numerical stability

## Data Processing

### Transforms
```
Resize(128, 128) → Grayscale(1ch) → ToTensor → Normalize(0.5, 0.5)
```

Grayscale conversion is justified because emotion is encoded in facial geometry and texture, not color. This reduces input channels 3→1 and removes irrelevant variation (lighting tone, skin tone).

### Splits
- **70% train / 15% val / 15% test**
- Splits computed via shuffled indices, with the same indices applied to two `ImageFolder` instances (one with train transforms, one with val/test transforms) to allow per-split transform pipelines
- Random seed not fixed in v1 (TODO for v2 reproducibility)

### Class Imbalance Handling
Class weights computed using the inverse-frequency formula:

$$w_c = \frac{N_{\text{train}}}{K \cdot n_c}$$

where N is total training samples, K is the number of classes, and nₒ is samples in class c. Weights are computed from the **training split only** (no leakage from val/test) and passed to `CrossEntropyLoss`.

## Training Loop

- **Optimizer:** Adam (lr=0.001, default β values)
- **Loss:** Class-weighted CrossEntropyLoss
- **Batch size:** 32
- **Epochs:** 50 (no early stopping in v1)
- **Checkpointing:** Best model saved by validation accuracy

Each epoch performs:
1. Training phase (`model.train()`) — forward, backward, optimizer step per batch
2. Validation phase (`model.eval()` + `torch.no_grad()`) — loss and accuracy on held-out val set
3. Logging — train/val loss and accuracy printed per epoch
4. Checkpoint save if val accuracy improved

Training curves showed steady train loss decrease (1.56 → 0.57), smooth train accuracy gains (28% → 79%), and moderate val accuracy gains (36% → 72%). Some val loss volatility late in training suggests the fixed learning rate is too aggressive at convergence — a learning rate scheduler is the first priority for v2.

## Project Structure

```
emotion-cnn/
├── data_loading/
│   └── dataloading.py        # Transforms, splits, DataLoaders, class weights
├── model/
│   └── CNN.py                # CNN architecture definition
├── train_test/
│   └── train_test.py         # train(), evaluate(), test() functions
├── weights/
│   └── best_model.pth        # Saved best-validation checkpoint
├── main.py                   # Orchestration: load → build → train → test
└── README.md
```

## Usage

### Requirements
```
torch
torchvision
numpy
scikit-learn
matplotlib
seaborn
```

### Training
1. Place the dataset in `dataset/Data/` with subfolders per emotion class
2. Adjust config at the top of `main.py` if needed:
   ```python
   DATADIR = 'dataset/Data'
   BATCH_SIZE = 32
   EPOCHS = 50
   LEARNING_RATE = 0.001
   NUM_CLASSES = 5
   ```
3. Run `python main.py`

The script loads data, builds the model, trains for the specified epochs, saves the best model to `weights/best_model.pth`, then evaluates on the test set with full classification report and confusion matrix.

## What's Next (v2)

The v1 model is still under-fitting (train accuracy climbing through epoch 50, val volatility late in training). Planned improvements:

1. **Data augmentation** — horizontal flips, ±10° rotation, color jitter on the training transform. Expected gain: +3-5% accuracy.
2. **Learning rate scheduler** — `ReduceLROnPlateau` on val loss, or cosine annealing. Should smooth late-training volatility.
3. **Early stopping** — patience-based, monitoring val loss.
4. **DataLoader optimizations** — `num_workers=4`, `pin_memory=True`, `persistent_workers=True`. Speed only (~2-3x faster epochs).
5. **Reproducibility** — fixed random seed for splits.
6. **Inference API** — clean `predict(image)` interface for downstream applications.
7. **Model export** — ONNX and TorchScript exports so the model can be used without a PyTorch dependency.

## Notes

- Dataset label "Suprise" is misspelled in the source — preserved as-is in code to match folder names.
- Trained on a single RTX 4060 (8GB) — batch size 32 uses well under 1GB of GPU memory, leaving headroom for larger experiments.

## Acknowledgments

Dataset by [Samith Sachidanandan on Kaggle](https://www.kaggle.com/datasets/samithsachidanandan/human-face-emotions).
