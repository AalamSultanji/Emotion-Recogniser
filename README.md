# Facial Emotion Recognition CNN

A custom convolutional neural network built from scratch in PyTorch for classifying facial expressions across 5 emotion classes. Reproducible end-to-end pipeline including data loading, class-weighted training, evaluation, and per-class diagnostics.

## Results (v1.0)

| Metric | Value |
|---|---|
| **Test Accuracy** | **72.76%** |
| **Macro F1** | **0.71** |
| **Weighted F1** | **0.73** |
| **Best Val Accuracy** | 71.71% (epoch 47) |
| Training Time | ~2 hours (RTX 4060, 50 epochs) |

### Per-class performance

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Angry | 0.67 | 0.64 | 0.65 | 1,521 |
| Fear | 0.63 | 0.46 | 0.53 | 1,470 |
| Happy | 0.89 | 0.85 | 0.87 | 2,795 |
| Sad | 0.56 | 0.78 | 0.65 | 1,836 |
| Surprise | 0.89 | 0.81 | 0.85 | 1,244 |

**Notable findings:**
- **Happy** achieves the highest F1 (0.87) with strong precision (0.89) — smiles are visually distinctive and consistently classified.
- **Surprise** F1 of 0.85 despite being the smallest class — class-weighted loss successfully countered the imbalance.
- **Fear** remains the weakest class (0.53 F1), frequently misclassified as Sad. This reflects genuine perceptual overlap (furrowed brows, downturned expressions) seen even in human emotion-recognition studies.
- **Angry → Sad confusion** (367 samples) is the largest individual error mode, consistent with shared facial features (tense brow, downturned mouth).

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

### Known dataset limitations

- **Multi-face images:** A subset of training images contain multiple faces with potentially different emotions, while the label applies to the whole image. This introduces label noise. Addressing this via face detection + cropping is identified as the most likely path to substantial accuracy improvements in future work.
- **Label spelling:** The "Surprise" class is misspelled as "Suprise" in the source dataset. Preserved as-is in code to match folder names.

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

**Design choices and rationale:**
- **3×3 kernels throughout** — better receptive field per parameter than larger kernels; two stacked 3×3 layers have the same receptive field as a single 5×5 with fewer parameters and an additional non-linearity.
- **`bias=False` on conv layers** — BatchNorm's shift parameter (β) makes the conv bias redundant.
- **Global Average Pooling instead of Flatten** — drastically reduces parameter count before dense layers (~134M parameters saved vs. naive flatten).
- **Dropout only in dense layers** — conv feature maps are too spatially correlated for dropout to be effective; dropout earns its keep on dense representations.
- **Raw logits output** — softmax is applied implicitly by `CrossEntropyLoss` for numerical stability.
- **Conv → BN → ReLU order** — canonical ordering with the strongest theoretical justification.

## Data Processing

### Transforms
```
Resize(128, 128) → Grayscale(1ch) → ToTensor → Normalize(0.5, 0.5)
```

**Why grayscale:** Emotion is encoded in facial geometry and texture, not color. This reduces input channels 3→1 and removes irrelevant variation (lighting tone, skin tone), making the model more robust and parameter-efficient.

**Why 128×128:** Balance between detail preservation (subtle facial features matter for emotion) and compute cost.

### Splits

- **70% train / 15% val / 15% test**
- Splits computed via shuffled indices with a fixed random seed
- Same indices applied to two `ImageFolder` instances (one with train transforms, one with val/test transforms) to allow per-split transform pipelines without data leakage
- Reproducible: `torch.manual_seed(42)`, `np.random.seed(42)`

### Class Imbalance Handling

Class weights computed using the inverse-frequency formula:

$$w_c = \frac{N_{\text{train}}}{K \cdot n_c}$$

where N is total training samples, K is the number of classes, and n_c is samples in class c. Weights are computed from the **training split only** (no leakage from val/test) and passed to `CrossEntropyLoss`.

## Training Loop

- **Optimizer:** Adam (lr=0.001, default β values)
- **Loss:** Class-weighted CrossEntropyLoss
- **Batch size:** 32
- **Epochs:** 50
- **Checkpointing:** Best model saved by validation accuracy
- **DataLoader:** `num_workers=4, pin_memory=True, persistent_workers=True` for efficient GPU utilization

Each epoch performs:
1. Training phase (`model.train()`) — forward, backward, optimizer step per batch
2. Validation phase (`model.eval()` + `torch.no_grad()`) — loss and accuracy on held-out val set
3. Logging — train/val loss and accuracy printed per epoch
4. Checkpoint save if val accuracy improved

Training showed steady train loss decrease (1.56 → 0.52), accuracy gains (28% → 81% train, 36% → 72% val peak), and a ~9% train/val gap by epoch 50 — modest overfitting that doesn't appear to harm test-set generalization. Val loss is volatile across epochs, characteristic of training with BatchNorm + Dropout on smaller validation sets.

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

### Reproducing Results

1. Download the [dataset](https://www.kaggle.com/datasets/samithsachidanandan/human-face-emotions) and place it in `dataset/Data/` with subfolders per emotion class
2. Adjust config at the top of `main.py` if needed:
   ```python
   DATADIR = 'dataset/Data'
   BATCH_SIZE = 32
   EPOCHS = 50
   LEARNING_RATE = 0.001
   NUM_CLASSES = 5
   ```
3. Run `python main.py`

The script sets random seeds, loads data, builds the model, trains for the specified epochs, saves the best model to `weights/best_model.pth`, then evaluates on the test set with full classification report and confusion matrix.

### Inference with the trained model

```python
import torch
from model.CNN import CNN

model = CNN(num_classes=5)
model.load_state_dict(torch.load('weights/best_model.pth', weights_only=True))
model.eval()

# Preprocess your image: resize 128×128, grayscale, normalize (0.5, 0.5)
# Then: logits = model(image_tensor)
# Predicted class index: logits.argmax(dim=1)
# Class names (in label order): ['Angry', 'Fear', 'Happy', 'Sad', 'Suprise']
```

## Lessons Learned

This project went through multiple iterations exploring augmentation, weight decay, and regularization strategies. Key findings from those experiments:

- **Aggressive augmentation hurt performance** on this model + dataset combination. The custom CNN's modest capacity meant that perturbations like `RandomRotation` and `ColorJitter` applied on every batch prevented the model from converging to good solutions. Test accuracy dropped 10-15% compared to no augmentation.
- **The original architecture was the right size** for the task — large enough to learn, small enough not to drown in over-regularization. Adding regularization on top of existing BatchNorm + Dropout(0.5) was unnecessary.
- **Training dynamics matter as much as architecture.** Most of the project's iteration was on the training process, not the model. The same network produced anywhere from 60% to 73% test accuracy depending on the training recipe.

## Future Work

Potential improvements identified during this project, ordered by expected impact:

1. **Face detection + cropping preprocessing** — biggest expected unlock. Removes label noise from multi-face images and aligns inputs to a consistent face-centered view. Tools like MediaPipe or MTCNN would integrate cleanly as a one-time preprocessing step.
2. **Larger model capacity** — adding a 5th conv block (256→512) or wider dense layers would give more room to fit cleaner data after preprocessing.
3. **Transfer learning baseline** — comparing the from-scratch CNN against a fine-tuned EfficientNet or ResNet would quantify the value of pretrained face/edge features for this task.
4. **Learning rate scheduler** — `ReduceLROnPlateau` on val loss or cosine annealing could smooth late-training volatility.
5. **Model export** — ONNX and TorchScript exports for use in downstream applications without PyTorch as a runtime dependency.
6. **Inference API** — a clean `predict(image)` interface wrapping preprocessing + model + label decoding.

## Acknowledgments

Dataset by [Samith Sachidanandan on Kaggle](https://www.kaggle.com/datasets/samithsachidanandan/human-face-emotions).
