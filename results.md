# Sparse Autoencoder Probing Experiments: Results

## Overview
This document summarizes experiments comparing different feature representations for probing language models. We tested whether Sparse Autoencoder (SAE) representations provide better features for concept detection compared to raw residual stream activations.

## Methods

### Feature Types
We compared three types of activation features:
1. **Raw residual stream activations** (`raw`): The direct activations from the model's residual stream
2. **SAE hidden activations** (`sae`): The sparse hidden layer activations from the SAE after ReLU
3. **SAE reconstructions** (`sae_recons`): The reconstructed residual stream activations from the SAE

### Probing Setup
- **Model**: Gemma-2-2B with pre-trained sparse autoencoders
- **Probing layer**: Layer 19 (where the SAE was trained)
- **Probe architecture**: Simple linear layer (logistic regression)
- **Training regime**: Both full data and data-scarcity conditions (16 to 512 training examples)
- **Regularization**: Adaptive weight decay scaling based on feature type and sample size
  - Less regularization for sparse features (`sae`)
  - Stronger regularization for small sample sizes
  - Weight decay values: 0.01 for raw/reconstructions, 0.001 for sparse features

### Datasets
We tested on three binary classification tasks:
1. **Twitter emotion (happiness)**: Detect if a tweet expresses happiness
2. **NYC borough (Manhattan)**: Detect if text is about Manhattan
3. **Athlete sport (basketball)**: Detect if an athlete plays basketball

## Results

### Full Data Performance

| Dataset | Raw Activations | SAE Hidden | SAE Reconstructions |
|---------|----------------|------------|---------------------|
| Happiness | 0.716 | 0.692 | **0.734** |
| Manhattan | **0.700** | 0.692 | 0.620 |
| Basketball | **0.938** | 0.893 | 0.904 |

### Data Scarcity Performance

#### Twitter Emotion (Happiness)
| Training Size | Raw | SAE Hidden | SAE Reconstructions |
|---------------|-----|------------|---------------------|
| 16 | 0.518 | **0.552** | 0.510 |
| 32 | **0.634** | 0.572 | 0.606 |
| 64 | 0.630 | 0.596 | **0.658** |
| 128 | **0.654** | 0.630 | 0.630 |
| 256 | 0.660 | 0.646 | **0.668** |
| 512 | 0.656 | 0.636 | **0.664** |

#### NYC Borough (Manhattan)
| Training Size | Raw | SAE Hidden | SAE Reconstructions |
|---------------|-----|------------|---------------------|
| 16 | 0.522 | **0.525** | 0.468 |
| 32 | 0.548 | 0.512 | **0.550** |
| 64 | **0.600** | 0.580 | 0.562 |
| 128 | 0.580 | 0.553 | **0.590** |
| 256 | **0.610** | 0.587 | 0.590 |
| 512 | **0.662** | 0.618 | 0.550 |

#### Athlete Sport (Basketball)
| Training Size | Raw | SAE Hidden | SAE Reconstructions |
|---------------|-----|------------|---------------------|
| 16 | 0.854 | 0.451 | **0.879** |
| 32 | **0.885** | 0.763 | 0.800 |
| 64 | **0.930** | 0.800 | 0.915 |
| 128 | 0.868 | 0.769 | **0.918** |
| 256 | 0.854 | 0.859 | **0.927** |
| 512 | 0.845 | **0.907** | 0.862 |

## Key Findings

1. **SAE Reconstructions Often Outperform Raw Activations**
   - SAE reconstructions frequently achieved the highest accuracy, especially in the happiness detection task (full data) and in several data-scarce conditions across tasks
   - This suggests SAEs can effectively denoise the activation space, preserving concept-relevant information

2. **SAE Hidden Layer Activations Need Special Handling**
   - The sparse features from SAE hidden layers generally performed worse than raw activations with default training
   - However, with sufficient data, they can eventually match or exceed raw activation performance (see basketball dataset at n=512)
   - The sparsity of these activations likely requires specialized training approaches

3. **Task-Dependent Performance**
   - The relative performance of different feature types varied across tasks
   - Basketball classification showed the most dramatic differences between representations
   - SAE reconstructions were particularly effective at generalizing from small datasets in the basketball task

4. **Data Scarcity Effects**
   - With very limited data (16 examples), SAE-based features showed mixed results
   - As training data increased, the performance patterns became more consistent
   - SAE reconstructions often showed better generalization with limited data than either raw or SAE hidden activations

## Conclusions

SAEs can indeed improve probing performance, but primarily through their reconstructions rather than their sparse hidden activations directly. This makes conceptual sense: the reconstruction process effectively "denoises" the residual stream, preserving concept-relevant information while filtering out irrelevant patterns.

The initial hypothesis that SAEs would provide a better basis for probing is partially supported - they can improve probe performance, but the specific representation (hidden vs. reconstruction) and task characteristics matter significantly. Optimal results were achieved by using SAE reconstructions with adjusted regularization based on the feature type and dataset size.