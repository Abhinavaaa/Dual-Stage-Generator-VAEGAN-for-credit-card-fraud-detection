# Dual-Generator VAEGAN for Credit Card Fraud Detection

This project implements a novel Dual-Generator Variational Autoencoder Generative Adversarial Network (VAEGAN) for generating synthetic credit card fraud samples to improve fraud detection performance. The system uses two competing generators with latent space interaction and stabilized training techniques.

## Table of Contents
1. [Problem Statement](#problem-statement)
2. [Architecture Overview](#architecture-overview)
3. [Key Innovations](#key-innovations)
4. [Training Process](#training-process)
5. [Results Analysis](#results-analysis)
6. [Visualizations](#visualizations)
7. [Installation & Usage](#installation--usage)
8. [Conclusion](#conclusion)

## Problem Statement

Credit card fraud detection faces significant challenges due to:
- Extreme class imbalance (typically <0.5% fraud cases)
- Constant evolution of fraud patterns
- Limited labeled fraud samples for training
- Need for high precision and recall simultaneously

Traditional oversampling methods like SMOTE often produce unrealistic samples in high-dimensional financial data spaces. Our solution generates high-quality synthetic fraud samples that preserve the complex statistical properties of real fraud patterns.

## Architecture Overview

The system consists of three core components:

### 1. Dual Generator VAE
- **Generator 1**: VAE with encoder-decoder structure (Input→128→Latent32→128→Output)
- **Generator 2**: Parallel VAE with identical structure but independent weights
- Both generators include:
  - Batch normalization layers
  - LeakyReLU activations (α=0.2)
  - Spectral normalization for stability
  - KL divergence weighting (0.005)

### 2. Stabilized Discriminator
- 3-layer neural network with spectral normalization
- Reduced capacity (64→32→1) to prevent overpowering generators
- Wasserstein loss with gradient penalty (λ=3)

### 3. Interaction Mechanisms
- Cross-reconstruction loss between generators' latent spaces
- Contrastive loss (weight=0.02) to encourage diversity
- Shared discriminator for adversarial training

## Key Innovations

### 1. **Dual-Generator Design**
- Prevents mode collapse through competition
- Generates more diverse samples than single-generator approaches
- Allows ensemble-like prediction averaging

### 2. **Stabilized Training**
- Spectral normalization in both generators and discriminator
- Reduced learning rates (5e-5 for G, 2e-5 for D)
- Gradient penalty for Lipschitz constraint
- Balanced KL divergence weighting

### 3. **Latent Space Interaction**
```python
# Cross-reconstruction implementation
z1 = generator1.reparameterize(mu1, logvar1)
z2 = generator2.reparameterize(mu2, logvar2)
recon1_from_z2 = generator1.decode(z2)
recon2_from_z1 = generator2.decode(z1)
```

### 4. **Dynamic Loss Balancing**
- VAE reconstruction loss (MSE)
- Adversarial loss (Wasserstein)
- Contrastive loss between generators
- KL divergence penalty

## Training Process

The training regimen follows this optimized schedule:

### Phase 1 - Warmup (Epochs 1-20)
- Train generators independently with emphasis on reconstruction
- Discriminator updates frozen

### Phase 2 - Adversarial (Epochs 21-80)
- Alternating training with 1:1 generator:discriminator update ratio
- Gradual introduction of contrastive loss

### Phase 3 - Refinement (Epochs 81-100)
- Reduced learning rates by factor of 5
- Increased gradient penalty weight
- Final fine-tuning

## Results Analysis

### Performance Metrics

| Synthetic Samples | Precision | Recall | F1-Score |
|-------------------|-----------|--------|----------|
| 0                 | 0.9195    | 0.8081 | 0.8602   |
| 84                | 0.9697    | 0.8496 | 0.9057   |
| 168               | 0.9808    | 0.8500 | 0.9107   |
| 337               | 0.9848    | 0.8725 | 0.9253   |
| 674               | 0.9746    | 0.9100 | 0.9412   |
| 1011              | 1.0000    | 0.9358 | 0.9668   |
| 1348              | 0.9878    | 0.9446 | 0.9657   |
| 2696              | 0.9968    | 0.9688 | 0.9826   |
| 3370              | 0.9960    | 0.9714 | 0.9835   |
| 6740              | 0.9979    | 0.9890 | 0.9934   |
| 33700             | 0.9990    | 0.9978 | 0.9984   |

### Key Findings

#### 1. **Precision-Recall Tradeoff Optimization**
- Baseline (0 synthetic samples) shows good precision (91.95%) but limited recall (80.81%)
- Adding just 84 synthetic samples boosts precision to 96.97% with minimal recall impact
- Optimal balance achieved at 1,011 samples (100% precision, 93.58% recall)

#### 2. **Scaling Effects**
- Performance improves logarithmically with sample size
- Diminishing returns observed beyond 6,740 samples
- Even extreme oversampling (33,700 samples) maintains stability

#### 3. **Fraud Detection Thresholds**
Financial institutions can select operating points based on their risk appetite:
- **High Security**: 337 samples (98.48% precision)
- **Balanced**: 1,348 samples (98.78% precision, 94.46% recall)
- **High Coverage**: 6,740 samples (98.90% recall)

## Visualizations

### Class Distribution
The extreme class imbalance (492 fraud vs 284,315 legitimate transactions) motivates our synthetic generation approach.

### Performance Curves
Performance metrics show logarithmic improvement with synthetic sample size, demonstrating the effectiveness of our approach.

## Installation & Usage

### Requirements
```bash
pip install torch torchvision
pip install numpy pandas scikit-learn
pip install matplotlib seaborn
```

### Basic Usage
```python
from dual_vaegan import DualGeneratorVAEGAN
from fraud_detector import FraudDetector

# Initialize the model
model = DualGeneratorVAEGAN(
    input_dim=30,
    latent_dim=32,
    hidden_dim=128
)

# Train the model
model.train(fraud_data, epochs=100)

# Generate synthetic samples
synthetic_samples = model.generate_samples(n_samples=1000)

# Train fraud detector with augmented data
detector = FraudDetector()
detector.train(original_data + synthetic_samples)
```

### Configuration
Key hyperparameters that can be tuned:
- `learning_rate_g`: Generator learning rate (default: 5e-5)
- `learning_rate_d`: Discriminator learning rate (default: 2e-5)
- `kl_weight`: KL divergence weight (default: 0.005)
- `contrastive_weight`: Contrastive loss weight (default: 0.02)
- `gradient_penalty_weight`: Gradient penalty weight (default: 3.0)

## Conclusion

This Dual-Generator VAEGAN demonstrates:

1. **Effectiveness**: Achieves near-perfect detection (99.9% precision, 99.8% recall) with sufficient samples
2. **Flexibility**: Allows tuning for different risk appetites
3. **Efficiency**: Requires relatively few synthetic samples for significant improvements
4. **Stability**: Maintains performance even with extreme oversampling

### Future Directions
- Temporal pattern modeling for transaction sequences
- Adaptive sample generation based on detector weaknesses
- Federated learning for privacy-preserving fraud detection
- Real-time fraud pattern adaptation

## Citation

If you use this work in your research, please cite:

```bibtex
@article{dual_vaegan_fraud,
  title={Dual-Generator VAEGAN for Credit Card Fraud Detection},
  author={[Abhinava Ram Palusa]},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or collaboration opportunities, please contact [palusaabhinavaram@gmail.com].
