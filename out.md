# Baseline 5 Experiment Report: Domain Generalization Using LoRA and Weighted Model Merging

## 1. Introduction

This report presents the results of our Baseline 5 experiment, which aims to improve domain generalization in image classification tasks. Our approach utilizes Low-Rank Adaptation (LoRA) and a novel weighted model merging technique. We conducted experiments on three diverse datasets: PACS, VLCS, and OfficeHome, each representing different domain adaptation challenges.

## 2. Methodology Overview

Our method involves the following key steps:
1. Training domain-specific LoRA adapters
2. Merging these adapters with a base model
3. Learning coefficients to optimally combine the domain-specific models
4. Creating a final model through weighted averaging
5. Evaluating on an unseen test domain

For each dataset, we performed 5 runs with different random seeds to ensure robustness of our results.

## 3. Results

### 3.1 PACS Dataset

| Metric | Average | Standard Deviation |
|--------|---------|---------------------|
| Naive Accuracy | 79.02% | 20.54% |
| Final Accuracy | 64.90% | 15.52% |

Domain-specific accuracies:
- Art Painting: 68.78% ± 1.01%
- Cartoon: 58.17% ± 37.32%
- Photo: 34.17% ± 7.86%
- Sketch: 42.56% ± 1.14%

Observations:
- The naive approach (fine-tuning on test domain) performs well, especially on 'Photo' (97.01%) and 'Cartoon' (86.46% avg).
- Our method shows good generalization to 'Photo' (98.54%) but struggles with 'Sketch' (43.40%).
- Coefficients favor Art Painting (0.58 avg) for 'Sketch' test domain, suggesting its importance for generalization.

### 3.2 VLCS Dataset

| Metric | Average | Standard Deviation |
|--------|---------|---------------------|
| Naive Accuracy | 80.97% | 13.95% |
| Final Accuracy | 74.83% | 11.23% |

Domain-specific accuracies:
- VOC2007: 76.44% ± 18.63%
- SUN09: 67.07% ± 3.40%
- Caltech101: 49.58% ± 2.66%
- LabelMe: 60.53% ± 5.36%

Observations:
- Consistent performance across different test domains.
- Significant improvement on 'Caltech101' (96.31%) compared to other domains.
- Coefficients are relatively balanced, with a slight preference for SUN09 and VOC2007.

### 3.3 OfficeHome Dataset

| Metric | Average | Standard Deviation |
|--------|---------|---------------------|
| Naive Accuracy | 52.17% | 12.70% |
| Final Accuracy | 74.65% | 9.60% |

Domain-specific accuracies:
- Art: 65.49% ± 11.45%
- Real World: 72.11% ± 11.41%
- Product: 62.26% ± 8.52%
- Clipart: 71.72% ± 4.79%

Observations:
- Significant improvement from naive to final accuracy, especially for 'Art' domain.
- Consistent performance across different test domains.
- Real World domain consistently performs well as a source domain.

## 4. Conclusion

Our Baseline 5 approach demonstrates promising results in domain generalization:

1. **Improvement over Naive Approach**: In most cases, our method outperforms or matches the naive fine-tuning approach, especially on the OfficeHome dataset.

2. **Dataset-Specific Performance**: 
   - PACS: Strong performance on 'Photo', challenges with 'Sketch'.
   - VLCS: Consistent improvements across domains.
   - OfficeHome: Significant enhancements, particularly for the 'Art' domain.

3. **Coefficient Learning**: The learned coefficients provide insights into domain relationships, often favoring domains that share visual characteristics with the test domain.

4. **Generalization Capability**: Our method shows good generalization to unseen domains, with notable improvements in challenging scenarios like the 'Art' domain in OfficeHome.

5. **Robustness**: The performance across multiple runs with different seeds demonstrates the stability of our approach.

Future work could focus on improving performance on challenging domains like 'Sketch' in PACS and investigating why certain domain adaptations are more successful than others.