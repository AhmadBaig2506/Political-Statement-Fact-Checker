# Political Statement Fact Checker
A machine learning system that classifies political statements as true or false using the LIAR dataset from PolitiFact. The project demonstrates text classification, feature engineering and innovative abstention strategy for high confidence predictions. 

## Project Overview
This project tackles the issues of automated fact checking using Natural Language Processing and Machine Learning. It includes: 
* 6 class classification (true, mostly true, half true, barely true, false, pants fire)
* Binary classification (true-ish vs false-ish) with 64% accuracy
* Abstention strategy taht acheives 80% accuracy on 24% of examples by only predicting when confident
* Speaker based evaluation (To prevent data leakage) 

### Key Results

| Model | Accuracy | F1 Score | Coverage |
|-------|----------|----------|----------|
| 6-Class (LinearSVC) | 26.4% | 0.261 | 100% |
| Binary (Logistic Reg) | 64.3% | 0.660 | 100% |
| Binary + Abstention (t=0.70) | 79.9% | 0.826 | 24% |
| Speaker-Split (6-class) | 23.8% | 0.229 | 100% |

## Dataset
https://sites.cs.ucsb.edu/~william/papers/acl2017.pdf

## Methodology 

### 1. Feature Engineering 
Combines statement text with metadata using special markers:
**Example:**
"Healthcare costs are rising [SUBJECT] healthcare [CONTEXT] press conference [SPEAKER] John smith [PARTY] democrat"

### 2. TF-IDF Vectorization
Converts text to 30,000 dimensional numerical feature vectors. Filters words appearing in less than 2 documents as well as words appearing in 90% of the documents (eg. "the", "a" and etc). 

### Classification Models
***6 Class Model*** 
* Uses one vs rest strategy
* Acheives 26.4% test accuracy
* Shows difficulty of fine grained classification

***Binary Model***
* Achieves 64.3% test accuracy
* Balanced class weights handle imbalance
* Outputs probability estimiates

### 4. Abstention Strategy 
The model abstains when probability is between 0.3 and 0.7. Basically if:
* P(true) >= 0.7 -> TRUE
* P(true) <= 0.3 -> FALSE
* 0.3 < P < 0.7 -> ABSTAIN

**Threshold Analysis (Validation set)**
| Threshold | Coverage | Accuracy | F1 Score |
|-----------|----------|----------|----------|
| 0.55 | 77% | 0.682 | 0.690 |
| 0.60 | 56% | 0.711 | 0.722 |
| 0.65 | 38% | 0.753 | 0.759 |
| **0.70** | **24%** | **0.806** | **0.804** |
| 0.75 | 14% | 0.835 | 0.824 |
| 0.80 | 7% | 0.915 | 0.907 |

**Trade-off**: Higher threshold = fewer predictions but higher accuracy

### 5. Speaker based evaluation
To prevent data leakage and test real-world generalization:
```python
GroupShuffleSplit(test_size=0.2, random_state=42)
```

- Ensures no speaker appears in both train and test sets
- 2,328 unique speakers in train, 583 in holdout
- Results: 23.8% accuracy (vs 26.4% standard split)
- Reveals model partially relies on speaker patterns

## Results & Analysis

### Confusion Matrix (6-Class, Validation Set)
```
              b-t   fal   h-t   m-t   p-f   true
barely-true    50    54    61    31    12    29
false          48    72    48    34    28    33
half-true      43    51    73    43     7    31
mostly-true    35    31    57    66     8    54
pants-fire     16    29    10     8    41    12
true           16    28    39    42     1    43
```

**Observations:**
- Most confusion between adjacent categories
- "barely-true" often confused with "half-true" 
- "mostly-true" often confused with "true"
- Shows difficulty of fine-grained truthfulness classification

### Label Distribution
```
half-true      2,114 (20.6%)
false          1,995 (19.5%)
mostly-true    1,962 (19.2%)
true           1,676 (16.4%)
barely-true    1,654 (16.2%)
pants-fire       839 (8.2%)
```

Relatively balanced dataset with slight under-representation of extreme lies.
