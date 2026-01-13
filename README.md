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

## Quick Start

### Prerequisites
```bash
pip install numpy pandas scikit-learn
