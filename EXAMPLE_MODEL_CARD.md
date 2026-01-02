# Model Card: Email Spam Classifier

**Version:** 2.1.0  
**Released:** 2026-01-02  
**Owner:** Machine Learning Team  
**License:** Apache-2.0  

---

## Model Overview

A machine learning model that classifies emails as spam or legitimate based on textual features and header information.

**Framework:** scikit-learn

---

## Intended Use

### Primary Use Cases

- Spam detection for user inboxes

- Training data for downstream models

- Email security analytics


### Intended Users

- Email administrators

- IT security teams

- Email service providers


---

## Dataset Summary

### Training Data
**Name:** Enron Email Corpus

A large collection of real email messages with manually labeled spam and legitimate categories.



**Size:** 755000 samples


**Features:** subject_line, body_text, sender_domain, header_features

**Target Variable:** spam_label

### Evaluation Data
**Name:** Recent Email Dataset

A holdout test set from more recent emails to evaluate temporal robustness.



**Size:** 50000 samples

**Target Variable:** spam_label

---

## Performance Summary

The model was evaluated using standard classification metrics.

**Overall Accuracy:** 96.3%

| Metric | Score |
|--------|-------|
| Accuracy | 0.963 |
| Precision | 0.951 |
| Recall | 0.945 |
| F1-Score | 0.948 |

| ROC AUC | 0.985 |



### Confusion Matrix

The confusion matrix shows prediction counts across classes:

```
[[47500, 1500], [2500, 500], [1000, 45500]]
```



### Additional Metrics


- **Specificity:** 0.969

- **False Negative Rate:** 0.055




---

## Metric Interpretation

Model performance is excellent.


### Key Findings


- **Class Balance** (low): Small gap (1.5%) between accuracy and F1-score suggests reasonable class balance. Metrics are reliable.

- **Overall Performance** (low): Overall performance is excellent (F1: 94.8%, Accuracy: 96.3%). Model demonstrates strong ability to identify both positive and negative cases accurately.

- **Roc Auc** (low): ROC AUC is 98.5%, indicating excellent ability to distinguish between classes across all probability thresholds.


### Recommendations











---

## Known Limitations

### Unsuitable Inputs
The model should not be used with the following types of data:


- Non-English emails

- Encrypted or binary content

- Severely truncated messages


### Environmental Constraints

Requires Python 3.10+. Typical inference time: <50ms per email.


---

## Ethical Considerations and Risks

The following risks have been identified during development and evaluation:


### Domain-Specific Performance

**Description:** Model was trained on Enron corpus (2000s) and may not capture modern spam patterns.

**Mitigation:** Retrain quarterly with recent email data. Monitor performance metrics continuously.

**Severity:** Medium


### Language Bias

**Description:** Training data contains primarily English emails. Non-English emails may have degraded performance.

**Mitigation:** Collect and evaluate performance on non-English email samples.

**Severity:** Low



---

## Out-of-Scope Uses

This model is **not** intended for the following purposes:


- Blocking legitimate user email

- Discriminatory filtering based on sender identity


---

## Model Card Information

This model card was automatically generated to provide transparency about model capabilities and limitations. For questions or additional information, contact Machine Learning Team.

**License:** Apache-2.0  
**Last Updated:** 2026-01-02