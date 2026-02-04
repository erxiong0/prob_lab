---
dataset_info:
  features:
  - name: id
    dtype: string
  - name: question
    dtype: string
  - name: question_concept
    dtype: string
  - name: choices
    sequence:
    - name: label
      dtype: string
    - name: text
      dtype: string
  - name: answerKey
    dtype: string
  - name: inputs
    dtype: string
  - name: targets
    dtype: string
  splits:
  - name: train
    num_bytes: 3875948
    num_examples: 9741
  - name: validation
    num_bytes: 480334
    num_examples: 1221
  - name: test
    num_bytes: 452620
    num_examples: 1140
  download_size: 2706083
  dataset_size: 4808902
---
# Dataset Card for "CSQA_preprocessed"

[More Information needed](https://github.com/huggingface/datasets/blob/main/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)