# QText: A Model-Free Reinforcement Approach for Text Data Augmentation

## Overview
QText is an advanced text data augmentation framework that leverages reinforcement learning principles to optimize augmentation strategies for text classification tasks. It integrates various augmentation techniques with a Q-learning-based agent that dynamically selects the most effective transformations, ensuring improved performance on text classification models.

## Features
- **Multiple Augmentation Techniques**: Supports word-level, character-level, and sentence-level augmentations, including deletion, swapping, spelling correction, and abstractive summarization.
- **Reinforcement Learning (Q-learning)**: Optimizes augmentation strategies based on model performance feedback.
- **Support for Multiple Datasets**: Can be applied to SMS spam detection, news classification, AI-generated text, and email spam filtering.
- **FastAPI Integration**: Provides an API for easy interaction and training.
- **Automatic Evaluation Metrics**: Computes accuracy, precision, recall, and F1-score after training.

## Installation
```sh
# Clone the repository
git clone https://github.com/yourusername/QText.git
cd QText

# Install dependencies
pip install -r requirements.txt
```

## Dependencies
- Python 3.x
- pandas
- nlpaug
- scikit-learn
- numpy
- fastapi
- torch
- transformers
- uvicorn

## Usage
### Running the API
```sh
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

### Training a Model with Data Augmentation
Send a POST request to the API with the desired augmentation methods:
```sh
curl -X 'POST' \
  'http://127.0.0.1:8000/model' \
  -H 'Content-Type: application/json' \
  -d '{"deletion": true, "swap": true, "spelling": false, "insert": true}'
```

### Available Endpoints
| Endpoint       | Description |
|---------------|-------------|
| `/`           | Returns a status message indicating the service is running. |
| `/model`      | Trains a model using the SMS spam dataset with selected augmentations. |
| `/model2`     | Trains a model using the news classification dataset. |
| `/model3`     | Trains a model using AI-generated text data. |
| `/model4`     | Trains a model using email spam data. |

## Augmentation Methods
- **Word-Level Augmentations**:
  - Deletion (`delete`)
  - Swapping (`swap`)
  - Cropping (`crop`)
  - TF-IDF-based insertion (`tfidf`)
  - Reserved word augmentation (`reserved`)
- **Character-Level Augmentations**:
  - Random character insertion (`insert`)
  - Substitution (`substitute`)
  - Deletion (`delete`)
- **Sentence-Level Augmentation**:
  - Abstractive summarization (`abstractive`)

## How Q-Learning Works
1. **Action Selection**: The agent selects an augmentation technique either randomly (exploration) or based on learned Q-values (exploitation).
2. **Model Training**: A classifier is trained on the augmented dataset.
3. **Performance Evaluation**: Accuracy and other metrics are computed.
4. **Q-Value Update**: The reinforcement learning agent updates its augmentation strategy based on model performance.
5. **Epsilon Decay**: The exploration rate decreases over time to favor exploitation.

## Example Workflow
1. Load a dataset (e.g., `sms_data` for spam detection).
2. Select augmentation techniques.
3. Train the model using Q-learning to determine the best augmentation strategy.
4. Evaluate performance and save augmented data.
5. Deploy the API for interactive augmentation and model training.

## Contributing
Contributions are welcome! If you find a bug or have an enhancement request, please open an issue or submit a pull request.

## License
This project is licensed under the MIT License.

## Acknowledgments
- Inspired by reinforcement learning techniques applied to data augmentation.
- Uses `nlpaug` for text augmentation.
- Built with FastAPI for API deployment.

---
**Author**: Savinu Hasalanka  

