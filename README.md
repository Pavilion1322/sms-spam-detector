# SMS Spam Detection with Logistic Regression and a Multilayer Perceptron Using TF-IDF

Final project for the course **Intelligent Systems** (Sistemas Inteligentes), Universidad de Ibagué.

**Authors:** Christian Cuellar, Daniel Reyes, Javier Galeano Zapata
**Professor:** Dr. Alexandra La Cruz

An intelligent agent that reads an SMS, decides whether it is spam or a legitimate message (ham), and acts on the answer: spam is flagged and ham is delivered to the inbox. The model is TF-IDF + Logistic Regression, served by a small Flask API and orchestrated with an n8n workflow. We also compare it with a multilayer perceptron written from scratch in NumPy.

## Results (test set without duplicates, 1,021 messages)

| Model | Accuracy | Precision | Recall | F1 (spam) |
|---|---|---|---|---|
| Logistic Regression | 0.983 | 0.933 | 0.925 | 0.929 |
| MLP (NumPy, 64 hidden units) | 0.984 | 0.948 | 0.917 | 0.932 |

In 5-fold cross-validation the F1-score is 0.917 for Logistic Regression and 0.910 for the MLP, so there is no clear winner. The agent uses Logistic Regression because it is as good, faster and easier to explain.

## Repository contents

| Path | Description |
|---|---|
| `SMS_Spam_Detector_Final.ipynb` | Google Colab notebook with all the analysis, models and results |
| `agent/api_server.py` | Flask API that loads the model and answers `POST /predict` (port 5001) |
| `agent/train_model.py` | Trains the final model and saves `spam_model.joblib` |
| `agent/spam_model.joblib` | Trained model (TF-IDF + Logistic Regression) |
| `agent/n8n_workflow_sms_spam.json` | n8n workflow: webhook → classify → is spam? → flag / deliver → respond |
| `data/SMSSpamCollection` | SMS Spam Collection dataset (Almeida et al., 2011), 5,574 messages |
| `n8n_screenshots/` | Screenshots of the workflow and of two real executions |

## Dataset

SMS Spam Collection v.1 by T. A. Almeida and J. M. Gómez Hidalgo (UCI Machine Learning Repository). A copy is in `data/` (see `data/readme` for its license and citation).

## How to run the notebook

Open `SMS_Spam_Detector_Final.ipynb` in Google Colab, run all cells and upload `data/SMSSpamCollection` when the second cell asks for it.

## How to run the agent

```bash
pip install -r requirements.txt
cd agent
cp ../data/SMSSpamCollection .
python train_model.py      # optional, the trained model is already included
python api_server.py       # starts the API on http://localhost:5001
```

Then, in n8n (`n8n start`), import `agent/n8n_workflow_sms_spam.json` and activate it. Test it with:

```bash
curl -X POST http://localhost:5678/webhook/sms-classify \
  -H "Content-Type: application/json" \
  -d '{"text": "WINNER! You have won a 1000 cash prize. Call now to claim"}'
```

The answer contains the label (`spam` / `ham`), the spam probability and the action taken (`flagged_as_spam` or `delivered_to_inbox`).
