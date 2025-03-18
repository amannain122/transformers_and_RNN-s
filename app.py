
from flask import Flask, render_template, request
import torch
from tensorflow.keras.models import load_model
from transformers import pipeline
import re
import emoji
import contractions
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import torch.nn as nn
from lime.lime_text import LimeTextExplainer

explainer = LimeTextExplainer(class_names=["Negative", "Neutral", "Positive"])

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')


class BiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.3):
        super(BiGRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers,
                          batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        gru_out, _ = self.gru(x)  # Don't unsqueeze here
        out = self.dropout(gru_out[:, -1, :])
        out = self.fc(out)
        return out


# Model Parameters
input_size = 100  # Embedding size (Word2Vec)
hidden_size = 128
output_size = 3  # 3 classes (Positive, Neutral, Negative)
num_layers = 1

# Initialize GRU Model
gru_model = BiGRU(input_size, hidden_size, output_size, num_layers)
gru_model.load_state_dict(torch.load("bidirectional_gru_model.pth"))
gru_model.eval()


# Sentiment analysis pipeline (for interpretability)
sentiment_pipeline = pipeline("sentiment-analysis")

app = Flask(__name__, template_folder="templates")
app.config['WTF_CSRF_ENABLED'] = True


def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#[A-Za-z0-9]+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text


def handle_emojis(text):
    return emoji.demojize(text)


def expand_contractions(text):
    return contractions.fix(text)


slang_dict = {"u": "you", "ur": "your", "idk": "i don't know",
              "btw": "by the way", "smh": "shaking my head"}


def replace_slang(text):
    words = text.split()
    return ' '.join([slang_dict[word] if word in slang_dict else word for word in words])


stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word)
              for word in tokens if word not in stop_words]
    return ' '.join(tokens)


def preprocess_input(text):
    text = clean_text(text)
    text = handle_emojis(text)
    text = expand_contractions(text)
    text = replace_slang(text)
    text = preprocess_text(text)
    return text


# Tokenizer setup (make sure it's the same tokenizer you used during training)
tokenizer = Tokenizer(num_words=5000)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']

    if not text:
        return render_template('index.html', error="Please enter some text!")

    # Preprocess the input text
    processed_text = preprocess_input(text)

    # Tokenization and Padding
    processed_seq = tokenizer.texts_to_sequences([processed_text])
    padded_seq = pad_sequences(processed_seq, maxlen=50)

    # Convert to PyTorch tensor with float32 dtype for GRU
    gru_input = torch.tensor(padded_seq, dtype=torch.float32)
    gru_input = gru_input.unsqueeze(-1)  # Add feature dimension
    gru_input = gru_input.expand(-1, -1, 100)  # Match input_size=100

    # Get prediction from GRU model
    gru_pred = gru_model(gru_input)
    gru_pred_class = torch.argmax(gru_pred, dim=1).item()

    # Get sentiment analysis result
    sentiment = sentiment_pipeline(text)[0]
    sentiment_label = sentiment['label']
    sentiment_score = sentiment['score']

    # LIME explanation

    def predict_fn(texts):
        # Prepare the texts for the model input
        processed_texts = [preprocess_input(t) for t in texts]
        processed_seq = tokenizer.texts_to_sequences(processed_texts)
        padded_seq = pad_sequences(processed_seq, maxlen=50)
        gru_input = torch.tensor(padded_seq, dtype=torch.float32)
        gru_input = gru_input.unsqueeze(-1)
        gru_input = gru_input.expand(-1, -1, 100)  # Match input_size=100
        gru_pred = gru_model(gru_input)
        return gru_pred.detach().numpy()

    # Generate explanation using LIME
    explanation = explainer.explain_instance(
        processed_text, predict_fn, num_features=10)

    # Render the explanation as a HTML format
    explanation_html = explanation.as_html()

    return render_template('index.html',
                           text=text,
                           gru_result=gru_pred_class,
                           sentiment_label=sentiment_label,
                           sentiment_score=sentiment_score,
                           explanation_html=explanation_html)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
