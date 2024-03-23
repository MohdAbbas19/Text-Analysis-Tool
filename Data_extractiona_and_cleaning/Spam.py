from flask import Flask, render_template, request, send_file, session
import pandas as pd 
import nltk
import re , string
import os
from textstat import syllable_count
from nltk.sentiment import SentimentIntensityAnalyzer
from tempfile import NamedTemporaryFile


app = Flask(__name__)



nltk.download('vader_lexicon')
def process_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove square brackets
    text = re.sub('\[.\]', '', text)
    # Remove non-word characters (symbols, punctuation)
    text = re.sub("\\W", " ", text)
    # Remove URLs
    text = re.sub('https?://\S+|www\.\S+', '', text)
    # Remove HTML tags
    text = re.sub('<.*?>+', '', text)
    # Remove punctuation
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    # Remove newline characters
    text = re.sub('\n', '', text)
    # Remove words containing numbers
    text = re.sub('\w*\d\w*', '', text)
    return text

def compute_variables(text):
    try:
        sentences = nltk.sent_tokenize(text)
        words = nltk.word_tokenize(text)
        
        sia = SentimentIntensityAnalyzer()
        sentiment_scores = sia.polarity_scores(text)
        
        avg_sentence_length = sum(len(sent.split()) for sent in sentences) / len(sentences) if len(sentences) > 0 else 0
        num_complex_words = sum(1 for word in words if syllable_count(word) > 2)
        percentage_complex_words = (num_complex_words / len(words)) * 100 if len(words) > 0 else 0
        fog_index = 0.4 * (avg_sentence_length + percentage_complex_words)
        avg_words_per_sentence = len(words) / len(sentences) if len(sentences) > 0 else 0
        complex_word_count = num_complex_words
        word_count = len(words)
        syllable_per_word = sum(syllable_count(word) for word in words) / len(words) if len(words) > 0 else 0
        personal_pronouns = ['i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours']
        personal_pronoun_count = sum(1 for word in words if word.lower() in personal_pronouns)
        avg_word_length = sum(len(word) for word in words) / len(words) if len(words) > 0 else 0
        
        return [sentiment_scores['pos'], sentiment_scores['neg'], sentiment_scores['compound'], sentiment_scores['compound'],
                avg_sentence_length, percentage_complex_words, fog_index, avg_words_per_sentence,
                complex_word_count, word_count, syllable_per_word, personal_pronoun_count, avg_word_length]
    except Exception as e:
        print(f"Error computing variables: {e}")
        return [0] * 13  # Return zeros if computation fails

def get_column_names(df):
    try:
        
        return df.columns.tolist()
    except Exception as e:
        print(f"Error reading file: {e}")
        return []

@app.route('/')
def home():
    return render_template('home.html')
@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if request.method == 'POST':
            excel_file = request.files['excel_file']
            if excel_file:
                df = pd.read_csv(excel_file)
                
                column_names = get_column_names(df)
                df.to_csv('uploaded.csv', index=False)
                return render_template('home.html', column_names=column_names)
            else:
                return "No file part"
    except Exception as e:
        return str(e)

@app.route('/analyze', methods=['POST'])
def get_data():
    try:
        if request.method == 'POST':
            selected_column = request.form['selected_column']
            df = pd.read_csv('uploaded.csv')
            if selected_column:
               # df = pd.read_csv(excel_file)
                df[selected_column] = df[selected_column].apply(process_text).apply(pd.Series)
                df[['POSITIVE SCORE', 'NEGATIVE SCORE', 'POLARITY SCORE', 'SUBJECTIVITY SCORE',
                     'AVG SENTENCE LENGTH', 'PERCENTAGE OF COMPLEX WORDS', 'FOG INDEX',
                     'AVG NUMBER OF WORDS PER SENTENCE', 'COMPLEX WORD COUNT', 'WORD COUNT',
                     'SYLLABLE PER WORD', 'PERSONAL PRONOUNS', 'AVG WORD LENGTH']] = df[selected_column].apply(compute_variables).apply(pd.Series)
                df.to_csv('processed_data.csv', index=False)
                result = df.head(10)  # Just an example; replace with your processing logic
                return render_template('home.html', data=result.to_html())
                 
            else:
                return "No file part"
    except Exception as e:
        return str(e)

@app.route('/download')
def download_file():
    try:
        # Assume df is the DataFrame containing processed data
        df = pd.read_csv('processed_data.csv')
        return send_file('processed_data.csv', as_attachment=True)
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)
