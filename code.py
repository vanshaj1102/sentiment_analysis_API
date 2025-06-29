from flask import Flask, request, jsonify
from flask_cors import CORS
import re
import string
from collections import Counter
import math

app = Flask(__name__)
CORS(app)

# Positive and negative word lists (simplified version)
POSITIVE_WORDS = {
    'amazing', 'awesome', 'brilliant', 'excellent', 'fantastic', 'good', 'great', 
    'happy', 'incredible', 'love', 'outstanding', 'perfect', 'pleased', 'positive',
    'remarkable', 'satisfied', 'superb', 'wonderful', 'best', 'beautiful', 'nice',
    'glad', 'excited', 'thrilled', 'delighted', 'enjoy', 'impressed', 'magnificent',
    'marvelous', 'phenomenal', 'spectacular', 'tremendous', 'fabulous', 'terrific',
    'splendid', 'sublime', 'divine', 'blissful', 'ecstatic', 'elated', 'jubilant'
}

NEGATIVE_WORDS = {
    'awful', 'bad', 'terrible', 'horrible', 'disgusting', 'hate', 'dislike',
    'disappointed', 'frustrated', 'angry', 'sad', 'depressed', 'annoyed', 'upset',
    'furious', 'disgusted', 'outraged', 'irritated', 'bothered', 'concerned',
    'worried', 'anxious', 'stressed', 'miserable', 'unhappy', 'gloomy', 'bitter',
    'resentful', 'hostile', 'aggressive', 'nasty', 'cruel', 'harsh', 'rude',
    'offensive', 'disturbing', 'shocking', 'appalling', 'dreadful', 'ghastly'
}

# Intensifiers that modify sentiment strength
INTENSIFIERS = {
    'very': 1.5, 'extremely': 2.0, 'incredibly': 1.8, 'absolutely': 1.7,
    'completely': 1.6, 'totally': 1.5, 'really': 1.3, 'quite': 1.2,
    'pretty': 1.1, 'rather': 1.1, 'somewhat': 0.8, 'slightly': 0.7
}

# Negation words that flip sentiment
NEGATION_WORDS = {
    'not', 'no', 'never', 'nothing', 'nowhere', 'neither', 'nobody', 'none',
    'hardly', 'scarcely', 'barely', 'seldom', 'rarely', "don't", "doesn't",
    "won't", "wouldn't", "shouldn't", "couldn't", "can't", "isn't", "aren't"
}

class SentimentAnalyzer:
    def __init__(self):
        self.positive_words = POSITIVE_WORDS
        self.negative_words = NEGATIVE_WORDS
        self.intensifiers = INTENSIFIERS
        self.negation_words = NEGATION_WORDS
    
    def preprocess_text(self, text):
        """Clean and preprocess the input text."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def get_sentiment_score(self, text):
        """Calculate sentiment score for the given text."""
        processed_text = self.preprocess_text(text)
        words = processed_text.split()
        
        positive_score = 0
        negative_score = 0
        total_words = len(words)
        
        i = 0
        while i < len(words):
            word = words[i]
            
            # Check for intensifiers
            intensifier = 1.0
            if i > 0 and words[i-1] in self.intensifiers:
                intensifier = self.intensifiers[words[i-1]]
            
            # Check for negation (look back up to 3 words)
            negated = False
            for j in range(max(0, i-3), i):
                if words[j] in self.negation_words:
                    negated = True
                    break
            
            # Calculate sentiment for current word
            if word in self.positive_words:
                score = 1.0 * intensifier
                if negated:
                    negative_score += score
                else:
                    positive_score += score
            elif word in self.negative_words:
                score = 1.0 * intensifier
                if negated:
                    positive_score += score
                else:
                    negative_score += score
            
            i += 1
        
        # Normalize scores
        if total_words > 0:
            positive_score = positive_score / total_words
            negative_score = negative_score / total_words
        
        # Calculate compound score
        compound_score = positive_score - negative_score
        
        return {
            'positive': round(positive_score, 4),
            'negative': round(negative_score, 4),
            'neutral': round(1 - (positive_score + negative_score), 4),
            'compound': round(compound_score, 4)
        }
    
    def classify_sentiment(self, scores):
        """Classify sentiment based on compound score."""
        compound = scores['compound']
        
        if compound >= 0.05:
            return 'positive'
        elif compound <= -0.05:
            return 'negative'
        else:
            return 'neutral'
    
    def get_confidence(self, scores):
        """Calculate confidence score based on the strength of sentiment."""
        compound = abs(scores['compound'])
        
        if compound >= 0.5:
            return 'high'
        elif compound >= 0.1:
            return 'medium'
        else:
            return 'low'

# Initialize the sentiment analyzer
analyzer = SentimentAnalyzer()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'message': 'Sentiment Analysis API is running'
    })

@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    """Analyze sentiment of provided text."""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                'error': 'Missing required field: text'
            }), 400
        
        text = data['text']
        
        if not text or not text.strip():
            return jsonify({
                'error': 'Text cannot be empty'
            }), 400
        
        # Analyze sentiment
        scores = analyzer.get_sentiment_score(text)
        sentiment = analyzer.classify_sentiment(scores)
        confidence = analyzer.get_confidence(scores)
        
        return jsonify({
            'text': text,
            'sentiment': sentiment,
            'confidence': confidence,
            'scores': scores,
            'word_count': len(text.split())
        })
    
    except Exception as e:
        return jsonify({
            'error': f'An error occurred: {str(e)}'
        }), 500

@app.route('/batch-analyze', methods=['POST'])
def batch_analyze():
    """Analyze sentiment for multiple texts."""
    try:
        data = request.get_json()
        
        if not data or 'texts' not in data:
            return jsonify({
                'error': 'Missing required field: texts (array)'
            }), 400
        
        texts = data['texts']
        
        if not isinstance(texts, list):
            return jsonify({
                'error': 'texts must be an array'
            }), 400
        
        if len(texts) > 100:
            return jsonify({
                'error': 'Maximum 100 texts allowed per batch'
            }), 400
        
        results = []
        
        for i, text in enumerate(texts):
            if not text or not text.strip():
                results.append({
                    'index': i,
                    'error': 'Empty text'
                })
                continue
            
            scores = analyzer.get_sentiment_score(text)
            sentiment = analyzer.classify_sentiment(scores)
            confidence = analyzer.get_confidence(scores)
            
            results.append({
                'index': i,
                'text': text,
                'sentiment': sentiment,
                'confidence': confidence,
                'scores': scores,
                'word_count': len(text.split())
            })
        
        return jsonify({
            'results': results,
            'total_processed': len(results)
        })
    
    except Exception as e:
        return jsonify({
            'error': f'An error occurred: {str(e)}'
        }), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get statistics about the sentiment analyzer."""
    return jsonify({
        'positive_words_count': len(analyzer.positive_words),
        'negative_words_count': len(analyzer.negative_words),
        'intensifiers_count': len(analyzer.intensifiers),
        'negation_words_count': len(analyzer.negation_words),
        'supported_features': [
            'Text preprocessing',
            'Sentiment scoring',
            'Negation handling',
            'Intensifier detection',
            'Batch processing',
            'Confidence scoring'
        ]
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Endpoint not found'
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        'error': 'Method not allowed'
    }), 405

if __name__ == '__main__':
    print("Starting Sentiment Analysis API...")
    print("Available endpoints:")
    print("  GET  /health - Health check")
    print("  POST /analyze - Analyze single text")
    print("  POST /batch-analyze - Analyze multiple texts")
    print("  GET  /stats - Get analyzer statistics")
    print("\nExample usage:")
    print("curl -X POST http://localhost:5000/analyze \\")
    print("  -H 'Content-Type: application/json' \\")
    print("  -d '{\"text\": \"I love this product!\"}'")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
