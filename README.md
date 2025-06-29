# sentiment_analysis_API
sentiment analysis API for you using Python with Flask.
### Key Features:
## Core Functionality:

# Text preprocessing (URL removal, special character handling)
Sentiment scoring with positive, negative, neutral, and compound scores
Sentiment classification (positive/negative/neutral)
Confidence levels (high/medium/low)

# Advanced Features:

Negation handling (e.g., "not good" → negative)
Intensifier detection (e.g., "very good" → stronger positive)
Batch processing for multiple texts
Word count analysis

# API Endpoints:

GET /health - Health check
POST /analyze - Analyze single text
POST /batch-analyze - Analyze multiple texts (up to 100)
GET /stats - Get analyzer statistics
