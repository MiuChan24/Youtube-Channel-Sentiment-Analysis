## 🧠 YouTube Comments Sentiment Analysis using BERT

This project analyzes the sentiment of comments from a YouTube channel using a BERT model, and displays interactive visualizations through a user-friendly web interface.

<img width="1374" height="996" alt="Screenshot 2025-07-29 151304" src="https://github.com/user-attachments/assets/0f46593a-500a-4d40-9161-72869f89dc4c" />

## 🔁 Project Workflow

<img width="1536" height="1024" alt="ChatGPT Image Jul 31, 2025, 03_42_43 PM" src="https://github.com/user-attachments/assets/2a426b87-3d37-468d-8845-525536a312ca" />


### 1. **Frontend (HTML + JS + CSS)**

* A simple and clean interface lets users input a **YouTube Channel ID**.
* Clicking “Analyze” sends the ID to the backend using a JavaScript `fetch()` call.

### 2. **Flask Backend**

* The Flask server receives the request and invokes:

  * `get_channel_video_ids()`: Fetches recent video IDs from the channel.
  * `get_comments()`: Retrieves top-level comments from each video.
  * `analyze_comments()`: Applies a **BERT-based sentiment analysis model** (with humor detection).
* The backend processes and returns:

  * Sentiment-labeled comment data.
  * Count of each sentiment.
  * Paths to bar and pie chart images.

### 3. **Sentiment Analysis with BERT**

* Uses `transformers.pipeline("sentiment-analysis")`.
* Detects humor using custom keyword logic (e.g., "lol", "😂") to reclassify sarcastic negatives as positive.

### 4. **Visualization**

* Matplotlib and Seaborn generate:

  * Bar chart for sentiment count.
  * Pie chart for sentiment distribution.
* Charts are saved to `/static/` and displayed on the frontend.

---

## ✅ Features

* Works directly with any public YouTube channel.
* Uses transformer-based NLP (BERT).
* Clean, pastel-themed frontend with responsive chart rendering.
* Humor-aware sentiment classification.
* Fully modular Flask backend.









