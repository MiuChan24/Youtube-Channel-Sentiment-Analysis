from googleapiclient.discovery import build
from transformers import pipeline
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------------------
# API Setup
# ----------------------------------------
api_key = "API Key"  # 🔑 Add your YouTube Data API key here
youtube = build('youtube', 'v3', developerKey=api_key)
analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
HUMOR_KEYWORDS = ['lol', 'lmao', '😂', '🤣', 'haha', 'ikuyo', 'wtf', 'omg', 'that’s crazy', 'lmaoo']

# ----------------------------------------
# YouTube Helpers
# ----------------------------------------

def get_channel_video_ids(channel_id, max_results=3):
    res = youtube.channels().list(id=channel_id, part='contentDetails').execute()
    if not res['items']:
        print("Invalid or private channel ID.")
        return []
    uploads_playlist_id = res['items'][0]['contentDetails']['relatedPlaylists']['uploads']
    playlist_items = youtube.playlistItems().list(
        playlistId=uploads_playlist_id,
        part='snippet',
        maxResults=max_results
    ).execute()
    return [item['snippet']['resourceId']['videoId'] for item in playlist_items['items']]

def get_comments_from_videos(video_ids, max_comments=50):
    comments = []
    for vid in video_ids:
        try:
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=vid,
                maxResults=min(max_comments, 100),
                textFormat="plainText"
            )
            response = request.execute()
            for item in response.get("items", []):
                comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                comments.append(comment)
        except Exception as e:
            print(f"Error on video {vid}: {e}")
    return comments

# ----------------------------------------
# Sentiment Analysis
# ----------------------------------------

def adjust_sentiment_for_humor(text, sentiment, score):
    if sentiment == "NEGATIVE" and any(kw in text.lower() for kw in HUMOR_KEYWORDS):
        return "POSITIVE", min(score, 0.80)
    return sentiment, score

def analyze_comments_with_humor(comments):
    results = []
    for comment in comments:
        result = analyzer(comment[:512])[0]
        sentiment, score = adjust_sentiment_for_humor(comment, result['label'], result['score'])
        results.append({
            "comment": comment,
            "sentiment": sentiment,
            "score": round(score, 2)
        })
    return pd.DataFrame(results)

# ----------------------------------------
# Visualization
# ----------------------------------------

def visualize_sentiment(df, output_path="static/sentiment_plot.png"):
    sentiment_counts = df['sentiment'].value_counts()

    # Pie Chart
    plt.figure(figsize=(6, 6))
    plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=140)
    plt.title("Sentiment Distribution (Pie Chart)")
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(output_path.replace('.png', '_pie.png'))
    plt.close()

    # Bar Chart
    plt.figure(figsize=(6, 4))
    sentiment_counts.plot(kind='bar', color=['skyblue', 'salmon'])
    plt.title("Sentiment Count (Bar Chart)")
    plt.xlabel("Sentiment")
    plt.ylabel("Comments")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return output_path

# ----------------------------------------
# Optional Test Run
# ----------------------------------------

