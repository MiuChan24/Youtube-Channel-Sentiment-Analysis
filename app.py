from flask import Flask, request, jsonify, render_template
from sentiment_module import (
    get_channel_video_ids,
    get_comments_from_videos,
    analyze_comments_with_humor,
    visualize_sentiment
)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    channel_id = data.get("channelId")

    try:
        # Step 1: Get video IDs
        video_ids = get_channel_video_ids(channel_id, max_results=3)

        # Step 2: Get comments from those videos
        comments = get_comments_from_videos(video_ids)

        # Step 3: Run BERT + humor-aware sentiment analysis
        df = analyze_comments_with_humor(comments)

        # Step 4: Visualize sentiment and save plot
        plot_path = visualize_sentiment(df)

        # Step 5: Return results to frontend
        return jsonify({
            "sentiment_counts": df['sentiment'].value_counts().to_dict(),
            "plot_path": plot_path
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
