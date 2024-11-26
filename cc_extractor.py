import re
from youtube_transcript_api import YouTubeTranscriptApi

def get_transcript(youtube_url):
    # extract video ID with regex
    video_id_regex = r'(?:v=|\/)([0-9A-Za-z_-]{11}).*'
    match = re.search(video_id_regex, youtube_url)

    if match:
        video_id = match.group(1)
        try:
            #extract transcript
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            transcript_text = '\n'.join([entry['text'] for entry in transcript])

            #Store Transcript in the txt file
            #with open(f"{video_id}_transcript.txt","w",encoding="utf-8") as file:
            #   file.write(transcript_text)
            return transcript_text
            #return f"Transcripted saved as {video_id}_transcript.txt"
        except Exception as e:
            return f"Error fetching transcript: {e}"
    return 'Invalid Youtube URL'


if __name__ == "__main__":
    youtube_url = input('Enter the YouTube URL: ')
    #print(get_transcript(youtube_url))
