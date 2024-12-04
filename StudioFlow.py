import os
from flask import Flask, request, jsonify
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.schema import Document  # For LangChain's document processing
from dotenv import load_dotenv
from flask_cors import CORS

import re
import json
import requests
import xml.etree.ElementTree as ET

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

app = Flask(__name__)
cors = CORS(app)


def get_video_id(video_url):
    """
    Extract video ID from the YouTube video URL.

    Args:
        video_url (str): YouTube video URL

    Returns:
        str: The video ID extracted from the URL
    """
    # More comprehensive video ID extraction
    # Handle full YouTube URLs, shortened URLs, and embedded URLs
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',  # Standard and embed URLs
        r'youtu\.be/([0-9A-Za-z_-]{11})',  # Shortened URLs
        r'youtube\.com/embed/([0-9A-Za-z_-]{11})'  # Embed URLs
    ]

    for pattern in patterns:
        match = re.search(pattern, video_url)
        if match:
            return match.group(1)

    raise ValueError("Invalid YouTube URL. Could not extract video ID.")


def get_cc(video_url):
    """
    Extract captions from a YouTube video using a more comprehensive approach.

    Args:
        video_url (str): YouTube video URL

    Returns:
        str: Extracted captions as plain text
    """
    # Comprehensive set of headers to mimic browser request
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Referer': 'https://www.youtube.com/',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1'
    }

    # Get the video ID
    video_id = get_video_id(video_url)

    # Construct the watch page URL
    watch_url = f'https://www.youtube.com/watch?v={video_id}'

    try:
        # Send request to YouTube watch page
        response = requests.get(watch_url, headers=headers)
        response.raise_for_status()

        # Multiple strategies to find caption tracks
        caption_strategies = [
            # Strategy 1: Original regex for caption tracks
            r'captionTracks":\s*(\[.*?\])',
            # Strategy 2: Alternative regex pattern
            r'"captionTracks":(\[.*?\])',
            # Strategy 3: More lenient search
            r'captionTracks.*?(\[.*?\])'
        ]

        captions_match = None
        for strategy in caption_strategies:
            captions_match = re.search(strategy, response.text, re.DOTALL)
            if captions_match:
                break

        if not captions_match:
            raise ValueError("No caption tracks found. The video might not have captions.")

        # Parse the captions JSON
        try:
            captions_json = json.loads(captions_match.group(1))
        except json.JSONDecodeError:
            # If JSON parsing fails, try to clean the string
            cleaned_json_str = re.sub(r',\s*}', '}', captions_match.group(1))
            captions_json = json.loads(cleaned_json_str)

        # Prepare to store captions
        all_captions = []

        # Iterate through available captions
        for caption_track in captions_json:
            caption_url = caption_track.get('baseUrl')

            if not caption_url:
                continue

            try:
                # Fetch the caption XML
                caption_response = requests.get(caption_url, headers=headers)
                caption_response.raise_for_status()

                # Parse XML captions
                root = ET.fromstring(caption_response.text)

                # Extract caption texts
                language_captions = []
                for text_elem in root.findall('text'):
                    caption_text = text_elem.text.strip() if text_elem.text else ''
                    if caption_text:
                        language_captions.append(caption_text)

                # Combine captions for this language
                if language_captions:
                    all_captions.append(' '.join(language_captions))

            except (ET.ParseError, requests.RequestException) as caption_error:
                print(f"Error processing captions: {caption_error}")
                continue

        # Return combined captions as a single string
        if not all_captions:
            raise ValueError("No valid captions could be extracted.")

        return ' '.join(all_captions)

    except Exception as e:
        print(f"Unexpected error in get_cc: {e}")
        raise


def generate(context, template):
    prompt_template = template

    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)

    prompt = PromptTemplate(template=prompt_template, input_variables=[context])

    input_documents = [Document(page_content=context)]

    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    result = chain.run(input_documents)

    # stripping spaces and new lines
    return result.strip()


def regenerate(context, template, user_prompt):
    prompt_template = template

    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "user_prompt"])

    input_documents = [context, user_prompt]
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    result = chain.run(input_documents)

    # stripping spaces and new lines
    return result.strip()


# Define API route POST request for processing YouTube transcript
@app.route('/process', methods=["POST"])
def process():
    # getting data from the request
    data = request.json
    url = data.get('url')
    choice = data.get('choice')

    # validation of URL
    if not url or not choice:
        return jsonify({"error": "Please provide a valid choice"}), 400

    try:
        cc = get_cc(video_url=url)

        # Selects the appropriate template
        if choice == 1:
            template = """
                        I have a youtube video transcript that needs to be made into a youtube title. 
                        Ensure the title generated from transcript is attractive, short and maintains the context of the video
                        and is only one sentence. Only send the title.
                        Here's the transcript:

                        CONTEXT: \n{context}\n

                        ANSWER:
                        """
        elif choice == 2:
            template = """
                        I have a youtube video transcript that needs to be made into a youtube description. 
                        Ensure the description generated from transcript is well-structured, accurate, and maintains the context of the video.
                        Here's the transcript:

                        CONTEXT: \n{context}\n

                        ANSWER:
                        """
        elif choice == 3:
            template = """
                        I have a youtube video transcript that needs to be made into youtube tags. 
                        Ensure the tags generated from transcript are individual words and maintain the context of the video.
                        Here's the transcript:

                        CONTEXT: \n{context}\n

                        ANSWER:
                        """
        else:
            return jsonify({"error": "Invalid choice. Must be 1, 2, or 3."}), 400

        # Generate result
        result = generate(cc, template)
        return jsonify({'result': result}), 201

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/enhance', methods=["POST"])
def enhance():
    data = request.json
    text = "Title :" + data.get('text')
    contentType = data.get('contentType')
    user_prompt = data.get('user_prompt')

    if contentType == "title":
        template = """
        I will provide a YouTube video title.
        Your task is to enhance it based on the given context and current YouTube trends.
        Ensure the enhanced title adheres to the same character limit as the original.
        Provide only one title. Make it so however the user requests.    
        CONTEXT: \n{context}\n

        ENHANCED TITLE:

        \nUSER: 
        """ + user_prompt
    elif contentType == "description":
        template = """
        I will provide a YouTube video description.
        Your task is to enhance it based on the given context and current YouTube trends.
        Provide only one description. Make it so however the user requests.
        CONTEXT: \n{context}\n

        ENHANCED DESCRIPTION:

        \nUSER: 
        """ + user_prompt
    else:
        return jsonify({"error": "Invalid choice. Must be 'title' or 'description'."}), 400

    regen = generate(text, template)

    return jsonify({'result': regen}), 201


@app.route('/check', methods=["GET"])
def check():
    youtube_url = request.args.get('url')

    # Validation
    if not youtube_url:
        return jsonify({"error": "Provide a valid youtube url"}), 400

    try:
        has_transcript, auto_generated = get_cc(video_url=youtube_url)

        if has_transcript:
            return jsonify({"has_transcript": True, "auto_generated": auto_generated}), 200
        else:
            return jsonify({"has_transcript": False}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def main():
    app.run(host="0.0.0.0", port=8080)


if __name__ == "__main__":
    main()
