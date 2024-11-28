import os
from random import choice

from flask import Flask, request, jsonify
import streamlit as st
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.schema import Document  # For LangChain's document processing
from dotenv import load_dotenv
from cc_extractor import get_transcript

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

app = Flask(__name__)


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

    prompt = PromptTemplate(template=prompt_template, input_variables=["context","user_prompt"])


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
        cc = get_transcript(url)

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

    print(text)
    print(contentType)

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


def main():
    app.run(debug=True)


if __name__ == "__main__":
    main()
