import os
from random import choice

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

def prompts(choice, context):
    if choice == 1:
        template = """
                    I have a youtube video transcript that needs to be made into a youtube title. 
                    Ensure the title generated from transcript is attractive, short and maintains the context of the video
                    and is only one sentence. Only send the title.
                    Here's the transcript:
                    CONTEXT: \n{context}\n
                    ANSWER:
                    """
        generate(context, template)

    elif choice == 2:
        template = """
            I have a youtube video transcript that needs to be made into a youtube description. 
            Ensure the description generated from transcript is well-structured, accurate, and maintains the context of the video.
            Here's the transcript:
            CONTEXT: \n{context}\n
            ANSWER:
            """
        generate(context, template)

    elif choice == 3:
        template = """
                    I have a youtube video transcript that needs to be made into youtube tags. 
                    Ensure the tags generated from transcript are individual words and maintains the context of the video.
                    Here's the transcript:
                    CONTEXT: \n{context}\n
                    ANSWER:
                    """
        generate(context, template)


def generate(context, template):
    prompt_template = template


    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context"])

    input_documents = [Document(page_content=context)]
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    result = chain.run(input_documents)

    print(result)


def main():
    url = input("Enter youtube link: ")
    cc = get_transcript(url)
    choice = int(input("1. Title\n 2.Description\n 3.Tags\n"))
    prompts(choice, cc)


if __name__ == "__main__":
    main()
