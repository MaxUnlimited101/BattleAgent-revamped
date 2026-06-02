import os
import base64
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

def run_gpt4v(image_path, textual_prompt):
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode()

    llm = ChatOpenAI(
        model="gpt-4-vision-preview",
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    message = HumanMessage(content=[
        {"type": "text", "text": textual_prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_image}"}},
    ])

    response = llm.invoke([message])
    return response.content
