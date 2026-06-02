import os
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

def run_claude(message):
    llm = ChatAnthropic(
        model="claude-3-opus-20240229",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        max_tokens=3000,
        temperature=1,
    )
    response = llm.invoke([HumanMessage(content=message)])
    return response.content

def run_gpt(text_prompt, temperature: float = 0, model="gpt-4-1106-preview", seed=440):
    llm = ChatOpenAI(
        model=model,
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=temperature,
        model_kwargs={"seed": seed},
    )
    response = llm.invoke([HumanMessage(content=text_prompt)])
    resp = response.content
    resp = resp.replace("```json", "").replace("```", "")
    return resp

def run_openrouter(text_prompt, temperature: float = 0, seed=440):
    model = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o")
    llm = ChatOpenAI(
        model=model,
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        temperature=temperature,
        model_kwargs={"seed": seed},
    )
    response = llm.invoke([HumanMessage(content=text_prompt)])
    resp = response.content
    resp = resp.replace("```json", "").replace("```", "")
    return resp

def run_ollama(text_prompt, temperature: float = 0):
    model = os.getenv("OLLAMA_MODEL", "llama3")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    llm = ChatOpenAI(
        model=model,
        api_key="ollama",
        base_url=base_url,
        temperature=temperature,
    )
    response = llm.invoke([HumanMessage(content=text_prompt)])
    resp = response.content
    resp = resp.replace("```json", "").replace("```", "")
    return resp

def run_LLM(model_type, prompt):
    if model_type == "gpt":
        return run_gpt(prompt)
    elif model_type == "claude":
        return run_claude(prompt)
    elif model_type == "openrouter":
        return run_openrouter(prompt)
    elif model_type == "ollama":
        return run_ollama(prompt)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
