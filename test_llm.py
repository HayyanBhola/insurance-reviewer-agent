import os
from llm import get_llm

llm = get_llm()
reply = llm.invoke("In one sentence: what does an insurance claims adjuster do?")
print("Provider:", os.getenv("LLM_PROVIDER"))
print("Reply:", reply.text)
