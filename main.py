from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain 
from dotenv import load_dotenv
import argparse

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="return a list of numbers")
parser.add_argument("--language", default="python")
args = parser.parse_args()



llm = OpenAI(
    model_name="gpt-3.5-turbo-instruct",
)

code_prompt = PromptTemplate(
    input_variables=["language", "task"],
    template="Write a very short {language} function that will {task}",
)

code_chain = LLMChain(
    llm = llm,
    prompt=code_prompt
)

result = code_chain.invoke({
    "language": args.language,
    "task":args.task
})

print(result["text"])


