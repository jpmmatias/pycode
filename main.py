from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
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
    prompt=code_prompt,
    output_key="code"
)

test_prompt = PromptTemplate(
    input_variables=[ "code"],
    template="Write a simple test that will test this code: {code}",
)

test_chain= LLMChain(
    llm = llm,
    prompt=test_prompt,
    output_key="test"
)

chain = SequentialChain(
    chains=[code_chain, test_chain],
    input_variables=["task", "language"],
    output_variables=["test", "code"]
)

result = chain({
    "language": args.language,
    "task":args.task
})

print(result["test"])


