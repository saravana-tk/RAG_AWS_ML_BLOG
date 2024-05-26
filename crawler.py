import os
import yaml
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load OPENAI & Langsmith API Keys

def load_config(configfile):
    with open(configfile, 'r') as file:
        config = yaml.safe_load(file)
        return config

config = load_config('apikeys.yaml')

langchain_api_key = config['apikeys']['langsmith']
openai_api_key = config['apikeys']['openai']

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_API_KEY'] = langchain_api_key
os.environ['OPENAI_API_KEY'] = openai_api_key

# Initialize the LLM
llm = ChatOpenAI(model='gpt-4o')

# Load, chunk and index the contents of the blog.
loader = WebBaseLoader(
    web_paths=("https://aws.amazon.com/blogs/machine-learning/how-20-minutes-empowers-journalists-and-boosts-audience-engagement-with-generative-ai-on-amazon-bedrock/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("blog-post-content lb-rtxt",)
        )
    ),
)

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

# Retrieve and generate using the relevant snippets of the blog

retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print(rag_chain.invoke("Who is a major player in French landscape?"))