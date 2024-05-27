from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_astradb import AstraDBVectorStore
from langchain import hub
import os
import yaml

# Load OPENAI & Langsmith API Keys
print("Loading Configuration data")

def load_config(configfile):
    with open(configfile, 'r') as file:
        config = yaml.safe_load(file)
        return config

config = load_config('apikeys.yaml')

ASTRA_DB_APPLICATION_TOKEN = config['apikeys']['astra_vector_db']
ASTRA_DB_API_ENDPOINT = "https://3787f08e-f351-42f4-b3bd-cfb6f9f924b2-us-east1.apps.astra.datastax.com"
ASTRA_DB_KEYSPACE = "default_keyspace"
LANGCHAIN_API_KEY = config['apikeys']['langsmith']
OPENAI_API_KEY = config['apikeys']['openai']

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_API_KEY'] = LANGCHAIN_API_KEY
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

# Initialize the LLM
llm = ChatOpenAI(model='gpt-4o')

# Retrieve and generate using the relevant snippets of the blog

embeddings = OpenAIEmbeddings()

vstore = AstraDBVectorStore(
    embedding = embeddings,
    namespace = ASTRA_DB_KEYSPACE,
    collection_name="docs",
    token = ASTRA_DB_APPLICATION_TOKEN,
    api_endpoint = ASTRA_DB_API_ENDPOINT
)

retriever = vstore.as_retriever()
prompt = hub.pull("rag_detailed")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print(rag_chain.invoke("நம் கல்விமுறையின் அடிப்படைச் சிக்கல்?"))