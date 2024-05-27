import os
import yaml
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_astradb import AstraDBVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_openai import OpenAIEmbeddings


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

print("Loading the blog contents") 
# Load, chunk and index the contents of the blog.
loader = WebBaseLoader(
    web_paths=("https://www.jeyamohan.in/200801/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("td-post-content tagdiv-type",)
        )
    ),
)

docs = loader.load()

print("Splitting the blog content into smaller chunks")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# Retrieve and generate using the relevant snippets of the blog

embeddings = OpenAIEmbeddings()

vstore = AstraDBVectorStore(
    embedding = embeddings,
    namespace = ASTRA_DB_KEYSPACE,
    collection_name="docs",
    token = ASTRA_DB_APPLICATION_TOKEN,
    api_endpoint = ASTRA_DB_API_ENDPOINT
)

inserted_ids = vstore.add_documents(splits)
print(f"\nInserted {len(inserted_ids)} documents.")

