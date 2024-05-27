import yaml
import os
from langchain_astradb import AstraDBVectorStore
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from datasets import load_dataset

print("Loading Configuration data")

def read_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
        return config
    
config = read_config('apikeys.yaml')

ASTRA_DB_APPLICATION_TOKEN = config['apikeys']['astra_vector_db']
ASTRA_DB_API_ENDPOINT = "https://3787f08e-f351-42f4-b3bd-cfb6f9f924b2-us-east1.apps.astra.datastax.com"
ASTRA_DB_KEYSPACE = "default_keyspace"
OPENAI_API_KEY = config['apikeys']['openai']
LANGCHAIN_API_KEY = config['apikeys']['langsmith']

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_API_KEY'] = LANGCHAIN_API_KEY
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

embeddings = OpenAIEmbeddings()

vstore = AstraDBVectorStore(
    embedding = embeddings,
    namespace = ASTRA_DB_KEYSPACE,
    collection_name="test",
    token = ASTRA_DB_APPLICATION_TOKEN,
    api_endpoint = ASTRA_DB_API_ENDPOINT
)

print("Loading the dataset") 
philo_dataset = load_dataset("datastax/philosopher-quotes")["train"]
print("An example entry:")
print(philo_dataset[16])

docs = []
for entry in philo_dataset:
    metadata = {"author": entry["author"]}
    if entry["tags"]:
        # Add metadata tags to the metadata dictionary
        for tag in entry["tags"].split(";"):
            metadata[tag] = "y"
    # Add a LangChain document with the quote and metadata tags
    doc = Document(page_content=entry["quote"], metadata=metadata)
    docs.append(doc)

inserted_ids = vstore.add_documents(docs)
print(f"\nInserted {len(inserted_ids)} documents.")

results = vstore.similarity_search("the roots of education", k=5)
for res in results:
    print(f"* {res.page_content} [{res.metadata}]")