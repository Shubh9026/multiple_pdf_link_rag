import os
from openai import OpenAI

##code for multiple pdf links

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_community.embeddings import HuggingFaceEmbeddings
from pymongo import MongoClient
from dotenv import load_dotenv
import os
load_dotenv('.env')

# List of PDF URLs
pdf_links = [
    "https://www2.deloitte.com/content/dam/Deloitte/in/Documents/consumer-business/immersion/Deloitte-Chaper-india-Food-Report.pdf",
    "https://web-assets.bcg.com/img-src/BCG-Ten-Trends-That-Are-Altering-Consumer-Behavior-in-India-Oct-2019_tcm9-231429.pdf"
]

# Initialize variables to store all documents
all_docs = []

# Process each PDF link
for pdf_link in pdf_links:
    # Load PDF
    loader = PyPDFLoader(pdf_link)
    pages = loader.load()

    # Split text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    docs = text_splitter.split_documents(pages)
    
    # Append documents to the list
    all_docs.extend(docs)

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')

# Function to connect to MongoDBpython
def connection(uri):
    try:
        client = MongoClient(uri)
        database = client['Resto']
        print('Mongo connected!')
        return database
    except:
        print('Connection issue: Check internet or uri or cluster.')

# Connect to the database
db = connection(os.getenv('DB'))

# Store documents in MongoDB Atlas Vector Search
docstore = MongoDBAtlasVectorSearch.from_documents(
    all_docs, embeddings, collection=db['Food'], index_name='shubh_index'
)

# Response generation from PDFs
index_name = 'shubh_index'
vector_index = MongoDBAtlasVectorSearch(db['Food'], HuggingFaceEmbeddings(model_name='all-mpnet-base-v2'), index_name=index_name)
print(vector_index)

# Assuming you have OpenAI API configured
from openai import OpenAI

client = OpenAI(api_key=os.getenv('openai_key'))

# Generate response
prompt = 'Health in Food'
context = vector_index.similarity_search(prompt)
messages = [{"role": "user", "content": f"This is the context:{context}. Answer this question: {prompt}"}]

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=messages,
    temperature=0.5,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
)

# Extract and print the content text
content_text = response.choices[0].message.content
print(content_text
