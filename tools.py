from bs4 import BeautifulSoup
from urllib.request import Request, urlopen
from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools import tool

@tool
def scrape_menu(url: str) -> str:
    """"Scrape the menu from the restaurant URL and follow the steps below to return the text content"""
    # 1. Load and convert HTML → text
    loader = AsyncChromiumLoader([url])
    tt = Html2TextTransformer()
    docs = tt.transform_documents(loader.load())

    # 2. Split text into chunks
    ts = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=0)
    fd = ts.split_documents(docs)

    return fd[0].page_content

