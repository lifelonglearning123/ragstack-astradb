import os
import streamlit as st
from azure.identity import DefaultAzureCredential
from azure.appconfiguration import AzureAppConfigurationClient
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import AstraDB

# Azure App Configuration Endpoint
APP_CONFIG_ENDPOINT = "https://exemplaschatbot.azconfig.io"

# Authenticate and connect to Azure App Configuration
credential = DefaultAzureCredential()
client = AzureAppConfigurationClient(base_url=APP_CONFIG_ENDPOINT, credential=credential)

# Load secrets into Streamlit's st.secrets dynamically
def load_secrets():
    keys = [
        "OPENAI_API_KEY", "ASTRA_ENDPOINT", "ASTRA_TOKEN",
        "LANGCHAIN_TRACING_V2", "datastax_password",
        "innovate_password", "datastax_language", "delete_option"
    ]
    for key in keys:
        setting = client.get_configuration_setting(key=key)
        if setting:
            st.secrets[key] = setting.value

load_secrets()

# Initialize OpenAI Embeddings
@st.cache_resource()
def load_embedding():
    return OpenAIEmbeddings()

# Initialize AstraDB Vector Store
@st.cache_resource()
def load_vectorstore():
    collection_name = "vector_context_innovate"  # Change if needed
    astra_db = AstraDB(
        embedding=load_embedding(),
        collection_name=collection_name,
        token=st.secrets["ASTRA_TOKEN"],
        api_endpoint=st.secrets["ASTRA_ENDPOINT"],
    )

    # Check if collection exists, create if necessary
    existing_collections = astra_db.list_collections()
    if collection_name not in existing_collections:
        print(f"Creating collection: {collection_name}")
        astra_db.create_collection(collection_name)

    return astra_db