from dotenv import load_dotenv
import os

load_dotenv()  # load variables from .env

# Store API keys here (never hardcode them in main code)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")