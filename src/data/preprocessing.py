import unicodedata
import re

def preprocess_text(text: str) -> str:
    """Standardizes text for financial classification with multilingual support."""
    if not isinstance(text, str): 
        return ""
    # Normalize lower case
    text = text.lower()
    # Normalize accents
    text = ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
    # Keep alphanumeric and spaces
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.strip()
