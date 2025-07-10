import re
import logging
from typing import Dict, Any, List
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


def extract_source_from_text(text: str) -> Dict[str, Any]:
    """
    Extract source information from text
    """
    try:
        # Look for URL patterns
        url_pattern = r'https?://[^\s<>"]+|www\.[^\s<>"]+'
        urls = re.findall(url_pattern, text)

        # Look for file references
        file_pattern = r'[\w\-_]+\.pdf'
        files = re.findall(file_pattern, text)

        # Clean text content
        clean_text = re.sub(url_pattern, '', text)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()

        # Determine source
        source = "Unknown"
        if urls:
            source = urls[0]
        elif files:
            source = files[0]

        return {
            'content': clean_text,
            'source': source,
            'urls': urls,
            'files': files
        }
    except Exception as e:
        logger.error(f"Error extracting source: {e}")
        return {
            'content': text,
            'source': "Unknown",
            'urls': [],
            'files': []
        }


def validate_url(url: str) -> bool:
    """Validate URL format"""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False


def clean_text(text: str) -> str:
    """Clean and normalize text"""
    if not text:
        return ""

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)

    # Remove special characters but keep Vietnamese
    text = re.sub(r'[^\w\s\u00C0-\u024F\u1E00-\u1EFF.,!?;:()\-]', '', text)

    return text.strip()


def truncate_text(text: str, max_length: int = 200) -> str:
    """Truncate text to specified length"""
    if len(text) <= max_length:
        return text

    truncated = text[:max_length]
    last_space = truncated.rfind(' ')

    if last_space > max_length * 0.8:
        return truncated[:last_space] + "..."

    return truncated + "..."