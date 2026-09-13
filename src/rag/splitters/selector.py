import re
from .faq_splitter import FAQSplitter
from .recursive_splitter import RecursiveSplitter


def is_faq(text):
    return bool(re.search(r"(?m)^\s*\d+[\.、]\s*", text))


def get_splitter(filetype, text=""):
    filetype = (filetype or "").lower()

    if filetype in {".pdf", ".docx"}:
        return RecursiveSplitter()

    if filetype in {".txt", ".md"} and is_faq(text):
        return FAQSplitter()

    return RecursiveSplitter()
