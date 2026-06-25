import urllib.request
import json
import re
import math
import sqlite3
import sys
import os
from urllib.parse import quote_plus
from youtube_transcript_api import YouTubeTranscriptApi

DATABASE = os.path.join(os.path.dirname(__file__), '..', 'data', 'brain.db')

def init_db():
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    # Create tables if they don't exist
    c.execute('''
        CREATE TABLE IF NOT EXISTS paragraphs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS term_doc_count (
            term TEXT PRIMARY KEY,
            doc_count INTEGER NOT NULL
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT UNIQUE NOT NULL
        )
    ''')
    # Embeddings table for ML-based retrieval. Composite PK lets baseline
    # and fine-tuned variants coexist. See scripts/embeddings.py.
    c.execute('''
        CREATE TABLE IF NOT EXISTS paragraph_embeddings (
            paragraph_id INTEGER NOT NULL,
            model        TEXT    NOT NULL,
            vector       BLOB    NOT NULL,
            dim          INTEGER NOT NULL,
            PRIMARY KEY (paragraph_id, model)
        )
    ''')
    # Synthetic Q&A pairs for fine-tuning + eval. See scripts/synthetic_data.py.
    c.execute('''
        CREATE TABLE IF NOT EXISTS synthetic_pairs (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            query         TEXT    NOT NULL,
            paragraph_id  INTEGER NOT NULL,
            split         TEXT    NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def get_wikipedia_page(title):
    """Fetch the plain text content of a Wikipedia page."""
    url = f"https://en.wikipedia.org/w/index.php?title={quote_plus(title)}&printable=yes"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'AI-Model-Trainer/1.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            html = response.read().decode('utf-8')
    except Exception as e:
        print(f"Error fetching Wikipedia page: {e}")
        return None

    # Remove script and style tags
    html = re.sub(r'<script\b[^>]*>.*?</script>', ' ', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<style\b[^>]*>.*?</style>', ' ', html, flags=re.DOTALL | re.IGNORECASE)
    # Replace block-level tags with newline to preserve paragraph structure
    html = re.sub(r'<(p|div|h[1-6]|li|tr|blockquote|pre)[^>]*>', '\n', html, flags=re.IGNORECASE)
    html = re.sub(r'</(p|div|h[1-6]|li|tr|blockquote|pre)[^>]*>', '\n', html, flags=re.IGNORECASE)
    # Replace <br> and <hr> with newline
    html = re.sub(r'<br\s*/?>', '\n', html, flags=re.IGNORECASE)
    html = re.sub(r'<hr\s*/?>', '\n', html, flags=re.IGNORECASE)
    # Remove all other tags
    text = re.sub(r'<[^>]+>', ' ', html)
    # Replace multiple newlines with a single newline, then multiple spaces with space
    text = re.sub(r'\n+', '\n', text)  # multiple newlines to single newline
    text = re.sub(r'[ \t]+', ' ', text)  # multiple spaces to single space
    # Strip leading/trailing spaces and newlines
    text = text.strip()
    return text


# --- Language-specific documentation fetchers ---
# Each fetcher returns plain text suitable for paragraph splitting, or None on
# failure. Sources are public HTML/JSON endpoints that don't require API keys.

_LANG_ALIASES = {
    'matlab': 'matlab', 'octave': 'matlab',
    'r': 'r', 'cran': 'r', 'rlang': 'r',
    'c++': 'cpp', 'cpp': 'cpp', 'cplusplus': 'cpp', 'cxx': 'cpp',
}

def _strip_html(html):
    """Reuse the same HTML->text logic Wikipedia uses."""
    html = re.sub(r'<script\b[^>]*>.*?</script>', ' ', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<style\b[^>]*>.*?</style>', ' ', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<(p|div|h[1-6]|li|tr|blockquote|pre)[^>]*>', '\n', html, flags=re.IGNORECASE)
    html = re.sub(r'</(p|div|h[1-6]|li|tr|blockquote|pre)[^>]*>', '\n', html, flags=re.IGNORECASE)
    html = re.sub(r'<br\s*/?>', '\n', html, flags=re.IGNORECASE)
    html = re.sub(r'<hr\s*/?>', '\n', html, flags=re.IGNORECASE)
    text = re.sub(r'<[^>]+>', ' ', html)
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

def _http_get_text(url, timeout=10):
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'AI-Model-Trainer/1.0'})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.read().decode('utf-8', errors='replace')
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None

def get_matlab_doc(query):
    """Fetch a MATLAB documentation snippet for the given query.

    Uses MathWorks' public search page and returns the title + first descriptive
    paragraph from the top hit.
    """
    url = f"https://www.mathworks.com/help/search.html?q={quote_plus(query)}&type=function"
    html = _http_get_text(url)
    if not html:
        return None
    # Top result title and description live in anchor + sibling <p> within
    # `.search-results` items. Be defensive about absent elements.
    m = re.search(r'<a[^>]+href="([^"]+)"[^>]*>([^<]+)</a>', html)
    title = m.group(2).strip() if m else f"MATLAB: {query}"
    # Grab a small window of text after the first result link.
    snippet_match = re.search(r'<a[^>]+href="[^"]+"[^>]*>[^<]+</a>\s*</h\d+>\s*<p[^>]*>(.+?)</p>',
                              html, flags=re.DOTALL | re.IGNORECASE)
    snippet = ''
    if snippet_match:
        snippet = _strip_html(snippet_match.group(1))
    if not snippet:
        snippet = _strip_html(html)[:1500]
    return f"MATLAB documentation: {title}\n{snippet}"

def get_r_doc(query):
    """Fetch an R documentation snippet for the given query.

    Uses RDocumentation's public search page and returns the title + first
    descriptive paragraph from the top hit.
    """
    url = f"https://www.rdocumentation.org/search?q={quote_plus(query)}"
    html = _http_get_text(url)
    if not html:
        return None
    m = re.search(r'<a[^>]+href="(/[^"]+)"[^>]*>([^<]+)</a>', html)
    title = m.group(2).strip() if m else f"R: {query}"
    snippet_match = re.search(r'<a[^>]+href="/[^"]+"[^>]*>[^<]+</a>(.{0,800}?)</li>',
                             html, flags=re.DOTALL | re.IGNORECASE)
    snippet = ''
    if snippet_match:
        snippet = _strip_html(snippet_match.group(1))
    if not snippet:
        snippet = _strip_html(html)[:1500]
    return f"R documentation: {title}\n{snippet}"

def get_cpp_doc(query):
    """Fetch a C++ documentation snippet for the given query.

    Uses cppreference's MediaWiki search API (JSON) and returns the title +
    short snippet from the top hit. Falls back to a direct page fetch.
    """
    api_url = (
        "https://en.cppreference.com/w/api.php?"
        f"action=query&list=search&srsearch={quote_plus(query)}"
        "&srlimit=1&format=json"
    )
    html = _http_get_text(api_url)
    if not html:
        return None
    try:
        data = json.loads(html)
    except json.JSONDecodeError:
        return None
    results = data.get('query', {}).get('search', [])
    if not results:
        return f"C++ documentation: no results for '{query}'"
    top = results[0]
    title = top.get('title', f"C++: {query}")
    snippet = _strip_html(top.get('snippet', ''))
    # cppreference snippets contain <span> markup from search highlighting;
    # _strip_html already removes those tags.
    return f"C++ documentation: {title}\n{snippet}"

LANGUAGE_FETCHERS = {
    'matlab': get_matlab_doc,
    'r': get_r_doc,
    'cpp': get_cpp_doc,
}

def detect_language_topic(topic):
    """If `topic` starts with a recognized language token, return
    (language, query). Otherwise return None.
    """
    parts = topic.split(None, 1)
    if len(parts) < 2:
        return None
    lang = _LANG_ALIASES.get(parts[0].lower())
    if not lang:
        return None
    return lang, parts[1]

def get_language_doc(language, query):
    fetcher = LANGUAGE_FETCHERS.get(language)
    if not fetcher:
        print(f"No fetcher registered for language '{language}'.")
        return None
    return fetcher(query)

def train_language_doc(language, query):
    """Train on the documentation page for `query` in the given language."""
    print(f"Fetching {language.upper()} documentation for: {query}")
    text = get_language_doc(language, query)
    if not text:
        print(f"Failed to fetch {language.upper()} documentation.")
        return False
    title = f"{language.upper()}:{query}"
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    if not paragraphs:
        print("No content found.")
        return False
    if len(paragraphs) > 100:
        paragraphs = paragraphs[:100]

    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    try:
        c.execute('INSERT INTO documents (title) VALUES (?)', (title,))
        doc_id = c.lastrowid
    except sqlite3.IntegrityError:
        c.execute('SELECT id FROM documents WHERE title = ?', (title,))
        doc_id = c.fetchone()[0]
    conn.commit()
    conn.close()

    for para in paragraphs:
        para_id = add_paragraph(para)
        terms = tokenize(para)
        update_document_frequency(terms)

    print(f"Trained on {len(paragraphs)} paragraph(s) from {language.upper()} doc '{query}'.")
    return True

# --- End language-specific fetchers ---

def get_youtube_transcript(url):
    """Fetch the transcript of a YouTube video."""
    try:
        # Extract video ID from URL
        if "youtu.be" in url:
            video_id = url.split("/")[-1].split("?")[0]
        elif "youtube.com" in url:
            if "v=" in url:
                video_id = url.split("v=")[1]
                if "&" in video_id:
                    video_id = video_id.split("&")[0]
            else:
                video_id = None
        else:
            video_id = None

        if video_id is None:
            print("Could not extract video ID from URL.")
            return None

        # Fetch transcript
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        # Join the text parts
        transcript = " ".join([entry['text'] for entry in transcript_list])
        return transcript
    except Exception as e:
        print(f"Error fetching YouTube transcript: {e}")
        return None
def tokenize(text):
    """Simple tokenization: lowercase and split on non-alphanumeric."""
    words = re.findall(r"\b[\w']+\b", text.lower())
    return words

def update_document_frequency(terms):
    """Update document frequency for terms in a document (set of terms)."""
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    unique_terms = set(terms)
    for term in unique_terms:
        c.execute('''
            INSERT INTO term_doc_count (term, doc_count)
            VALUES (?, 1)
            ON CONFLICT(term) DO UPDATE SET doc_count = doc_count + 1
        ''', (term,))
    conn.commit()
    conn.close()

def add_paragraph(text):
    """Add a paragraph to the database and return its ID."""
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute('INSERT INTO paragraphs (text) VALUES (?)', (text,))
    para_id = c.lastrowid
    conn.commit()
    conn.close()
    return para_id

def train_topic(topic):
    """Train the model on a Wikipedia topic or language documentation."""
    # If the topic starts with a recognized language token, dispatch to the
    # language-doc pipeline instead of Wikipedia.
    lang_match = detect_language_topic(topic)
    if lang_match is not None:
        language, query = lang_match
        return train_language_doc(language, query)

    print(f"Fetching Wikipedia page for: {topic}")
    if topic.startswith(('http://', 'https://')) and ('youtube.com' in topic or 'youtu.be' in topic):
        text = get_youtube_transcript(topic)
    else:
        text = get_wikipedia_page(topic)
    if not text:
        print("Failed to fetch Wikipedia page.")
        return False

    # Split into paragraphs (by newline)
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    if not paragraphs:
        print("No content found.")
        return False

    # Limit to first 100 paragraphs to avoid too much data
    if len(paragraphs) > 100:
        print(f"Limiting to first 100 paragraphs (out of {len(paragraphs)})")
        paragraphs = paragraphs[:100]

    # Store the document (we'll store the title for listing)
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    try:
        c.execute('INSERT INTO documents (title) VALUES (?)', (topic,))
        doc_id = c.lastrowid
    except sqlite3.IntegrityError:
        # Document already exists, get its ID
        c.execute('SELECT id FROM documents WHERE title = ?', (topic,))
        doc_id = c.fetchone()[0]
    conn.commit()
    conn.close()

    # Add each paragraph and update document frequency (per paragraph)
    for para in paragraphs:
        para_id = add_paragraph(para)
        terms = tokenize(para)
        update_document_frequency(terms)

    print(f"Trained on {len(paragraphs)} paragraphs from '{topic}'.")
    return True

def get_total_paragraphs():
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute('SELECT COUNT(*) FROM paragraphs')
    count = c.fetchone()[0]
    conn.close()
    return count

def get_vocab_size():
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute('SELECT COUNT(*) FROM term_doc_count')
    count = c.fetchone()[0]
    conn.close()
    return count

def get_document_count():
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute('SELECT COUNT(*) FROM documents')
    count = c.fetchone()[0]
    conn.close()
    return count

def compute_idf(term, total_docs):
    """Compute inverse document frequency for a term."""
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute('SELECT doc_count FROM term_doc_count WHERE term = ?', (term,))
    row = c.fetchone()
    conn.close()
    df = row[0] if row else 0
    # Add 1 to avoid division by zero
    return math.log((total_docs + 1) / (df + 1))

def compute_tfidf_vector(text, idf_dict):
    """Compute TF-IDF vector for a text given IDF values."""
    words = tokenize(text)
    tf = {}
    for word in words:
        tf[word] = tf.get(word, 0) + 1
    tfidf = {}
    for term, count in tf.items():
        tfidf[term] = count * idf_dict.get(term, 0)
    return tfidf

def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors."""
    dot_product = 0.0
    norm_a = 0.0
    norm_b = 0.0
    all_terms = set(vec1.keys()) | set(vec2.keys())
    for term in all_terms:
        a = vec1.get(term, 0.0)
        b = vec2.get(term, 0.0)
        dot_product += a * b
        norm_a += a * a
        norm_b += b * b
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (math.sqrt(norm_a) * math.sqrt(norm_b))

def query_question(question):
    """Query the database for answers to a question."""
    print(f"Processing question: {question}")
    # Get total number of paragraphs for IDF
    total_docs = get_total_paragraphs()
    if total_docs == 0:
        print("No documents trained yet. Please train on some topics first.")
        return []

    # Tokenize question
    question_terms = tokenize(question)
    # Compute IDF for each term in question
    idf_dict = {}
    for term in question_terms:
        idf_dict[term] = compute_idf(term, total_docs)
    # Compute TF-IDF vector for question
    question_vector = compute_tfidf_vector(question, idf_dict)

    # Fetch all paragraphs
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute('SELECT id, text FROM paragraphs')
    rows = c.fetchall()
    conn.close()

    similarities = []
    for para_id, para_text in rows:
        para_terms = tokenize(para_text)
        # Compute IDF for paragraph terms
        para_idf_dict = {}
        for term in para_terms:
            if term not in para_idf_dict:
                para_idf_dict[term] = compute_idf(term, total_docs)
        para_vector = compute_tfidf_vector(para_text, para_idf_dict)
        similarity = cosine_similarity(question_vector, para_vector)
        similarities.append((para_id, para_text, similarity))

    # Sort by similarity descending
    similarities.sort(key=lambda x: x[2], reverse=True)
    # Return top 3
    return similarities[:3]

def list_topics():
    """List all trained topics."""
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute('SELECT title FROM documents')
    rows = c.fetchall()
    conn.close()
    return [row[0] for row in rows]

def info_stats():
    """Return statistics about the database."""
    return {
        'documents': get_document_count(),
        'paragraphs': get_total_paragraphs(),
        'vocabulary_size': get_vocab_size()
    }

def clear_database():
    """Clear all data from the database."""
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute('DELETE FROM paragraphs')
    c.execute('DELETE FROM term_doc_count')
    c.execute('DELETE FROM documents')
    c.execute('DELETE FROM paragraph_embeddings')
    c.execute('DELETE FROM synthetic_pairs')
    conn.commit()
    conn.close()
    # Recreate tables (optional, but ensures tables exist)
    init_db()
    print("Database cleared.")

def get_related_wikipedia_titles(title, limit=3):
    """Fetch related Wikipedia article titles via the MediaWiki 'links' API.

    Returns up to `limit` titles from the article's outbound links, skipping
    self-references and Wikipedia meta pages (Help:, Special:, Wikipedia:).
    """
    api_url = (
        "https://en.wikipedia.org/w/api.php?"
        f"action=query&prop=links&titles={quote_plus(title)}"
        f"&pllimit={max(limit * 5, 20)}&format=json&redirects=1"
    )
    try:
        req = urllib.request.Request(api_url, headers={'User-Agent': 'AI-Model-Trainer/1.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode('utf-8'))
    except Exception as e:
        print(f"Error fetching related titles for '{title}': {e}")
        return []

    pages = data.get('query', {}).get('pages', {})
    related = []
    skip_prefixes = ('Help:', 'Special:', 'Wikipedia:', 'Talk:', 'Template:', 'Portal:', 'File:')
    for page in pages.values():
        for link in page.get('links', []):
            lt = link.get('title', '')
            if not lt or lt == title:
                continue
            if any(lt.startswith(p) for p in skip_prefixes):
                continue
            related.append(lt)
            if len(related) >= limit:
                return related
    return related

def auto_train(topic):
    """Auto train on a topic and then answer a few sample questions."""
    print(f"Auto-training on topic: {topic}")
    success = train_topic(topic)
    if not success:
        print("Auto-training failed.")
        return
    print("\nAuto-training complete. Now ready to answer questions.")
    print("I am ready! You can ask me questions about", topic + ".")
    print("Try: /ask What is", topic + "?")
    print("Or just type your question after /ask.")

def autotrain_topic(topic, related_limit=3):
    """Train on a topic and up to `related_limit` related Wikipedia articles.

    If `topic` looks like a YouTube URL, the YouTube transcript path is used
    for the primary source and no related-article expansion is performed.
    Returns the number of sources successfully trained on.
    """
    is_youtube = topic.startswith(('http://', 'https://')) and ('youtube.com' in topic or 'youtu.be' in topic)
    print(f"=== Auto-training pipeline for: {topic} ===")

    trained = 0
    if train_topic(topic):
        trained += 1
    else:
        print("Primary source failed; aborting auto-training pipeline.")
        return trained

    if is_youtube:
        print("YouTube URL detected; skipping related-article expansion.")
        return trained

    print(f"\nDiscovering up to {related_limit} related Wikipedia articles...")
    related = get_related_wikipedia_titles(topic, limit=related_limit)
    if not related:
        print("No related articles found.")
        return trained

    print(f"Found {len(related)} related article(s):")
    for r in related:
        print(f"  - {r}")
    print()

    for r in related:
        print(f"--- Training on related: {r} ---")
        if train_topic(r):
            trained += 1

    print(f"\n=== Auto-training complete: {trained} source(s) trained ===")
    print(f"You can now ask questions about {topic} (and related topics).")
    return trained

def main():
    if len(sys.argv) < 2:
        print("Usage: python brain.py <command> [args]")
        print("Commands: train <topic>, query <question>, list, info, clear,")
        print("          auto_train <topic>, autotrain <topic>,")
        print("          langtrain <matlab|r|cpp> <query>, ready")
        print("          embed_rebuild, model, eval, finetune")
        return

    # Initialize database (creates tables if not exist)
    init_db()

    command = sys.argv[1].lower()
    if command == "train":
        if len(sys.argv) < 3:
            print("Please specify a topic to train on.")
            return
        topic = " ".join(sys.argv[2:])
        train_topic(topic)
        print("Training complete. I am ready to answer questions.")
    elif command == "query":
        if len(sys.argv) < 3:
            print("Please ask a question.")
            return
        question = " ".join(sys.argv[2:])
        results = query_question(question)
        if not results:
            return
        print("\nTop 3 relevant paragraphs:")
        for i, (para_id, text, score) in enumerate(results, 1):
            print(f"{i}. Score: {score:.4f}")
            print(f"   {text[:200]}...")  # Show first 200 chars
            print()
    elif command == "list":
        topics = list_topics()
        if topics:
            print("Trained topics:")
            for topic in topics:
                print(f"  - {topic}")
        else:
            print("No topics trained yet.")
    elif command == "info":
        stats = info_stats()
        print(f"Documents (topics): {stats['documents']}")
        print(f"Paragraphs: {stats['paragraphs']}")
        print(f"Vocabulary size: {stats['vocabulary_size']}")
    elif command == "clear":
        clear_database()
    elif command == "auto_train":
        if len(sys.argv) < 3:
            print("Please specify a topic to auto-train on.")
            return
        topic = " ".join(sys.argv[2:])
        auto_train(topic)
    elif command == "autotrain":
        if len(sys.argv) < 3:
            print("Please specify a topic (or YouTube URL) to autotrain on.")
            return
        topic = " ".join(sys.argv[2:])
        autotrain_topic(topic)
    elif command == "langtrain":
        # langtrain <language> <query...>
        if len(sys.argv) < 4:
            print("Usage: python brain.py langtrain <matlab|r|cpp> <query>")
            return
        language = _LANG_ALIASES.get(sys.argv[2].lower())
        if not language:
            print(f"Unknown language '{sys.argv[2]}'. Supported: matlab, r, cpp.")
            return
        query = " ".join(sys.argv[3:])
        train_language_doc(language, query)
    elif command == "embed_rebuild":
        # Rebuild paragraph embeddings for the active model. Delta-encodes
        # only paragraphs missing from the embeddings table.
        try:
            from embeddings import rebuild_embeddings
            stats = rebuild_embeddings(verbose=True)
            print(f"Active model: {stats['model']}")
            print(f"Embedding dim: {stats['dim']}")
            print(f"Newly encoded: {stats['new']}")
            print(f"Already encoded: {stats['skipped']}")
        except Exception as e:
            print(f"Error rebuilding embeddings: {e}")
    elif command == "model":
        # Print embedding-model diagnostics.
        try:
            from embeddings import stats as emb_stats
            s = emb_stats()
            print(f"Active model: {s['model']}")
            print(f"Embedding dim: {s['dim']}")
            print(f"Encoded rows: {s['encoded']} / {s['paragraphs']} paragraphs")
        except Exception as e:
            print(f"Error getting model info: {e}")
    elif command == "eval":
        # Run held-out evaluation comparing baseline vs fine-tuned (if any).
        try:
            from eval import compare
            print(compare())
        except Exception as e:
            print(f"Error during eval: {e}")
    elif command == "finetune":
        # Run fine-tuning on synthetic_pairs, then eval.compare() to pick winner.
        try:
            from finetune import finetune
            result = finetune()
            print(result)
        except Exception as e:
            print(f"Error during fine-tune: {e}")
    elif command == "ready":
        # Check if there is any data
        para_count = get_total_paragraphs()
        if para_count > 0:
            print("I am ready! I have been trained on", para_count, "paragraph(s).")
            print("Ask me questions using /ask <your question>.")
        else:
            print("I have not been trained yet. Please train me first using /train <topic>.")
    else:
        print(f"Unknown command: {command}")
        print("Available commands: train, query, list, info, clear,")
        print("                    auto_train, autotrain, langtrain, ready,")
        print("                    embed_rebuild, model, eval, finetune")

if __name__ == "__main__":
    main()
