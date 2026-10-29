def split_text(content, chunk_size, overlap):
    chunks = []
    step = chunk_size - overlap
    start = 0
    while start < len(content):
        chunks.append(content[start:start + chunk_size])
        start += step
    return chunks