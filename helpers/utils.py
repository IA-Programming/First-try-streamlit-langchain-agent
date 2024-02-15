import os
from hashlib import md5
def add_text(text: str) -> str:
    doc_hash = md5()
    doc_hash.update(text.encode())
    doc_id = doc_hash.hexdigest()

    return doc_id

def list_files_with_extension(folder_path: str, extension: str) -> list[str]:
    file_names = []
    for file in os.listdir(folder_path):
        if file.endswith(extension):
            name_without_extension = os.path.splitext(file)[0]
            file_names.append(name_without_extension)
    return file_names