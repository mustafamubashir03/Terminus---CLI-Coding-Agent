import os
from langchain.tools import tool

_MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024

@tool
def read_file(file_path:str )->str:
    """ Read a file """
    if not file_path or not file_path.strip():
        return "No file path provided"
    file_path = file_path.strip()
    if not os.path.exists(file_path):
        return f"File not found: {file_path}"
    if os.path.getsize(file_path) > _MAX_FILE_SIZE_BYTES:
        return f"File is too large: {file_path}"
    try:
        with open(file_path, "r") as f:
            return f.read()
    except UnicodeDecodeError:
        return f"Cannot decode file: {file_path}"
    except PermissionError:
        return f"Permission denied: {file_path}"
    except Exception as e:
        return f"Error reading file: {str(e)}"



@tool
def write_file(file_path:str, content:str)->str:
    """ Write content to  a file, creating it and any parent directories as needed """
    if not file_path or not file_path.strip():
        return "No file path provided"
    if not content:
        return "No content provided"
    file_path = file_path.strip()
    try:
        if os.path.dirname(file_path):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w",encoding="utf-8") as f:
            f.write(content)
        return f"File written successfully: {file_path}"
    except PermissionError:
        return f"Permission denied: {file_path}"
    except Exception as e:
        return f"Error writing file: {str(e)}"



@tool
def append_file(file_path:str, content:str)->str:
    """ Append content to  a file, creating it and any parent directories as needed """
    if not file_path or not file_path.strip():
        return "No file path provided"
    if not content:
        return "No content provided"
    file_path = file_path.strip()
    try:
        if os.path.dirname(file_path):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "a",encoding="utf-8") as f:
            f.write(content)
        return f"File appended successfully: {file_path}"
    except PermissionError:
        return f"Permission denied: {file_path}"
    except Exception as e:
        return f"Error appending to file: {str(e)}"


@tool
def delete_file(file_path:str)->str:
    """ Delete a file """
    if not file_path or not file_path.strip():
        return "No file path provided"
    file_path = file_path.strip()
    try:
        os.remove(file_path)
        return f"File deleted successfully: {file_path}"
    except PermissionError:
        return f"Permission denied: {file_path}"
    except FileNotFoundError:
        return f"File not found: {file_path}"
    except Exception as e:
        return f"Error deleting file: {str(e)}"


@tool
def list_directory(directory:str)->str:
    """ List directory contents """
    if not directory or not directory.strip():
        return "No directory provided"
    directory = directory.strip()
    if not os.path.exists(directory):
        return f"Directory not found: {directory}"
    if not os.path.isdir(directory):
        return f"Path is not a directory: {directory}"
    try:
        return "\n".join(os.listdir(directory))
    except PermissionError:
        return f"Permission denied: {directory}"
    except Exception as e:
        return f"Error listing directory: {str(e)}"


@tool
def file_exists(file_path:str)->str:
    """ Check if a file exists """
    if not file_path or not file_path.strip():
        return "No file path provided"
    file_path = file_path.strip()
    try:
        if os.path.exists(file_path) and os.path.isfile(file_path):
            return f"File exists: {file_path}"
        else:
            return f"File does not exist: {file_path}"
    except Exception as e:
        return f"Error checking file: {str(e)}"