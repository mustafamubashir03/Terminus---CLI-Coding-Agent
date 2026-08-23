import subprocess
import os
from langchain.tools import tool

_BLOCKED_COMMANDS={"rm -rf /","rm -rf","fdisk","mkfs","dd if=/dev/urandom","shutdown /h","sudo su","sudo", ":(){:|:&};:(){:|:&}","curl -sSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh | bash"}
_TIMEOUT_SECONDS= 30

def _is_blocked(command:str)->bool:
    return any(blocked in command for blocked in _BLOCKED_COMMANDS)


def _format_result(result: subprocess.CompletedProcess)->str:
    parts =[]
    if result.stdout:
        parts.append(result.stdout.rstrip())
    if result.stderr:
        parts.append(f"ERROR:\n{result.stderr.rstrip()}")
    if result.returncode != 0:
        parts.append(f"Exit code {result.returncode}")
    return "\n".join(parts) if parts else "No output"

@tool("Run Shell Command")
def run_command(command:str)->str:
    """ Run a shell command. Times out after 30 seconds """
    if not command or not command.strip():
        return "No command provided"
    command = command.strip()
    if _is_blocked(command):
        return f"Blocked command: {command}"
    try:
        result=subprocess.run(
            command,
            shell=True,
            text=True,
            capture_output=True,
            timeout=_TIMEOUT_SECONDS
        )
        return _format_result(result)
    except subprocess.TimeoutExpired:
        return f"Command timed out after {_TIMEOUT_SECONDS} seconds: {command}"
    except Exception as e:
        return f"Error running command: {str(e)}"
    

@tool("Run command in directory")
def run_in_directory(command:str, directory:str=None)->str:
    """ Run a shell command inside a specific directory. Times out after 30 seconds """
    if not command or not command.strip():
        return "No command provided"
    command = command.strip()
    if _is_blocked(command):
        return f"Blocked command: {command}"
    try:
        result=subprocess.run(
            command,
            shell=True,
            text=True,
            capture_output=True,
            timeout=_TIMEOUT_SECONDS,
            cwd=directory
        )
        return _format_result(result)
    except subprocess.TimeoutExpired:
        return f"Command timed out after {_TIMEOUT_SECONDS} seconds: {command}"
    except Exception as e:
        return f"Error running command: {str(e)}"
        