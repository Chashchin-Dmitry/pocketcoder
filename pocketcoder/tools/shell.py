"""
Shell command execution for PocketCoder.

Provides safe command execution with confirmation and timeout.
Smart handling of long-running commands (servers, watchers).
"""

from __future__ import annotations

import re
import subprocess
import time
import threading
from pathlib import Path
from typing import Callable


# Track background processes for cleanup
_background_processes: list[subprocess.Popen] = []
MAX_BACKGROUND_PROCESSES = 3


# Long-running command patterns (servers, watchers, etc.)
LONG_RUNNING_PATTERNS = [
    r"python\s+-m\s+http\.server",
    r"python.*manage\.py\s+runserver",
    r"flask\s+run",
    r"uvicorn",
    r"gunicorn",
    r"npm\s+(start|run\s+dev|run\s+serve)",
    r"yarn\s+(start|dev|serve)",
    r"node\s+.*server",
    r"nodemon",
    r"next\s+dev",
    r"vite",
    r"webpack.*serve",
    r"http-server",
    r"live-server",
    r"docker\s+run\s+(?!.*-d)",  # docker run without -d
    r"docker-compose\s+up\s+(?!.*-d)",
    r"tail\s+-f",
    r"watch\s+",
    r"sleep\s+\d{3,}",  # sleep > 100 seconds
]


def is_long_running_command(cmd: str) -> bool:
    """Check if command is a long-running process (server, watcher)."""
    for pattern in LONG_RUNNING_PATTERNS:
        if re.search(pattern, cmd, re.IGNORECASE):
            return True
    return False


def cleanup_background_processes():
    """Kill all background processes."""
    global _background_processes
    for proc in _background_processes:
        try:
            if proc.poll() is None:  # Still running
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()
        except Exception:
            pass
    _background_processes = []


# Dangerous command patterns
DANGEROUS_PATTERNS = [
    r"rm\s+-rf\s+/",
    r"rm\s+-rf\s+~",
    r"rm\s+-rf\s+\*",
    r"sudo\s+rm",
    r">\s*/dev/",
    r"chmod\s+777",
    r"mkfs\.",
    r"dd\s+if=.*of=/dev/",
    r":\(\)\{.*\}",  # Fork bomb
]


def is_dangerous_command(cmd: str) -> bool:
    """
    Check if command matches dangerous patterns.

    Args:
        cmd: Command string

    Returns:
        True if command looks dangerous
    """
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, cmd, re.IGNORECASE):
            return True
    return False


def execute_command(
    cmd: str,
    cwd: Path | None = None,
    timeout: int | None = None,
    confirm_callback: Callable[[str, bool], bool] | None = None,
) -> tuple[bool, str, str, int]:
    """
    Execute a shell command with smart timeout handling.

    - Long-running commands (servers): run in background, return after startup
    - Normal commands: wait for completion with timeout
    - Install commands: longer timeout (300s)

    Args:
        cmd: Command to execute
        cwd: Working directory (defaults to cwd)
        timeout: Timeout in seconds (None = auto-detect)
        confirm_callback: Function to confirm execution (cmd, is_dangerous) -> bool

    Returns:
        Tuple of (executed, stdout, stderr, returncode)
        - executed: False if cancelled or blocked
        - returncode: -1 if not executed, 0 for background processes
    """
    global _background_processes

    if cwd is None:
        cwd = Path.cwd()

    is_dangerous = is_dangerous_command(cmd)

    # Confirm if callback provided
    if confirm_callback is not None:
        if not confirm_callback(cmd, is_dangerous):
            return False, "", "Cancelled by user", -1

    # Determine timeout based on command type
    if timeout is None:
        if is_long_running_command(cmd):
            # Long-running: start in background
            return _execute_background(cmd, cwd)
        elif re.search(r"(npm|pip|yarn|brew|apt|cargo)\s+install", cmd, re.IGNORECASE):
            timeout = 300  # Install commands need more time
        else:
            timeout = 120  # Default timeout

    # Execute normal command
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )

        # Add helpful hints for common errors
        stderr = result.stderr
        if result.returncode != 0 and stderr:
            stderr_lower = stderr.lower()

            # Path not found errors
            if "no such file or directory" in stderr_lower:
                stderr += "\n\n💡 Hint: Path does not exist. Use list_files to check directory structure."

            # Command not found
            elif "command not found" in stderr_lower or "not recognized" in stderr_lower:
                # Extract command name
                cmd_name = cmd.split()[0] if cmd else "command"
                stderr += f"\n\n💡 Hint: '{cmd_name}' is not installed. Install it first or check spelling."

            # Permission denied
            elif "permission denied" in stderr_lower:
                stderr += "\n\n💡 Hint: Permission denied. May need sudo or check file permissions."

            # Directory not empty (for rm)
            elif "directory not empty" in stderr_lower:
                stderr += "\n\n💡 Hint: Use 'rm -rf' to remove non-empty directory (be careful!)."

        return True, result.stdout, stderr, result.returncode

    except subprocess.TimeoutExpired:
        return False, "", f"Command timed out after {timeout} seconds. Use /ps to check background processes.", -1

    except Exception as e:
        return False, "", str(e), -1


def _execute_background(cmd: str, cwd: Path) -> tuple[bool, str, str, int]:
    """
    Execute long-running command in background.

    Starts process, waits a bit for startup, returns initial output.
    """
    global _background_processes

    # Limit background processes
    _background_processes = [p for p in _background_processes if p.poll() is None]
    if len(_background_processes) >= MAX_BACKGROUND_PROCESSES:
        return False, "", f"Too many background processes (max {MAX_BACKGROUND_PROCESSES}). Use cleanup_background_processes() first.", -1

    try:
        # Start process
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=cwd,
        )

        _background_processes.append(process)

        # Wait for startup (2-3 seconds)
        output_lines = []
        start_time = time.time()
        startup_wait = 3  # seconds

        while time.time() - start_time < startup_wait:
            # Check if process died
            if process.poll() is not None:
                # Process ended, get output
                stdout, stderr = process.communicate()
                return True, stdout, stderr, process.returncode

            # Try to read output (non-blocking would be better but this works)
            time.sleep(0.5)

        # Process still running - good!
        # Try to get any initial output
        try:
            # Set non-blocking would be ideal, but just report running
            pass
        except Exception:
            pass

        pid = process.pid
        return True, f"🚀 Process started in background (PID: {pid})\nCommand: {cmd}\n\nProcess is running. Use Ctrl+C to stop or continue working.", "", 0

    except Exception as e:
        return False, "", str(e), -1


def execute_command_streaming(
    cmd: str,
    cwd: Path | None = None,
    timeout: int = 300,
    output_callback: Callable[[str], None] | None = None,
) -> tuple[bool, str, int]:
    """
    Execute command with streaming output.

    Args:
        cmd: Command to execute
        cwd: Working directory
        timeout: Timeout in seconds
        output_callback: Called with each line of output

    Returns:
        Tuple of (success, full_output, returncode)
    """
    if cwd is None:
        cwd = Path.cwd()

    output_lines = []

    try:
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=cwd,
            bufsize=1,
        )

        for line in iter(process.stdout.readline, ""):
            output_lines.append(line)
            if output_callback:
                output_callback(line)

        process.wait(timeout=timeout)

        return True, "".join(output_lines), process.returncode

    except subprocess.TimeoutExpired:
        if process:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        return False, "".join(output_lines), -1

    except Exception as e:
        return False, str(e), -1


def parse_commands_from_response(response: str) -> list[str]:
    """
    Extract shell commands from LLM response.

    Looks for:
    - ```bash or ```shell code blocks
    - <<<<<<< SHELL blocks

    Args:
        response: LLM response text

    Returns:
        List of command strings
    """
    commands = []

    # Pattern 1: ```bash code blocks
    bash_pattern = r"```(?:bash|shell|sh)\n(.+?)```"
    for match in re.finditer(bash_pattern, response, re.DOTALL):
        cmd = match.group(1).strip()
        if cmd:
            commands.append(cmd)

    # Pattern 2: <<<<<<< SHELL blocks
    shell_pattern = r"<<<<<<< SHELL\n(.+?)\n>>>>>>> SHELL"
    for match in re.finditer(shell_pattern, response, re.DOTALL):
        cmd = match.group(1).strip()
        if cmd:
            commands.append(cmd)

    return commands
