"""
Main Coder class - orchestrates all PocketCoder functionality.

v0.7.0: Added ChatStorage for raw conversation persistence and grep-based retrieval.
v0.8.0: Dynamic Context Discovery - compact prompts, dynamic few-shot loading.
v2.0.0: SESSION_CONTEXT architecture - LLM always sees full context, no keyword detection.
"""

from __future__ import annotations

import platform
import time
from pathlib import Path
from typing import Any, Optional

from pocketcoder.core.models import Message, FileContext, Edit, ParsedResponse, ToolCall, AgentStats
from pocketcoder.core.project_context import ProjectContext, TaskSummarizer
from pocketcoder.core.content_preview import ContentPreview
from pocketcoder.core.summary import build_session_context_xml, generate_full_summary
from pocketcoder.core.parser import parse_response
from pocketcoder.tools import TOOLS, get_tool, is_dangerous_command
from pocketcoder.core.prompt_loader import (
    build_compact_prompt,
    get_tool_hint,
    validate_tool_params,
    is_compact_mode_enabled,
)
from pocketcoder.core.applier import (
    apply_edit,
    apply_edits_to_content,
    generate_diff,
    group_edits_by_file,
    ChangeTracker,
)
from pocketcoder.providers import create_provider
from pocketcoder.providers.base import BaseProvider
from pocketcoder.hooks import hooks
from pocketcoder.tools.files import is_binary, resolve_path


class Coder:
    """
    Main orchestrator class for PocketCoder.

    Manages:
    - File context
    - LLM provider communication
    - Edit parsing and application
    - Chat history
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize Coder with configuration.

        Args:
            config: Configuration dictionary

        Raises:
            ConnectionError: If provider is unavailable and user declines to continue
        """
        self.config = config
        self.files: dict[Path, FileContext] = {}
        self.history: list[Message] = []
        self.change_tracker = ChangeTracker()

        # v2.5.0: TODO now in ProjectContext (see project_context.todo)
        # self.current_todo removed — use self.project_context.todo instead

        # v2.4.0: Universal preview generator (gearbox)
        self.preview = ContentPreview()

        # Initialize provider
        self.provider: BaseProvider = create_provider(config.get("provider", {}))
        self.provider_name = config.get("provider", {}).get("name", "unknown")
        self.model = config.get("provider", {}).get("default_model", "")

        # Check connection
        ok, msg = self.provider.check_connection()
        if not ok:
            print(f"Warning: {msg}")
            response = input("Continue anyway? [y/n]: ")
            if response.lower() != "y":
                raise ConnectionError(msg)

        # v0.7.0: Initialize ChatStorage for raw conversation persistence
        self.chat_storage = None
        try:
            from pocketcoder.core.memory import ChatStorage
            self.chat_storage = ChatStorage(project_path=".")
        except Exception:
            pass  # ChatStorage is optional

        # v2.0.0: Initialize ProjectContext for SESSION_CONTEXT architecture
        self.project_context = ProjectContext(project_path=".")
        self.task_summarizer = TaskSummarizer(provider=self.provider)

        # v2.5.0: Set TodoStateMachine reference for tool functions
        from pocketcoder.tools import set_todo_machine
        set_todo_machine(self.project_context.todo)
        self._task_summarized = False  # Track if first request was summarized

        # v2.3.0: Initialize EpisodeManager for Episodic Memory
        # Episodes = chain of summaries from request to request (append-only)
        from pocketcoder.core.episodes import EpisodeManager
        self.episode_manager = EpisodeManager(Path(".pocketcoder"))

        # v2.1.0: Debug flag (set by run_interactive/run_agent_loop)
        self.debug = False

        # Trigger session start hook
        hooks.trigger("session_start")

    def add_file(self, path: Path | str) -> bool:
        """
        Add a file to the chat context.

        Args:
            path: Path to file

        Returns:
            True if file was added successfully
        """
        path = Path(path) if isinstance(path, str) else path
        path = resolve_path(str(path))

        if not path.exists():
            print(f"File not found: {path}")
            return False

        if is_binary(path):
            print(f"Cannot add binary file: {path}")
            return False

        # Check if already added
        if path in self.files:
            # Check if modified
            if path.stat().st_mtime > self.files[path].mtime:
                print(f"{path.name} was modified. Reloading...")
                self.files[path] = FileContext.from_path(path)
            else:
                print(f"{path.name} already in chat")
            return True

        # Check size limits
        max_lines = self.config.get("max_file_lines", 500)
        content = path.read_text()
        lines = len(content.splitlines())

        if lines > max_lines:
            print(f"Warning: {path.name} has {lines} lines (limit: {max_lines})")
            print("Adding anyway - consider using a range for large files")

        # Add file
        self.files[path] = FileContext.from_path(path)
        print(f"Added {path.name} ({lines} lines)")

        # Trigger hook
        hooks.trigger("file_added", path)

        return True

    def remove_file(self, path: Path | str) -> bool:
        """
        Remove a file from chat context.

        Args:
            path: Path to file

        Returns:
            True if file was removed
        """
        path = Path(path) if isinstance(path, str) else path
        path = resolve_path(str(path))

        if path in self.files:
            del self.files[path]
            print(f"Removed {path.name}")
            hooks.trigger("file_removed", path)
            return True
        else:
            print(f"{path.name} not in chat")
            return False

    def build_messages(self, user_input: str) -> list[Message]:
        """
        Build message list for LLM request.

        Args:
            user_input: Current user message

        Returns:
            List of Message objects
        """
        messages = []

        # 1. System prompt (v0.8.0: pass user_input for category detection)
        system_prompt = self._load_system_prompt(user_input)
        messages.append(Message("system", system_prompt))

        # 2. Files as context
        for path, ctx in self.files.items():
            file_msg = f"# {path.name}\n```\n{ctx.content}\n```"
            messages.append(Message("user", file_msg))
            messages.append(Message("assistant", "I see the file."))

        # 3. Chat history
        history_limit = self.config.get("history_limit", 10)
        for msg in self.history[-history_limit:]:
            messages.append(msg)

        # 4. Current request
        messages.append(Message("user", user_input))

        return messages

    def _load_system_prompt(self, user_input: str = "") -> str:
        """
        Load system prompt with memory injection.

        v0.8.0: Dynamic Context Discovery - compact prompts when enabled.

        Args:
            user_input: Current user input for category detection

        Returns:
            System prompt string
        """
        # v0.8.0: Use compact prompt if enabled (default: ON)
        if is_compact_mode_enabled(self.config):
            compact_prompt = build_compact_prompt(user_input)

            # Inject memory context
            try:
                from pocketcoder.core.memory import MemoryManager
                mm = MemoryManager()
                memory_context = mm.build_memory_context(max_facts=15)
                if memory_context:
                    compact_prompt += f"\n\n{memory_context}"
            except Exception:
                pass  # Memory system not critical

            return compact_prompt

        # Legacy: Full prompt (for backwards compatibility or weak models)
        base_prompt = """You are an AI coding assistant. You have access to 22 tools.

IMPORTANT: Always respond in the SAME LANGUAGE as the user's message!
If user writes in Russian → respond in Russian.
If user writes in English → respond in English.
But ALWAYS use XML format for tools regardless of language.

## Code Changes (SEARCH/REPLACE)

For ALL code changes, you MUST use this exact format:

filename.ext
<<<<<<< SEARCH
exact code to find
=======
new code
>>>>>>> REPLACE

Example for EDITING existing file:
main.py
<<<<<<< SEARCH
def hello():
    print("hi")
=======
def hello():
    print("Hello, World!")
>>>>>>> REPLACE

Example for CREATING new file (empty SEARCH):
newfile.html
<<<<<<< SEARCH
=======
<!DOCTYPE html>
<html>
<body>Hello</body>
</html>
>>>>>>> REPLACE

Rules:
- ALWAYS use <<<<<<< SEARCH and >>>>>>> REPLACE markers
- For new files: leave SEARCH section empty
- For edits: SEARCH must exactly match existing code

## SYSTEM INFO (IMPORTANT!)

""" + f"""OS: {platform.system()} ({platform.release()})
Platform: {platform.machine()}
""" + """
### OS-Specific Commands (USE CORRECT ONES!)

**macOS (Darwin):**
- USB devices: system_profiler SPUSBDataType
- PCI devices: system_profiler SPPCIDataType
- Disks: diskutil list
- Network: ifconfig
- Install packages: brew install X (NOT apt-get!)
- CPU info: sysctl -n machdep.cpu.brand_string
- /sys and /proc do NOT exist on macOS!

**Linux:**
- USB: lsusb
- PCI: lspci
- Disks: lsblk
- Network: ip a
- Install: apt-get / yum / pacman
- System info: /proc/cpuinfo, /sys/...

**Windows:**
- Devices: Get-PnpDevice (PowerShell)
- Disks: Get-Disk, wmic diskdrive list
- Network: ipconfig
- Install: winget / choco

⚠️ ALWAYS check OS before suggesting commands!
If command fails with "not found" → you probably used wrong OS command!

## Available Tools (22 total)

⚠️ CRITICAL: Tools MUST be in XML format, NOT in code blocks!

❌ WRONG (will NOT work):
```shell
list_files
```

❌ WRONG (will NOT work):
```
write_file
path: hello.py
content: print("hi")
```

✅ CORRECT (ONLY this format works):
<list_files><path>.</path><recursive>false</recursive></list_files>

✅ CORRECT:
<write_file><path>hello.py</path><content>print("hi")</content></write_file>

ALWAYS use XML tags like <tool_name>...</tool_name>, NEVER use code blocks for tools!

### FILE OPERATIONS

<read_file>
  <path>path/to/file</path>
</read_file>
Read file content. Supports fuzzy path matching.
⚠️ AFTER read_file, to MODIFY the file use:
- SEARCH/REPLACE (small edits)
- write_file (full rewrite)
- insert_at_line (add line)
NEVER call read_file on same file twice!

<write_file>
  <path>path/to/file</path>
  <content>file content here</content>
</write_file>
Create NEW file or completely overwrite existing. Use SEARCH/REPLACE for edits!

<list_files>
  <path>src/</path>
  <recursive>true</recursive>
</list_files>
List files in directory with tree structure.

<search_files>
  <pattern>TODO|FIXME</pattern>
  <path>.</path>
  <include>*.py</include>
</search_files>
Search for pattern in file contents (grep). Returns matching lines.

<find_file>
  <filename>config.py</filename>
</find_file>
Find FILE location by name. ⚠️ For FILES only, NOT folders! Use list_files to check folders.

<open_file>
  <path>index.html</path>
</open_file>
Open file in default application (browser, editor).
⚠️ After write_file → use SAME path for open_file! Don't search for it again.

<glob_files>
  <pattern>**/*.py</pattern>
  <path>.</path>
</glob_files>
Find files matching glob pattern.

### SHELL

<execute_command>
  <cmd>npm test</cmd>
</execute_command>
Execute shell command. Requires user confirmation.

### AGENT CONTROL

<ask_question>
  <question>Which database should I use?</question>
  <options>PostgreSQL,MySQL,SQLite</options>
</ask_question>
Ask user for clarification when unsure.

<update_todo>
  <tasks>[{"content": "Install deps", "status": "completed"}, {"content": "Add routes", "status": "in_progress"}]</tasks>
</update_todo>
Update task list to show progress.

<attempt_completion>
  <result>Created user authentication system</result>
  <command>npm test</command>
</attempt_completion>
Signal that the task is complete.

<switch_mode>
  <mode>architect</mode>
  <reason>User wants to discuss design</reason>
</switch_mode>
Switch mode: code (full access), architect (planning), ask (read-only), debug.

### ADVANCED EDIT

<apply_diff>
  <path>file.py</path>
  <diff>
@@ -1,3 +1,4 @@
 def hello():
-    print("hi")
+    print("Hello!")
+    return True
  </diff>
</apply_diff>
Apply unified diff. MUST have @@ header!

<multi_edit>
  <edits>[{"path": "a.py", "search": "old", "replace": "new"}]</edits>
</multi_edit>
Apply multiple SEARCH/REPLACE edits to files.

<insert_at_line>
  <path>file.py</path>
  <line_number>5</line_number>
  <content>new line content</content>
</insert_at_line>
Insert content at specific line.

<delete_lines>
  <path>file.py</path>
  <start_line>10</start_line>
  <end_line>15</end_line>
</delete_lines>
Delete lines from file.

## Tool Usage Guide

| User says | Use tool |
|-----------|----------|
| "what's in folder X?" | list_files |
| "find TODO" | search_files |
| "create file" | write_file |
| "where is file X?" | find_file |
| "show code" | read_file |
| "run tests" | execute_command |
| "which DB to use?" | ask_question |

## Rules

1. For editing EXISTING files: use SEARCH/REPLACE (not write_file!)
2. For creating NEW files: use write_file
3. For searching content: use search_files (not find_file)
4. For finding by name: use find_file
5. Shell commands require confirmation
6. Use attempt_completion when task is done
7. You CAN use ANY tool MULTIPLE TIMES! There is NO limit. If user asks to open_file again - just call open_file again!

## RECONNAISSANCE FIRST (CRITICAL!)

When user request contains CONDITIONS (if/or/unless/check):
- "create folder X OR use existing" → FIRST check what exists!
- "if file exists, edit it" → FIRST check if file exists!
- "find similar and..." → FIRST search, THEN decide!

### WRONG approach (Act First):
```
User: "create folder 'jan 16' or use existing"
You (WRONG): <list_files>...</list_files><write_file><path>jan 16/...</path>...  ← creates NEW without checking!
```
Problem: list_files and write_file execute TOGETHER — you don't SEE the result before writing!

### CORRECT approach (Think First):
```
User: "create folder 'jan 16' or use existing"

ITERATION 1 (Reconnaissance):
You: <list_files><path>.</path><recursive>false</recursive></list_files>
→ Result: shows "TESTING_19_JAN_2026/", "jan_15_project/"...

ITERATION 2 (Decision + Action):
You see similar folders exist!
You: <ask_question><question>Found similar folders: TESTING_19_JAN_2026, jan_15_project. Use one of these or create new 'jan 16'?</question><options>TESTING_19_JAN_2026,jan_15_project,Create new</options></ask_question>
→ Result: user chooses

ITERATION 3 (Execute):
You: <write_file>... (to chosen folder)
```

### Rule: For CONDITIONAL tasks — SEPARATE iterations!
1. Iteration 1: READ (list_files, read_file, search_files) — gather info
2. Iteration 2: DECIDE (ask_question if ambiguous, or proceed)
3. Iteration 3: WRITE (write_file, execute_command)

### For SIMPLE tasks (no conditions) — single iteration is OK:
```
User: "create file test.txt with hello"  ← no condition
You: <write_file><path>test.txt</path><content>hello</content></write_file>  ← OK, direct action
```

## TOOL EXECUTION ORDER (CRITICAL!)

When you call multiple tools, they execute IN THE ORDER you write them.
You MUST plan the correct order!

### Correct order pattern:
```
1. READ tools first (list_files, read_file) — understand context
2. WRITE tools next (write_file) — create/modify files
3. ACTION tools last (open_file, execute_command) — show result
4. attempt_completion — finish
```

### Example - "create folder TEST and index.html, then open":
CORRECT order:
<list_files><path>.</path><recursive>false</recursive></list_files>
<write_file><path>TEST/index.html</path><content>...</content></write_file>
<open_file><path>TEST/index.html</path></open_file>
<attempt_completion><result>Created and opened</result></attempt_completion>

WRONG order (open before create!):
<open_file><path>TEST/index.html</path></open_file>  ← ERROR: file not found!
<write_file><path>TEST/index.html</path><content>...</content></write_file>

### ALL tools execute in ONE iteration!
If you call 5 tools → ALL 5 will execute (dangerous tools ask confirmation).
Do NOT wait for each tool result — plan the full sequence upfront.

### If user cancels a tool:
You will receive "Cancelled: write_file" in results.
Then ask user why or offer alternative.

## IMPORTANT: Step-by-Step Execution

When user asks for MULTIPLE actions (create then edit, etc.):
1. FIRST use update_todo to create a plan
2. Execute ONE step at a time
3. Update todo after each step
4. NEVER skip steps or combine them!

Example - "create file X, then edit it":
Step 1: <update_todo><tasks>[{"content": "Create file", "status": "in_progress"}, {"content": "Edit file", "status": "pending"}]</tasks></update_todo>
Step 2: Create the file with SEARCH/REPLACE
Step 3: <update_todo><tasks>[{"content": "Create file", "status": "completed"}, {"content": "Edit file", "status": "in_progress"}]</tasks></update_todo>
Step 4: Edit the file with SEARCH/REPLACE
Step 5: <update_todo><tasks>[{"content": "Create file", "status": "completed"}, {"content": "Edit file", "status": "completed"}]</tasks></update_todo>
Step 6: <attempt_completion><result>Created and edited file</result></attempt_completion>

## CRITICAL: ALWAYS use tools!

❌ FORBIDDEN (common mistakes):
- "Sorry, I don't have access to the file system" → INSTEAD: call list_files or read_file
- "Here is the file content: [made up code]" → INSTEAD: first read_file, then respond
- "I cannot execute commands" → INSTEAD: call execute_command
- Just showing code in markdown → INSTEAD: call write_file to create file
- "Here is calculator code: ```python..." → INSTEAD: <write_file><path>calc.py</path><content>...</content></write_file>
- "I cannot reopen the file" → LIE! INSTEAD: just call <open_file> again!

✅ REQUIRED:
- Any question about files → first call tool (list_files, read_file), then respond
- Need to create file → write_file (don't just show code!)
- Need to edit file → SEARCH/REPLACE block
- Need to run command → execute_command

Example of proper file creation:
<write_file>
  <path>calc.py</path>
  <content>def add(a, b):
    return a + b
</content>
</write_file>

NOT LIKE THIS: "Here is code: ```python\ndef add(a,b)...\n```" - this will NOT create a file!

## CREATIVE TASKS (websites, games, apps, projects)

When user asks to CREATE something vague (site, game, app, bot, project):

### Step 1: Detect creative task
Keywords: "make website", "build game", "create app/bot/project"

### Step 2: Ask clarifying questions FIRST
Use <ask_question> to clarify:

For WEBSITE:
- Type: landing / SPA / multi-page?
- Stack: HTML+CSS+JS / React / Vue / other?
- Style: minimal / bright / corporate?

For GAME:
- Genre: platformer / arcade / puzzle / shooter?
- Graphics: 2D canvas / Phaser / Three.js?
- Controls: keyboard / mouse / touch?

For APP/BOT:
- Type: CLI / web / desktop / telegram bot?
- Language: Python / JavaScript / other?

### Step 3: Handle "you decide"
If user says:
- "you decide", "your choice", "whatever"

Then CHOOSE the simplest/fastest option and EXPLAIN:
"Ok, making a landing page with HTML+CSS+JS (fastest option, no dependencies)"

### Example flow:
User: make a website
You: <ask_question><question>What type of website?</question><options>Landing page,SPA (React/Vue),Multi-page</options></ask_question>
→ Result: Landing page
You: <ask_question><question>Need dark theme?</question><options>Yes,No,Toggle switch</options></ask_question>
→ Result: No
You: Great! Creating a landing page with HTML+CSS+JS, no dark theme.
[proceeds to create files]

### Example "you decide":
User: make a game
You: <ask_question><question>What genre?</question><options>Platformer,Arcade,Puzzle,Shooter</options></ask_question>
→ Result: you decide
You: Ok, making an arcade game on canvas — fastest option without external libraries.
[proceeds to create files]

## FEW-SHOT EXAMPLES

IMPORTANT: In examples below, "→" means "after tool executes, you receive result, then respond".
DO NOT output "[System shows result]" or "INSTRUCTION:" - these are NOT part of your response!

### FILE TOOLS

User: what files are here?
You: <list_files><path>.</path><recursive>false</recursive></list_files>
→ Result: main.py, config.py
You: There are 2 files: main.py and config.py.

User: read main.py
You: <read_file><path>main.py</path></read_file>
→ Result: print("hello")
You: main.py contains: print("hello")

User: create calc.py with add function
You: <write_file><path>calc.py</path><content>def add(a, b):
    return a + b</content></write_file>
→ Result: Created calc.py
You: Created calc.py with add function.

User: where is utils.py?
You: <find_file><filename>utils.py</filename></find_file>
→ Result: src/utils.py
You: utils.py is at src/utils.py

User: find TODO in code
You: <search_files><pattern>TODO</pattern><path>.</path></search_files>
→ Result: main.py:5: # TODO fix this
You: Found TODO in main.py line 5.

User: show all .py files
You: <glob_files><pattern>**/*.py</pattern><path>.</path></glob_files>
→ Result: main.py, src/utils.py
You: Found 2 Python files: main.py, src/utils.py

User: open index.html
You: <open_file><path>index.html</path></open_file>
→ Result: Opened
You: Opened index.html in browser.

### MULTIPLE TOOLS (correct order!)

User: create folder mysite with index.html, then open
You: <list_files><path>.</path><recursive>false</recursive></list_files><write_file><path>mysite/index.html</path><content><!DOCTYPE html>
<html><body><h1>Hello</h1></body></html></content></write_file><open_file><path>mysite/index.html</path></open_file>
→ Result: list_files: 5 files, write_file: Created mysite/index.html, open_file: Opened
You: Created folder mysite with index.html and opened in browser.

User: create game folder with index.html and game.js, then open
You: <write_file><path>game/index.html</path><content><!DOCTYPE html>
<html>
<head><title>Game</title></head>
<body><canvas id="c"></canvas><script src="game.js"></script></body>
</html></content></write_file><write_file><path>game/game.js</path><content>const canvas = document.getElementById('c');
const ctx = canvas.getContext('2d');
// Game code here</content></write_file><open_file><path>game/index.html</path></open_file>
→ Result: write_file: Created game/index.html, write_file: Created game/game.js, open_file: Opened
You: Created game folder with index.html and game.js, opened in browser.

### CONDITIONAL TASKS (check first!)

User: create folder jan16 or use existing if similar exists
ITERATION 1 (check first!):
You: <list_files><path>.</path><recursive>false</recursive></list_files>
→ Result: project/, TESTING_19_JAN_2026/, old_stuff/
ITERATION 2 (ask because similar exists):
You: Found similar folder TESTING_19_JAN_2026. <ask_question><question>Use it or create new jan16?</question><options>Use TESTING_19_JAN_2026,Create new jan16</options></ask_question>
→ Result: Use TESTING_19_JAN_2026
ITERATION 3 (now create file in chosen folder):
You: <write_file><path>TESTING_19_JAN_2026/index.html</path><content>...</content></write_file>
→ Result: Created
You: Created index.html in existing folder TESTING_19_JAN_2026.

User: if config.yaml exists, show it, otherwise create default
ITERATION 1 (check first!):
You: <find_file><filename>config.yaml</filename></find_file>
→ Result: Not found
ITERATION 2 (file not found → create):
You: config.yaml not found, creating default. <write_file><path>config.yaml</path><content>default: true</content></write_file>
→ Result: Created
You: Created config.yaml with default settings.

### SHELL

User: python version?
You: <execute_command><cmd>python --version</cmd></execute_command>
→ Result: Python 3.11.0
You: Python 3.11.0

User: what is full path to current directory?
You: <execute_command><cmd>pwd</cmd></execute_command>
→ Result: /home/user/project
You: Full path is /home/user/project

User: full path to main.py?
You: <execute_command><cmd>pwd</cmd></execute_command>
→ Result: /home/user/project
You: Full path is /home/user/project/main.py

### AGENT CONTROL

User: add caching
You: <ask_question><question>What type of caching?</question><options>LRU,Redis,File,In-memory</options></ask_question>
→ Result: User selected LRU
You: I'll use LRU caching.

User: make a website
You: <ask_question><question>What type of website?</question><options>Landing page,SPA (React),Multi-page</options></ask_question>
→ Result: you decide
You: Making a landing page with HTML+CSS+JS — fast, no dependencies.
[creates index.html, style.css, script.js]

User: create a game
You: <ask_question><question>What genre?</question><options>Platformer,Arcade,Puzzle,Shooter</options></ask_question>
→ Result: arcade
You: <ask_question><question>Graphics engine?</question><options>Canvas 2D,Phaser,Three.js</options></ask_question>
→ Result: you decide
You: I'll use Canvas 2D — simplest, no dependencies.
[creates game files]

User: tasks: 1) fix bug, 2) tests
You: <update_todo><tasks>[{"content":"Fix bug","status":"pending"},{"content":"Tests","status":"pending"}]</tasks></update_todo>
→ Result: Todo updated
You: Created task list.

User: let's discuss architecture
You: <switch_mode><mode>architect</mode><reason>User wants to discuss design</reason></switch_mode>
→ Result: Switched
You: Switched to architect mode.

### MEMORY (MUST use these tools!)

User: my name is Alex
You: <remember_fact><key>user_name</key><value>Alex</value></remember_fact>
→ Result: Saved
You: I'll remember your name is Alex.

User: what's my name?
You: <recall_fact><key>user_name</key></recall_fact>
→ Result: Alex
You: Your name is Alex.

User: what do you remember?
You: <list_facts></list_facts>
→ Result: user_name: Alex
You: I remember your name is Alex.

User: forget my name
You: <forget_fact><key>user_name</key></forget_fact>
→ Result: Forgot
You: Done, forgot your name.

### NOTES (MUST use these tools!)

User: note that project uses Flask
You: <save_note><category>architecture</category><content>Project uses Flask</content></save_note>
→ Result: Saved
You: Saved note about Flask.

User: what notes exist?
You: <load_notes></load_notes>
→ Result: architecture: Project uses Flask
You: Notes show project uses Flask.

### EDIT TOOLS

User: apply this diff
You: <apply_diff><path>main.py</path><diff>@@ -1,2 +1,3 @@
+import os
 def main():
     pass</diff></apply_diff>
→ Result: Applied
You: Applied diff.

User: replace foo with bar everywhere
You: <multi_edit><edits>[{"path":"main.py","search":"foo","replace":"bar"}]</edits></multi_edit>
→ Result: Edited
You: Replaced foo with bar.

User: insert import os at line 1
You: <insert_at_line><path>main.py</path><line_number>1</line_number><content>import os</content></insert_at_line>
→ Result: Inserted
You: Added import os at line 1.

User: delete lines 5-10
You: <delete_lines><path>main.py</path><start_line>5</start_line><end_line>10</end_line></delete_lines>
→ Result: Deleted
You: Deleted lines 5-10.

## HOW TO WORK

### Step 0: Know your workspace
BEFORE creating files or running commands:
1. Run <list_files><path>.</path><recursive>false</recursive></list_files> to see current structure
2. write_file AUTOMATICALLY creates parent directories (no need for mkdir!)
3. NEVER invent paths like "/home/user/..." — use pwd to get REAL path

Example - creating project:
WRONG: Just guess path and fail
RIGHT:
  → list_files to see where we are
  → write_file("myproject/src/main.py", code) — auto-creates myproject/src/
  → open_file to verify

### Step 1: Plan what tools you need
Before acting, think: "What do I need to answer this?"

Example - "full path to main.py?":
→ Need pwd to get directory
→ Then combine: directory + filename

Example - "create clock project":
→ list_files first (know where we are)
→ write_file for HTML (auto-creates dir)
→ write_file for JS
→ open_file to show result

### Step 2: Execute tools one by one
Call tool → get result → call next tool → get result → ...

### Step 3: Self-check before finishing
After each action, ask yourself:
- "Did I fully answer the user's request?"
- "What's left to do?"
- "Do I have enough info to respond?"

### COMMON MISTAKES — DON'T DO THIS!

❌ WRONG: find_file for folder
User: "create in folder myproject"
You: <find_file><filename>myproject</filename></find_file>  ← find_file is for FILES!
✅ RIGHT: <list_files><path>.</path>... → see myproject/ in results

❌ WRONG: read_file twice
User: "change token in bot.py"
You: <read_file><path>bot.py</path></read_file>
→ Result: TOKEN = "old"
You: <read_file><path>bot.py</path></read_file>  ← WHY again?
✅ RIGHT: Use SEARCH/REPLACE:
bot.py
<<<<<<< SEARCH
TOKEN = "old"
=======
TOKEN = "new_token"
>>>>>>> REPLACE

❌ WRONG: find_file + open_file after create
(just created "project/index.html")
User: "open it"
You: <find_file>index.html</find_file><open_file>...  ← finds DIFFERENT file!
✅ RIGHT: <open_file><path>project/index.html</path></open_file>  ← same path as write_file!

If YES → respond and finish
If NO → continue with next tool

Example self-check:
User asked: "create clock project"
I did: mkdir ✓, HTML ✓, JS ✓
Question: "Is task complete?"
Answer: Yes, clock project created → finish

Example self-check 2:
User asked: "full path to main.py"
I did: nothing yet
Question: "Can I answer?"
Answer: No, I don't know the path → need pwd first

NEVER guess. NEVER invent. ALWAYS use tools for real data!"""

        # INJECT MEMORY CONTEXT
        try:
            from pocketcoder.core.memory import MemoryManager
            mm = MemoryManager()
            memory_context = mm.build_memory_context(max_facts=15)
            if memory_context:
                base_prompt += f"\n\n{memory_context}"
        except Exception:
            pass  # Memory system not critical

        return base_prompt

    def send_message(self, user_input: str) -> ParsedResponse:
        """
        Send message to LLM and parse response.

        Args:
            user_input: User's message

        Returns:
            ParsedResponse with edits, questions, etc.
        """
        # Build messages
        messages = self.build_messages(user_input)

        # Hook: before_send
        messages = hooks.trigger("before_send", messages)
        if messages is None:
            return ParsedResponse(warnings=["Blocked by hook"])

        # Send to LLM
        try:
            from pocketcoder.core.condense import get_model_output_limit
            max_output = get_model_output_limit(self.model)
            response = self.provider.chat(
                messages,
                model=self.model,
                max_tokens=max_output,
                stream=False,
            )
            response_text = response.content
        except Exception as e:
            return ParsedResponse(warnings=[f"LLM error: {e}"])

        # Hook: after_response
        response_text = hooks.trigger("after_response", response_text)

        # Add to history
        self.history.append(Message("user", user_input))
        self.history.append(Message("assistant", response_text))

        # Parse response
        return parse_response(response_text)

    # =========================================================================
    # AGENT LOOP - Step-by-step execution
    # =========================================================================

    def _summarize_tool_result(self, tool_name: str, result: str, max_len: int = 200) -> str:
        """
        Summarize tool result for history to save context.

        Full result is shown to LLM in current iteration,
        but only summary is saved to task_context for future iterations.
        """
        # v1.0.2: ask_question - NEVER summarize! User's answer is critical.
        if tool_name == "ask_question":
            return result

        if len(result) <= max_len:
            return result

        lines = result.count('\n')
        chars = len(result)

        # Tool-specific summaries
        if tool_name == "read_file":
            # v2.5.1: Use ContentPreview gearbox — LLM needs to see actual code!
            return self.preview.generate(result).preview

        elif tool_name == "execute_command":
            # Keep more data for execute_command so LLM can use it later!
            # Previous: only first line → LLM forgets the actual data
            if chars <= 3000:
                # Small output - keep it all
                return result
            else:
                # Large output - keep first 2000 chars
                return result[:2000] + f"\n... [{chars - 2000} chars truncated, {lines} lines total]"

        elif tool_name == "list_files":
            return f"[Listed {lines} files/directories]"

        elif tool_name == "search_files":
            matches = result.count('\n')
            return f"[Search results: {matches} matches]"

        # Default: truncate with indicator
        return result[:max_len] + f"... [{chars - max_len} chars truncated]"

    def _send_raw(self, messages: list[Message]) -> str:
        """
        Send messages to LLM without modifying history.

        Args:
            messages: List of Message objects

        Returns:
            Raw response text from LLM
        """
        try:
            from pocketcoder.core.condense import get_model_output_limit
            max_output = get_model_output_limit(self.model)
            response = self.provider.chat(
                messages,
                model=self.model,
                max_tokens=max_output,
                stream=False,
            )
            return response.content
        except Exception as e:
            return f"[LLM Error: {e}]"

    def _format_current_todo(self) -> str:
        """
        v2.5.0: Format current_todo for injection into system prompt.

        Uses TodoStateMachine from project_context.
        Returns <current_todo>...</current_todo> block.
        """
        return self.project_context.todo.format_for_context()

    def _handle_rejection(self, tool_call) -> str:
        """
        v2.1.0: Handle user rejection of a tool call.

        Asks WHY rejected and returns feedback message for LLM.
        Loop continues instead of stopping - LLM can fix and retry.

        Args:
            tool_call: The rejected ToolCall

        Returns:
            Feedback message for LLM
        """
        from rich.console import Console
        from rich.panel import Panel

        console = Console()

        # Quick options for rejection reason
        options = [
            ("1", "Wrong file/path", "wrong_target"),
            ("2", "Code has issues", "code_issues"),
            ("3", "Want changes first", "want_changes"),
            ("4", "Cancel entire task", "cancel_all"),
        ]

        console.print(Panel(
            "\n".join([f"[{o[0]}] {o[1]}" for o in options]),
            title="[?] Why reject?",
            border_style="yellow"
        ))

        choice = input("> ").strip()

        # Find selected option
        reason_code = "unknown"
        reason_text = "User rejected"
        for opt in options:
            if choice == opt[0]:
                reason_text = opt[1]
                reason_code = opt[2]
                break

        # For some options, ask for details
        feedback = ""
        if reason_code in ("code_issues", "want_changes"):
            feedback = input("[?] Details (or Enter to skip): ").strip()

        # Handle cancel all
        if reason_code == "cancel_all":
            return f"""[TASK CANCELLED]
User cancelled the entire task.
Do NOT continue with current plan.

<thinking>
User cancelled. I should ask what they want to do instead.
</thinking>

Ask user what they want to do next with <ask_question>."""

        # Build feedback message for LLM
        tool_info = f"{tool_call.name}"
        if tool_call.name == "write_file":
            tool_info += f"({tool_call.params.get('path', '?')})"
        elif tool_call.name == "execute_command":
            tool_info += f"({tool_call.params.get('cmd', '?')[:30]})"

        msg = f"""[REJECTED] {tool_info}
Reason: {reason_text}"""

        if feedback:
            msg += f"\nUser feedback: \"{feedback}\""

        msg += """

<thinking>
User rejected my action. I need to analyze why and fix it.
</thinking>

Fix the issue based on user feedback, then try again with corrected tool call."""

        return msg

    def _has_pending_todo(self) -> bool:
        """
        v2.5.0: Check if there are pending or in_progress tasks.

        Uses TodoStateMachine from project_context.

        Returns:
            True if there are incomplete tasks
        """
        return self.project_context.todo.has_pending()

    # v2.0.0: Tool categories for context-aware result messages
    # This is the "gearbox" - universal mechanism for all tools
    INFO_GATHERING_TOOLS = {"ask_question", "read_file", "search_files", "list_files", "find_file", "glob_files"}
    ACTION_TOOLS = {"write_file", "execute_command", "apply_diff", "multi_edit", "insert_at_line", "delete_lines"}
    CONTROL_TOOLS = {"attempt_completion", "update_todo", "switch_mode"}
    MEMORY_TOOLS = {"remember_fact", "recall_fact", "list_facts", "forget_fact", "save_note", "load_notes"}

    def _build_result_message(
        self,
        tool_results: list,
        summary: str,
        pending_deferred: list
    ) -> str:
        """
        v2.1.0: Build context-aware result message with REFLECT prompt.

        Universal "gearbox" mechanism - no hardcoded per-tool logic.
        Now includes reflection prompt to make LLM think before acting.

        Args:
            tool_results: List of (ToolCall, result) tuples
            summary: Summarized results
            pending_deferred: List of deferred tools

        Returns:
            Context-aware instruction message for LLM
        """
        # Analyze executed tools by category
        executed_tools = {tc.name for tc, _ in tool_results} if tool_results else set()

        gathered_info = bool(executed_tools & self.INFO_GATHERING_TOOLS)
        did_action = bool(executed_tools & self.ACTION_TOOLS)
        has_pending = self._has_pending_todo()

        # Build deferred warning if any
        deferred_warning = ""
        if pending_deferred:
            deferred_info = []
            for t in pending_deferred:
                if t.name == "write_file":
                    path = t.params.get("path", "?")
                    deferred_info.append(f"write_file({path})")
                else:
                    deferred_info.append(t.name)
            deferred_warning = (
                f"\n[!!] DEFERRED: {', '.join(deferred_info)}\n"
                "Execute these BEFORE attempt_completion!"
            )

        # v2.1.0: REFLECT prompt - makes LLM think before acting
        reflect_prompt = """
REFLECT on this result:
1. Did it succeed or fail? (check for errors)
2. What should happen next? (check <todo> and <current>)
3. Any issues to address?

Write <thinking> with your analysis, then call the next tool."""

        # Build message based on context (universal logic)
        if gathered_info and has_pending:
            # Got new information + have plan → review plan and execute
            return f"""[TOOL RESULT]
{summary}
{deferred_warning}
{reflect_prompt}

After <thinking>, UPDATE <todo> if needed and EXECUTE next task."""

        elif gathered_info and not has_pending:
            # Got info but no plan yet → create plan and execute
            return f"""[TOOL RESULT]
{summary}
{reflect_prompt}

After <thinking>, CREATE <todo> with tasks and EXECUTE first task."""

        elif did_action and has_pending:
            # Did action, more tasks remain → continue
            return f"""[TOOL RESULT]
{summary}
{deferred_warning}
{reflect_prompt}

After <thinking>, continue with next pending task."""

        elif did_action and not has_pending:
            # Did action, no more tasks → maybe done
            return f"""[TOOL RESULT]
{summary}
{reflect_prompt}

After <thinking>:
- If ALL done: <attempt_completion>
- If more work needed: update <todo> and continue"""

        else:
            # Default fallback
            return f"""[TOOL RESULT]
{summary}
{deferred_warning}
{reflect_prompt}

After <thinking>, call next tool: <write_file>, <read_file>, or <attempt_completion>"""

    # v2.5.0: _merge_todo REMOVED — TodoStateMachine handles state

    def _build_nudge_message(self) -> str:
        """
        v1.0.7: Build specific nudge message with TODO tasks and tool examples.

        Returns:
            Nudge message that shows pending tasks and how to call tools
        """
        lines = [
            "[!] You have PENDING tasks but didn't call any tools!",
            "",
            "YOUR TODO:"
        ]

        # v2.5.0: Add formatted TODO tasks from TodoStateMachine
        for task in self.project_context.todo.tasks:
            if task.status == "completed":
                lines.append(f"  [ok] {task.text}")
            elif task.status == "in_progress":
                lines.append(f"  [~] {task.text} ← DO THIS NOW")
            else:
                lines.append(f"  [ ] {task.text}")

        lines.extend([
            "",
            "ACTION REQUIRED - call ONE of these tools:",
            "",
            "To CREATE a file:",
            "<write_file><path>filename.py</path><content>your code here</content></write_file>",
            "",
            "To READ a file first:",
            "<read_file><path>filename.py</path></read_file>",
            "",
            "To LIST files:",
            "<list_files><path>.</path></list_files>",
            "",
            "When ALL tasks done:",
            "<attempt_completion><result>Summary of what was created</result></attempt_completion>",
            "",
            "DO NOT describe what you will do - CALL A TOOL NOW!"
        ])

        return "\n".join(lines)

    def _build_agent_messages(self, task_context: list[Message]) -> list[Message]:
        """
        Build messages for agent loop iteration.

        v0.7.0: Added grep-based context injection from ChatStorage.
        v0.8.0: Pass user_input for dynamic prompt category detection.
        v2.0.0: SESSION_CONTEXT injection - LLM always sees full project context.

        Args:
            task_context: Current task's message history

        Returns:
            Full message list for LLM
        """
        messages = []

        # v0.8.0: Extract user_input from task_context for category detection
        user_input = ""
        for msg in task_context:
            if msg.role == "user":
                user_input = msg.content
                break

        # 1. System prompt (v0.8.0: pass user_input)
        system_prompt = self._load_system_prompt(user_input)

        # v2.0.0: Inject SESSION_CONTEXT (replaces grep context and TODO injection)
        try:
            session_context_xml = build_session_context_xml(
                project_context=self.project_context,
                current_todo=self.project_context.todo.to_list(),  # v2.5.0
                user_input=user_input,
                conversation=None,  # Don't extract soft data on every request (expensive)
                provider=None,
                include_soft_data=False,
                episode_manager=self.episode_manager  # v2.3.0: Episodic Memory
            )
            system_prompt += f"\n\n{session_context_xml}"

            # v2.1.0: Debug output for SESSION_CONTEXT
            if self.debug:
                try:
                    from pocketcoder.ui.feedback import show_session_context
                    show_session_context(session_context_xml, enabled=True)
                except Exception:
                    print(f"[D] SESSION_CONTEXT: {len(session_context_xml)} chars")
        except Exception as e:
            # Fallback to old TODO injection if SESSION_CONTEXT fails
            if self.debug:
                print(f"[D] SESSION_CONTEXT failed: {e}")
            todo_context = self._format_current_todo()
            if todo_context:
                system_prompt += f"\n\n{todo_context}"

        messages.append(Message("system", system_prompt))

        # 2. Files as context
        for path, ctx in self.files.items():
            file_msg = f"# {path.name}\n```\n{ctx.content}\n```"
            messages.append(Message("user", file_msg))
            messages.append(Message("assistant", "I see the file."))

        # 3. Global history (limited)
        history_limit = self.config.get("history_limit", 10)
        for msg in self.history[-history_limit:]:
            messages.append(msg)

        # 4. Task context (current task's conversation)
        for msg in task_context:
            messages.append(msg)

        return messages

    def _get_grep_context(self, task_context: list[Message]) -> str:
        """
        v0.7.0: Get relevant context from ChatStorage via grep.

        Extracts keywords from current question and searches chat history.

        Args:
            task_context: Current task's message history

        Returns:
            Formatted context string (max ~500 tokens)
        """
        if not self.chat_storage or not task_context:
            return ""

        # Get current question
        current_question = ""
        for msg in task_context:
            if msg.role == "user":
                current_question = msg.content
                break

        if not current_question:
            return ""

        # Extract keywords
        from pocketcoder.core.memory.chat_storage import extract_keywords
        keywords = extract_keywords(current_question, max_keywords=5)

        if not keywords:
            return ""

        # Get context
        context = self.chat_storage.get_context_for(keywords, max_tokens=500)

        return context

    def _has_completion(self, parsed: ParsedResponse) -> bool:
        """
        Check if response contains attempt_completion.

        Args:
            parsed: Parsed LLM response

        Returns:
            True if task is complete
        """
        for tc in parsed.tool_calls:
            if tc.name == "attempt_completion":
                return True
        return False

    # v0.7.0: Tool categories for reconnaissance enforcement
    READ_TOOLS = {"list_files", "read_file", "find_file", "search_files", "glob_files"}
    WRITE_TOOLS = {"write_file", "execute_command", "apply_diff", "multi_edit", "insert_at_line", "delete_lines"}

    def _apply_reconnaissance_rule(
        self,
        tools: list[ToolCall]
    ) -> tuple[list[ToolCall], str, list[ToolCall]]:
        """
        v0.7.0: Apply reconnaissance first rule.

        If both READ and WRITE tools are present, execute only READ tools
        and defer WRITE tools to next iteration.

        This prevents LLM from creating files before seeing what exists.

        Args:
            tools: List of tool calls

        Returns:
            Tuple of (tools_to_execute, warning_message, deferred_tools)
        """
        # Categorize tools
        read_tools = [t for t in tools if t.name in self.READ_TOOLS]
        write_tools = [t for t in tools if t.name in self.WRITE_TOOLS]
        other_tools = [t for t in tools if t.name not in self.READ_TOOLS and t.name not in self.WRITE_TOOLS]

        # If both READ and WRITE present --> execute only READ, defer WRITE
        if read_tools and write_tools:
            deferred_info = []
            for t in write_tools:
                if t.name == "write_file":
                    path = t.params.get("path", "?")
                    deferred_info.append(f"write_file(path='{path}')")
                else:
                    deferred_info.append(t.name)

            warning = (
                f"[!] RECONNAISSANCE FIRST: Deferred {len(write_tools)} write tool(s):\n"
                f"    {', '.join(deferred_info)}\n"
                "[!!] BLOCKED: You MUST call deferred tools in next iteration before attempt_completion."
            )
            return read_tools + other_tools, warning, write_tools

        # v1.0.3: ONE WRITE AT A TIME
        # If multiple write_file calls → execute only FIRST, defer rest
        # This gives user control and shows progress step-by-step (like Claude Code)
        if len(write_tools) > 1:
            first_write = write_tools[0]
            deferred_writes = write_tools[1:]

            deferred_info = []
            for t in deferred_writes:
                if t.name == "write_file":
                    path = t.params.get("path", "?")
                    deferred_info.append(f"write_file(path='{path}')")
                else:
                    deferred_info.append(t.name)

            warning = (
                f"[!] ONE AT A TIME: Executing 1/{len(write_tools)} write tool(s).\n"
                f"    Queued: {', '.join(deferred_info)}\n"
                "[i] Remaining tools will execute in next iterations."
            )
            return other_tools + [first_write], warning, deferred_writes

        # Otherwise execute all
        return tools, "", []

    def _execute_one_action(self, parsed: ParsedResponse) -> tuple[str | None, str, list[ToolCall]]:
        """
        Execute ALL actions from parsed response.

        v0.7.0: Reconnaissance enforcement - if READ + WRITE tools are mixed,
        execute only READ tools first and defer WRITE to next iteration.

        Executes all tool calls (except attempt_completion) in sequence.
        Dangerous tools (write_file, execute_command) will ask for confirmation.

        Args:
            parsed: Parsed LLM response

        Returns:
            Tuple of (action_type, result_message, deferred_tools, tool_results)
            action_type: "tool", "edit", or None
            deferred_tools: list of ToolCall that were deferred by reconnaissance rule
            tool_results: list of (ToolCall, result) for each executed tool (BUG 2+4 fix)
        """
        # 1. Execute ALL tool calls (except attempt_completion)
        active_tools = [tc for tc in parsed.tool_calls if tc.name != "attempt_completion"]
        if active_tools:
            # v0.7.0: RECONNAISSANCE ENFORCEMENT
            # If both READ and WRITE tools are present, execute only READ first
            tools_to_execute, deferred_warning, deferred_tools = self._apply_reconnaissance_rule(active_tools)

            # Execute tools
            results = self.execute_tools(tools_to_execute)
            result_text = "\n".join(results)

            # BUG 2+4 fix: Return individual tool results for separate callbacks
            tool_results = list(zip(tools_to_execute, results))

            # Add deferred warning if applicable
            if deferred_warning:
                result_text += f"\n\n{deferred_warning}"

            # Check if any tool was cancelled
            if any("Cancelled:" in r for r in results):
                result_text += "\n\n[Some tools were cancelled by user]"

            return ("tool", result_text, deferred_tools, tool_results)

        # 2. Apply ALL edits
        if parsed.edits:
            results = self.process_edits(parsed.edits, auto_apply=False)
            result_text = "\n".join(results)

            return ("edit", result_text, [], [])

        # 3. No actions
        return (None, "", [], [])

    def run_agent_loop(
        self,
        user_input: str,
        max_iterations: int = 10,
        on_iteration: callable = None,
        debug: bool = True,
    ) -> ParsedResponse:
        """
        Run agentic execution loop.

        Executes task step-by-step:
        1. Send user input to LLM
        2. Execute ONE action (tool or edit)
        3. Feed result back to LLM
        4. Repeat until attempt_completion or max_iterations

        Args:
            user_input: User's task description
            max_iterations: Maximum number of iterations
            on_iteration: Optional callback(iteration, action_type, result)
            debug: Show debug info

        Returns:
            Final ParsedResponse
        """
        # v2.1.0: Store debug flag for use in other methods
        self.debug = debug

        # === AUTO-CONDENSE CHECK ===
        from pocketcoder.core.condense import ContextCondenser, estimate_messages_tokens

        condenser = ContextCondenser(self.provider, model_name=self.model)
        status, tokens = condenser.check_context(self.history)

        if status in ("critical", "overflow"):
            # Auto-condense
            if on_iteration:
                on_iteration(0, "debug", f"[>] Auto-condensing ({status})...")

            try:
                new_history, summary = condenser.condense(self.history, self.model)
                old_tokens = estimate_messages_tokens(self.history)
                new_tokens = estimate_messages_tokens(new_history)
                self.history = new_history

                if on_iteration:
                    saved = old_tokens - new_tokens
                    on_iteration(0, "debug", f"[ok] Condensed: {old_tokens} --> {new_tokens} tokens (saved {saved})")
            except Exception as e:
                if on_iteration:
                    on_iteration(0, "debug", f"[!] Condense failed: {e}")

        # Task-local context (doesn't pollute global history)
        # v2.0.0: No more keyword detection for continue/run
        # LLM understands context from SESSION_CONTEXT injection
        task_context: list[Message] = []
        task_context.append(Message("user", user_input))

        # v2.3.0: Start new episode for this user request
        # Episode = one user request -> outcome cycle
        self.episode_manager.start_episode(user_input)

        # v2.0.0: Summarize task on first request (stores in ProjectContext)
        if not self._task_summarized:
            try:
                task_summary = self.task_summarizer.summarize(user_input, self.provider)
                self.project_context.set_task(task_summary)
                self.project_context.start_session()
                self._task_summarized = True
                if debug and on_iteration:
                    on_iteration(0, "debug", f"[>] Task: {task_summary.summary[:100]}...")
            except Exception as e:
                if debug and on_iteration:
                    on_iteration(0, "debug", f"[!] Task summarization failed: {e}")

        # v1.0.6: Track read files for duplicate detection (returns content + reminder)
        self._read_files_this_task = set()

        # v0.7.0: Save user input to ChatStorage
        if self.chat_storage:
            try:
                self.chat_storage.append("user", user_input)
            except Exception:
                pass

        final_parsed = None
        final_response_text = ""

        # Track errors to detect loops
        last_error = None
        error_count = 0
        MAX_SAME_ERRORS = 3

        # Track if tool was executed (for smart exit)
        tool_was_executed = False
        original_question = user_input  # Save for context in results

        # v1.0.7: Track no-progress iterations (for anti-loop protection)
        no_progress_count = 0
        MAX_NO_PROGRESS = 3  # Max iterations without tools when TODO has pending tasks

        # v1.0.5: Duplicate Detection REMOVED - replaced by File State Machine
        # See file_cache.py for new approach

        # v0.7.0: Track deferred tools (from reconnaissance rule)
        pending_deferred: list[ToolCall] = []

        # v2.5.0: TODO validation tracking REMOVED — TodoStateMachine handles state

        # Stats tracking
        start_time = time.time()
        total_input_tokens = 0
        total_output_tokens = 0
        tools_executed = 0
        iterations_count = 0

        for iteration in range(max_iterations):
            iterations_count += 1

            # Debug: show iteration
            if debug and on_iteration:
                on_iteration(iteration + 1, "debug", f"--- Iteration {iteration + 1} ---")

            # v1.0.3: AUTO-EXECUTE deferred tools (don't wait for LLM)
            # This ensures all queued write_file calls are executed one by one
            if pending_deferred:
                next_tool = pending_deferred[0]  # Get next deferred tool

                if debug and on_iteration:
                    on_iteration(iteration + 1, "debug",
                        f"[>] Auto-executing deferred: {next_tool.name}")

                # Execute single tool (with confirmation if dangerous)
                tool_result = self.execute_tools([next_tool])
                result_str = tool_result[0] if tool_result else "[x] Failed to execute"

                # Remove from pending ONLY if executed (not cancelled)
                if "Cancelled:" not in result_str:
                    pending_deferred.pop(0)

                    # Callback for UI
                    if on_iteration:
                        tool_info = {"name": next_tool.name, "params": next_tool.params}
                        on_iteration(iteration + 1, "tool", result_str, tool_info)

                    # Add to context for LLM
                    remaining = len(pending_deferred)
                    context_msg = f"[Auto-executed deferred tool: {next_tool.name}]\n{result_str}"
                    if remaining > 0:
                        context_msg += f"\n[{remaining} more deferred tool(s) pending]"
                    task_context.append(Message("user", context_msg))

                    tool_was_executed = True
                    tools_executed += 1
                else:
                    # User cancelled - clear all pending and let LLM know
                    pending_deferred.clear()
                    task_context.append(Message("user",
                        f"[User cancelled deferred tool: {next_tool.name}]\n"
                        "All pending tools cleared. Continue or complete task."))

                continue  # Next iteration (execute next deferred or call LLM)

            # 1. Build messages and send to LLM
            messages = self._build_agent_messages(task_context)

            # Estimate input tokens
            input_text = " ".join(m.content for m in messages)
            total_input_tokens += len(input_text) // 4  # ~4 chars per token

            response_text = self._send_raw(messages)
            final_response_text = response_text

            # Estimate output tokens
            total_output_tokens += len(response_text) // 4

            # v0.7.0: Save assistant response to ChatStorage
            if self.chat_storage:
                try:
                    self.chat_storage.append("assistant", response_text[:5000])
                except Exception:
                    pass

            # AUTO-EXTRACT facts from conversation
            try:
                from pocketcoder.core.memory import MemoryManager
                mm = MemoryManager()
                # Get user message from task_context
                user_msg = task_context[0].content if task_context else ""
                extracted = mm.auto_extract(user_msg, response_text)
                if extracted and debug and on_iteration:
                    keys = list(extracted.keys())
                    on_iteration(iteration + 1, "debug", f"💾 Auto-saved: {keys}")
            except Exception:
                pass  # Memory extraction not critical

            # Debug: show raw response (truncated)
            if debug and on_iteration:
                preview = response_text[:500].replace('\n', '\\n')
                on_iteration(iteration + 1, "debug", f"LLM: {preview}...")

            # v2.1.0: Show <thinking> block to user (visible reasoning)
            try:
                from pocketcoder.ui.feedback import show_thinking_if_present
                response_for_parse = show_thinking_if_present(response_text, debug=debug)
            except Exception:
                response_for_parse = response_text

            # 2. Parse response
            parsed = parse_response(response_for_parse)
            final_parsed = parsed

            # v2.5.0: TODO merge REMOVED — TodoStateMachine handles state via tools
            # LLM uses add_todo/mark_done/remove_todo instead of <todo> block

            # v1.1.0: Parse error feedback
            if parsed.parse_errors and not parsed.tool_calls:
                error_feedback = "[!] Tool format errors detected:\n"
                for err in parsed.parse_errors:
                    error_feedback += f"  - {err}\n"
                error_feedback += "Correct format: <tool_name><param>value</param></tool_name>"
                task_context.append(Message("user", error_feedback))
                if debug and on_iteration:
                    on_iteration(iteration + 1, "debug", f"Parse errors: {parsed.parse_errors}")

            # v1.0.5: ANTI-LOOP removed - File State Machine handles duplicates
            # Duplicate read_file calls now return from cache with reminder

            # Debug: show what was parsed
            if debug and on_iteration:
                on_iteration(iteration + 1, "debug",
                    f"Parsed: edits={len(parsed.edits)}, tools={[tc.name for tc in parsed.tool_calls]}, todo={len(parsed.todo)}")

            # v2.5.0: TODO VALIDATION REMOVED
            # No more "[!] Missing <todo>" retry — TodoStateMachine handles state via tools
            # LLM uses add_todo/mark_done/remove_todo, no need to parse <todo> block

            # 3. Add assistant response to task context
            task_context.append(Message("assistant", response_text))

            # 4. Execute ONE action FIRST (before checking completion)
            action_type, result, new_deferred, tool_results = self._execute_one_action(parsed)

            # v0.7.0: Track deferred tools
            if new_deferred:
                pending_deferred.extend(new_deferred)
                if debug and on_iteration:
                    deferred_names = [t.name for t in new_deferred]
                    on_iteration(iteration + 1, "debug",
                        f"[!] Deferred {len(new_deferred)} tool(s): {', '.join(deferred_names)}")

            if action_type:
                # BUG 2+4 fix: Callback for EACH tool separately (not joined)
                if on_iteration:
                    if action_type == "tool" and tool_results:
                        # Call callback for each tool with its own result
                        for tc, res in tool_results:
                            tool_info = {"name": tc.name, "params": tc.params}
                            on_iteration(iteration + 1, "tool", res, tool_info)
                    else:
                        # Edit or other action - single callback
                        on_iteration(iteration + 1, action_type, result, None)

                # Check for cancellation - stop loop
                if "Cancelled:" in result:
                    if debug and on_iteration:
                        on_iteration(iteration + 1, "debug", "[STOP] User cancelled - stopping loop")
                    break

                # v2.2.0: M4 fix - Clear deferred tools on task cancellation
                if "[TASK CANCELLED]" in result:
                    pending_deferred.clear()
                    if debug and on_iteration:
                        on_iteration(iteration + 1, "debug", "[STOP] Task cancelled - cleared deferred tools")
                    break

                # Check for error loop (same error 3+ times = stop)
                if "[x]" in result or "Error" in result:
                    if result == last_error:
                        error_count += 1
                        if error_count >= MAX_SAME_ERRORS:
                            if on_iteration:
                                on_iteration(iteration + 1, "debug",
                                    f"Stopping: same error {MAX_SAME_ERRORS}+ times")
                            break
                    else:
                        last_error = result
                        error_count = 1
                else:
                    # Reset error tracking on success
                    last_error = None
                    error_count = 0

                # Add result to task context for next iteration
                # LLM sees full result via callback, but we save summary to reduce context
                # v1.0.2: Summarize EACH tool separately, not just the first one!
                if action_type == "tool" and tool_results:
                    # Summarize EACH tool separately
                    summaries = []
                    for tc, res in tool_results:
                        tool_summary = self._summarize_tool_result(tc.name, res)
                        summaries.append(f"[{tc.name}] {tool_summary}")
                        # v2.3.0: Track action in current episode
                        self.episode_manager.add_action(tc.name, tc.params, res)
                    summary = "\n".join(summaries)
                    tool_was_executed = True  # Mark for smart exit
                    tools_executed += 1
                else:
                    summary = result[:300] if len(result) > 300 else result

                # v1.0.3: Check if executed tool was a deferred one --> clear from pending
                # Fixed: compare by (name, path) not just name to avoid clearing ALL write_files
                if action_type == "tool" and tool_results and pending_deferred:
                    # Build set of actually executed (name, path) pairs
                    executed_set = set()
                    for tc, res in tool_results:
                        if tc.name == "write_file":
                            path = tc.params.get("path", "")
                            executed_set.add(("write_file", path))
                        else:
                            executed_set.add((tc.name, None))

                    # Remove only matching tools from pending
                    new_pending = []
                    for t in pending_deferred:
                        if t.name == "write_file":
                            path = t.params.get("path", "")
                            if ("write_file", path) not in executed_set:
                                new_pending.append(t)
                        else:
                            if (t.name, None) not in executed_set:
                                new_pending.append(t)
                    pending_deferred = new_pending

                # v2.0.0: Build context-aware result message based on tool categories
                result_message = self._build_result_message(
                    tool_results=tool_results,
                    summary=summary,
                    pending_deferred=pending_deferred
                )
                task_context.append(Message("user", result_message))

                # v2.5.0: Render TODO AFTER tool results (uses TodoStateMachine)
                if self.project_context.todo.tasks and on_iteration:
                    on_iteration(iteration + 1, "todo", self.project_context.todo.to_list())

                # v2.3.0: Check if episode needs checkpoint (too large)
                if self.episode_manager.should_checkpoint():
                    if debug and on_iteration:
                        on_iteration(iteration + 1, "debug",
                            "[!] Episode large - requesting checkpoint")
                    checkpoint_prompt = (
                        "[CHECKPOINT REQUIRED] Current episode is getting large. "
                        "Please save your progress using checkpoint_progress(done, remaining) "
                        "where 'done' is what you completed and 'remaining' is bullet points of what's left. "
                        "User will continue with next message."
                    )
                    task_context.append(Message("user", checkpoint_prompt))

                # Continue to next iteration
                continue

            # 5. Check for completion AFTER actions are done
            if self._has_completion(parsed):
                # v0.7.0: BLOCK completion if deferred tools pending
                if pending_deferred:
                    deferred_info = [t.name for t in pending_deferred]
                    if debug and on_iteration:
                        on_iteration(iteration + 1, "debug",
                            f"[!!] BLOCKED completion: {len(pending_deferred)} deferred tool(s) pending: {', '.join(deferred_info)}")

                    # Add blocking message to context
                    block_message = (
                        f"[!!] COMPLETION BLOCKED: You have {len(pending_deferred)} deferred tool(s) that must be executed first:\n"
                        f"    {', '.join(deferred_info)}\n"
                        "Execute these tools now, then call attempt_completion."
                    )
                    task_context.append(Message("user", block_message))
                    continue  # Force next iteration

                # v2.5.0: Render TODO before completion (uses TodoStateMachine)
                if self.project_context.todo.tasks and on_iteration:
                    on_iteration(iteration + 1, "todo", self.project_context.todo.to_list())

                # Extract completion message
                for tc in parsed.tool_calls:
                    if tc.name == "attempt_completion":
                        result = tc.params.get("result", "Task completed")
                        # v2.3.0: Close episode with completion result
                        self.episode_manager.add_action("attempt_completion", tc.params, result)
                        self.episode_manager.close_episode(self.project_context.todo.to_list())
                        if on_iteration:
                            on_iteration(iteration + 1, "completion", result, None)
                break

            # 6. SMART EXIT: No actions found
            # v1.0.7: Check for pending TODO tasks FIRST (before tool_was_executed!)
            if self._has_pending_todo():
                # LLM didn't call tools but has pending tasks → nudge to continue
                no_progress_count += 1

                if no_progress_count >= MAX_NO_PROGRESS:
                    # Anti-loop: too many iterations without progress
                    if debug and on_iteration:
                        on_iteration(iteration + 1, "debug",
                            f"[!] Stopping: {no_progress_count} iterations without tools (pending tasks exist)")
                    break

                # Nudge LLM to use tools
                if debug and on_iteration:
                    on_iteration(iteration + 1, "debug",
                        f"[!] Pending tasks exist but no tools called ({no_progress_count}/{MAX_NO_PROGRESS})")

                # v1.0.7: Build specific nudge with TODO tasks and tool examples
                nudge_message = self._build_nudge_message()
                task_context.append(Message("user", nudge_message))
                continue  # Force another iteration
            elif tool_was_executed:
                # No pending tasks + tool was executed → task truly done
                if debug and on_iteration:
                    on_iteration(iteration + 1, "debug", "[ok] Smart exit: no pending tasks + tool executed = done")
                break
            else:
                # No pending tasks, no tools → just conversational response
                if debug and on_iteration:
                    on_iteration(iteration + 1, "debug", "No actions found, ending loop")
                break

        # Save to global history: only user input and final response
        self.history.append(Message("user", user_input))
        self.history.append(Message("assistant", final_response_text))

        # v2.3.0: Close episode if still open (no attempt_completion, max_iterations, or break)
        if self.episode_manager.current:
            self.episode_manager.close_episode(self.project_context.todo.to_list())

        # Add stats to response
        if final_parsed:
            final_parsed.stats = AgentStats(
                elapsed_time=time.time() - start_time,
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                iterations=iterations_count,
                tools_executed=tools_executed,
            )

        return final_parsed

    def process_edits(self, edits: list[Edit], auto_apply: bool = False) -> list[str]:
        """
        Process and apply edits.

        Args:
            edits: List of Edit objects
            auto_apply: If True, apply without confirmation

        Returns:
            List of result messages
        """
        results = []
        by_file = group_edits_by_file(edits)

        for filename, file_edits in by_file.items():
            path = resolve_path(filename)

            # Show diff
            if path.exists():
                old_content = path.read_text()
                new_content = apply_edits_to_content(old_content, file_edits)
                diff = generate_diff(filename, old_content, new_content)
                print(f"\n{filename}:")
                print(diff)
            else:
                print(f"\n{filename} (new file):")
                print(file_edits[0].replace[:500])

            # Confirm
            if not auto_apply:
                response = input("\nApply? [y/n]: ")
                if response.lower() != "y":
                    results.append(f"Skipped {filename}")
                    continue

            # Apply edits
            for edit in file_edits:
                # Hook: before_edit
                result = hooks.trigger("before_edit", path, edit.search, edit.replace)
                if result is None:
                    results.append(f"Blocked by hook: {filename}")
                    continue

                # Record for undo
                if path.exists():
                    self.change_tracker.record(path, path.read_text(), "")

                # Apply
                success, error = apply_edit(path, edit)

                if success:
                    results.append(f"Applied to {filename}")
                    # Update file context
                    if path.exists():
                        self.files[path] = FileContext.from_path(path)
                    # Hook: after_edit
                    hooks.trigger("after_edit", path, "", "")
                else:
                    results.append(f"Failed {filename}: {error}")

        return results

    def execute_tools(self, tool_calls: list[ToolCall]) -> list[str]:
        """
        Execute parsed tool calls.

        v0.6.0: Also extracts facts from tool results (POINTER/VALUE).

        Args:
            tool_calls: List of ToolCall objects from parser

        Returns:
            List of result messages
        """
        from pocketcoder.tools import is_tool_allowed
        from pocketcoder.core.memory import MemoryManager
        import json

        results = []
        tool_results_for_memory = []  # v0.6.0: For fact extraction

        for tool_call in tool_calls:
            tool = get_tool(tool_call.name)
            if not tool:
                results.append(f"Unknown tool: {tool_call.name}")
                continue

            # Check if tool allowed in current mode
            if not is_tool_allowed(tool_call.name):
                from pocketcoder.tools import get_current_mode
                results.append(f"Tool '{tool_call.name}' not allowed in {get_current_mode()} mode")
                continue

            # Confirmation for dangerous tools
            if tool.get("dangerous"):
                # v2.1.0: Flush stdin to prevent input collision
                # (user typing during LLM thinking shouldn't affect confirmation)
                try:
                    import sys
                    import termios
                    termios.tcflush(sys.stdin, termios.TCIFLUSH)
                except Exception:
                    pass  # Non-Unix systems

                if tool_call.name == "execute_command":
                    cmd = tool_call.params.get("cmd", tool_call.params.get("command", ""))
                    is_dangerous = is_dangerous_command(cmd)
                    if is_dangerous:
                        print(f"\n[!] Dangerous command: {cmd}")
                    confirm = input(f"\nRun '{cmd}'? [y/n]: ")

                elif tool_call.name == "write_file":
                    # v1.0.4: Show preview before confirmation
                    from rich.console import Console
                    from rich.panel import Panel
                    from rich.syntax import Syntax

                    console = Console()
                    path = tool_call.params.get("path", "")
                    content = tool_call.params.get("content", "")

                    # Prepare preview (first 12 lines)
                    lines = content.split('\n')
                    preview_lines = lines[:12]
                    preview_text = '\n'.join(preview_lines)
                    more = len(lines) - 12 if len(lines) > 12 else 0

                    # Detect language for syntax highlighting
                    ext = path.split('.')[-1] if '.' in path else 'txt'
                    lang_map = {'py': 'python', 'js': 'javascript', 'ts': 'typescript',
                                'md': 'markdown', 'json': 'json', 'yaml': 'yaml', 'yml': 'yaml'}
                    lang = lang_map.get(ext, ext)

                    # Show preview panel
                    try:
                        syntax = Syntax(preview_text, lang, theme="monokai", line_numbers=False)
                        title = f"[bold]Write: {path}[/bold]"
                        if more > 0:
                            title += f" [dim](+{more} more lines)[/dim]"
                        panel = Panel(syntax, title=title, border_style="green", padding=(0, 1))
                        console.print(panel)
                    except Exception:
                        # Fallback to simple output
                        print(f"\n--- {path} ---")
                        print(preview_text)
                        if more > 0:
                            print(f"... (+{more} more lines)")

                    confirm = input(f"[y] Write  [n] Skip: ")

                else:
                    # Other dangerous tools (apply_diff, etc.)
                    path = tool_call.params.get("path", tool_call.params.get("edits", ""))
                    confirm = input(f"\n[!] Execute {tool_call.name} on {path}? [y/n]: ")

                if confirm.lower() != "y":
                    # v2.1.0: Ask WHY rejected and send feedback to LLM
                    rejection_feedback = self._handle_rejection(tool_call)
                    results.append(rejection_feedback)
                    # Continue loop - LLM will see feedback and can fix
                    continue

            # Execute tool
            try:
                func = tool["func"]
                params = tool["params"]

                # Build kwargs from params
                kwargs = {}
                for param_name in params:
                    if param_name in tool_call.params:
                        value = tool_call.params[param_name]

                        # Parse JSON for complex params (tasks, edits, options)
                        if param_name in ("tasks", "edits", "options"):
                            if isinstance(value, str):
                                try:
                                    value = json.loads(value)
                                except json.JSONDecodeError:
                                    # Try comma-separated for options
                                    if param_name == "options":
                                        value = [v.strip() for v in value.split(",")]

                        # Convert boolean strings
                        elif value in ("true", "false", "True", "False"):
                            value = value.lower() == "true"

                        # Convert numeric strings
                        elif param_name in ("line_number", "start_line", "end_line", "limit"):
                            try:
                                value = int(value)
                            except ValueError:
                                pass

                        kwargs[param_name] = value

                # v0.8.0: Validate params and provide hints on error
                is_valid, error_msg = validate_tool_params(tool_call.name, kwargs)
                if not is_valid:
                    hint = get_tool_hint(tool_call.name, error_msg)
                    results.append(f"[x] {tool_call.name}: {hint}")
                    continue

                # Execute!
                result = func(**kwargs)

                # Handle different return types
                if isinstance(result, tuple):
                    # execute_command returns (executed, stdout, stderr, returncode)
                    if len(result) == 4:
                        executed, stdout, stderr, returncode = result
                        if executed:
                            output = stdout if stdout else stderr
                            results.append(f"[ok] {tool_call.name}:\n{output}" if output else f"[ok] {tool_call.name}: Done")
                            # v0.6.0: Save for memory extraction
                            tool_results_for_memory.append({
                                "name": tool_call.name,
                                "args": tool_call.params,
                                "result": output or ""
                            })
                        else:
                            results.append(f"[x] {tool_call.name}: {stderr}")
                            tool_results_for_memory.append({
                                "name": tool_call.name,
                                "args": tool_call.params,
                                "result": f"Error: {stderr}"
                            })
                    # Other tools return (success, message)
                    elif len(result) == 2:
                        success, message = result
                        if success:
                            results.append(f"[ok] {tool_call.name}: {message}")
                        else:
                            results.append(f"[x] {tool_call.name}: {message}")
                        # v0.6.0: Save for memory extraction
                        tool_results_for_memory.append({
                            "name": tool_call.name,
                            "args": tool_call.params,
                            "result": message if success else f"Error: {message}"
                        })
                    else:
                        results.append(f"[ok] {tool_call.name}: {result}")
                        tool_results_for_memory.append({
                            "name": tool_call.name,
                            "args": tool_call.params,
                            "result": str(result)
                        })
                else:
                    # read_file returns string directly
                    result_str = str(result)

                    # Check for error markers in result
                    error_markers = ["not found", "error", "failed", "command not found",
                                   "permission denied", "no such file", "does not exist"]
                    is_error = any(marker in result_str.lower() for marker in error_markers)

                    # v1.0.6: Add reminder for repeated read_file (content still returned!)
                    if tool_call.name == "read_file" and not is_error:
                        path = kwargs.get("path", "")
                        resolved = str(resolve_path(path))

                        if hasattr(self, '_read_files_this_task'):
                            if resolved in self._read_files_this_task:
                                result_str += "\n\n[!] Repeated read. To modify: use write_file or SEARCH/REPLACE"
                            self._read_files_this_task.add(resolved)

                    if is_error:
                        # Mark as failed so LLM doesn't hallucinate success!
                        results.append(f"[x] {tool_call.name}: {result_str}\n[!] THIS TOOL CALL FAILED - do NOT claim success!")
                    elif len(result_str) > 2000:
                        results.append(f"[ok] {tool_call.name}: {result_str[:2000]}...")
                    else:
                        results.append(f"[ok] {tool_call.name}: {result_str}")

                    # v0.6.0: Save for memory extraction
                    tool_results_for_memory.append({
                        "name": tool_call.name,
                        "args": tool_call.params,
                        "result": result_str if not is_error else f"Error: {result_str}"
                    })

            except Exception as e:
                error_msg = str(e)
                results.append(f"[x] Error in {tool_call.name}: {error_msg}")
                # Save errors for memory extraction
                tool_results_for_memory.append({
                    "name": tool_call.name,
                    "args": tool_call.params,
                    "result": f"Error: {error_msg}"
                })

        # v0.7.0: Extract facts from tool results (POINTER/VALUE)
        try:
            mm = MemoryManager()
            extracted = mm.extract_from_tool_results(tool_results_for_memory)
            # Optionally log what was saved (for debugging)
            # if extracted:
            #     results.append(f"📝 Memory: {', '.join(extracted)}")
        except Exception:
            pass  # Don't fail if memory extraction fails

        # v0.7.0: Save tool results to ChatStorage for grep retrieval
        if self.chat_storage:
            try:
                for tr in tool_results_for_memory:
                    self.chat_storage.append_tool_call(
                        tool_name=tr["name"],
                        args=tr["args"],
                        result=tr["result"]
                    )
            except Exception:
                pass  # ChatStorage is optional

        # v2.0.0: Track files and commands in ProjectContext
        try:
            last_action = ""
            last_result = ""
            last_file = None

            for tr in tool_results_for_memory:
                name = tr["name"]
                args = tr["args"]
                result = tr["result"]
                is_error = result.startswith("Error") or "[x]" in result

                if name == "write_file":
                    path = args.get("path", "")
                    content = args.get("content", "")
                    # v2.4.0: Universal preview via ContentPreview gearbox
                    pr = self.preview.generate(content)
                    self.project_context.files.track_write(path, lines=pr.total_lines, summary=pr.preview)
                    # v2.6.0: Invalidate repo_map cache (new file changes structure)
                    self.project_context.invalidate_repo_map()
                    # v2.1.0: Update current vector
                    last_action = f"write_file: {path}"
                    last_result = "ERROR" if is_error else "SUCCESS"
                    last_file = path

                elif name == "read_file":
                    path = args.get("path", "")
                    if not is_error:
                        # v2.4.0: Universal preview via ContentPreview gearbox
                        pr = self.preview.generate(result)
                        self.project_context.files.track_read(path, summary=pr.preview)
                    last_action = f"read_file: {path}"
                    last_result = "ERROR" if is_error else "SUCCESS"

                elif name == "execute_command":
                    cmd = args.get("cmd", args.get("command", ""))
                    exit_code = 1 if is_error else 0
                    error_text = result if is_error else ""

                    # v2.4.0: Universal preview with file saving for long output
                    output_preview = ""
                    output_path = ""
                    if not is_error and result:
                        pr = self.preview.generate(result, save_full=True)
                        output_preview = pr.preview
                        output_path = pr.full_path

                    self.project_context.terminal.track(
                        cmd, exit_code, error_text[:500],
                        output=output_preview, output_path=output_path
                    )
                    # v2.1.0: Update current vector
                    last_action = f"execute_command: {cmd[:50]}"
                    last_result = f"ERROR: {error_text[:100]}" if is_error else "SUCCESS"

                else:
                    # Other tools
                    last_action = f"{name}"
                    last_result = "ERROR" if is_error else "SUCCESS"

            # v2.5.0: Get pending task from TodoStateMachine for context
            pending_task = ""
            for task in self.project_context.todo.tasks:
                if task.status in ('pending', 'in_progress'):
                    pending_task = task.text
                    break

            # v2.1.0: Update current vector (enables follow-up like "can you run it?")
            if last_action:
                self.project_context.update_current(
                    action=last_action,
                    result=last_result,
                    file=last_file,
                    pending_task=pending_task
                )
            else:
                # Save project context even without action
                self.project_context._save()

        except Exception as e:
            # Debug mode: show tracking errors
            if self.debug:
                print(f"[DEBUG] Tracking error: {e}")

        return results

    def run_interactive(self, debug: bool = False) -> int:
        """
        Run interactive CLI loop.

        Args:
            debug: Show debug output (iterations, parsing info)

        Returns:
            Exit code
        """
        from pocketcoder.ui.cli import run_cli
        return run_cli(self, debug=debug)
