"""
Web UI for PocketCoder using Gradio.

Provides a browser-based interface as alternative to CLI.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pocketcoder.core.coder import Coder


def start_web(coder: "Coder", port: int = 7860, share: bool = False):
    """
    Start Gradio web interface.

    Args:
        coder: Coder instance
        port: Port number
        share: Create public URL
    """
    try:
        import gradio as gr
    except ImportError:
        print("Gradio not installed. Install with: pip install pocketcoder[web]")
        return

    def handle_message(message: str, history: list):
        """Handle user message."""
        if not message.strip():
            return history, ""

        # Add user message to history
        history = history + [(message, None)]

        # Send to LLM
        parsed = coder.send_message(message)

        # Build response
        response_parts = []

        if parsed.warnings:
            for w in parsed.warnings:
                response_parts.append(f"⚠️ {w}")

        if parsed.thinking:
            response_parts.append(f"*{parsed.thinking}*")

        if parsed.edits:
            response_parts.append("**Proposed changes:**")
            for edit in parsed.edits:
                response_parts.append(f"- {edit.filename}")

        if parsed.is_question:
            response_parts.append(parsed.question_text)
        elif not parsed.edits:
            response_parts.append(parsed.raw)

        response = "\n\n".join(response_parts)
        history[-1] = (message, response)

        return history, ""

    def add_files(files):
        """Handle file upload."""
        if not files:
            return "No files selected"

        added = []
        for f in files:
            if coder.add_file(f.name):
                added.append(f.name.split("/")[-1])

        return f"Added: {', '.join(added)}" if added else "No files added"

    def get_file_list():
        """Get current file list."""
        if not coder.files:
            return []
        return [[p.name, ctx.lines] for p, ctx in coder.files.items()]

    # Build UI
    with gr.Blocks(title="PocketCoder", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🔧 PocketCoder")

        with gr.Row():
            # Left panel - files
            with gr.Column(scale=1):
                gr.Markdown("### Files")
                file_list = gr.Dataframe(
                    value=get_file_list,
                    headers=["File", "Lines"],
                    interactive=False,
                )
                file_upload = gr.File(
                    label="Add files",
                    file_count="multiple",
                )
                upload_status = gr.Textbox(label="Status", interactive=False)

            # Center - chat
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(label="Chat", height=500)
                msg_input = gr.Textbox(
                    label="Message",
                    placeholder="Describe what you want to do...",
                    lines=3,
                )
                with gr.Row():
                    send_btn = gr.Button("Send", variant="primary")
                    clear_btn = gr.Button("Clear")

            # Right panel - settings
            with gr.Column(scale=1):
                gr.Markdown("### Settings")
                model_input = gr.Textbox(
                    label="Model",
                    value=coder.model,
                )
                gr.Markdown(f"Provider: {coder.provider_name}")

        # Event handlers
        send_btn.click(
            handle_message,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, msg_input],
        )

        msg_input.submit(
            handle_message,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, msg_input],
        )

        clear_btn.click(
            lambda: ([], ""),
            outputs=[chatbot, msg_input],
        )

        file_upload.change(
            add_files,
            inputs=[file_upload],
            outputs=[upload_status],
        ).then(
            get_file_list,
            outputs=[file_list],
        )

    print(f"🌐 Starting web UI at http://localhost:{port}")
    app.launch(server_port=port, share=share, show_error=True)
