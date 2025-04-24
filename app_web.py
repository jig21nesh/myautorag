# app_web.py
import streamlit as st
import pkgutil, importlib, textwrap
from pathlib import Path

from search_space import SEARCH_SPACE, reload_search_space

st.set_page_config(page_title="AutoRAG Module Explorer", layout="centered")
st.title("🔍 AutoRAG Pipeline Module Explorer")
st.markdown(
    """
    Select any RAG node from the sidebar to see all available modules for that step.
    """
)

# — Optionally re-scan for new query-expanders at runtime
if st.sidebar.button("🔄 Refresh All Modules"):
    SEARCH_SPACE.clear()
    SEARCH_SPACE.update(reload_search_space())
    st.rerun()

# Sidebar: pick a node
st.sidebar.header("Pipeline Nodes")
node = st.sidebar.radio("Choose a node", list(SEARCH_SPACE.keys()))

# Main: list existing modules
st.subheader(f"Candidates for **{node}**")
for mod in SEARCH_SPACE[node]:
    label = getattr(mod, "name", None) or mod.__class__.__name__
    st.write(f"- **{label}**   (`{mod.__class__.__name__}`)")

# If we're looking at query_expansion, show a form to add a new one
if node == "query_expansion":
    st.markdown("---")
    st.header("➕ Add a new Query-Expander")
    expander_name = st.text_input(
        "Expander name (snake_case)",
        placeholder="my_custom",
    ).strip()
    st.markdown(
        "Paste only the **body** of your `__call__(self, query: str)->str` method below.\n\n"
        "  • It will be auto-indented for you.\n\n"
        "  • Example:\n\n"
        "```python\n"
        "# modify query however you like\n"
        "return query.upper()\n"
        "```"
    )
    code_body = st.text_area("Code body", height=150)
    if st.button("Create Expander"):
        if not expander_name.isidentifier():
            st.error("❌ Invalid Python identifier!")
        elif not code_body.strip():
            st.error("❌ Please paste at least one line of code.")
        else:
            # prepare file path
            target_dir = Path(__file__).parent / "modules" / "query_expansion"
            target_dir.mkdir(parents=True, exist_ok=True)
            class_name = "".join(word.title() for word in expander_name.split("_")) + "Expander"
            filename = f"{expander_name}_expander.py"
            target = target_dir / filename

            if target.exists():
                st.error(f"❌ `{filename}` already exists.")
            else:
                # indent user body by 8 spaces so it's inside the __call__ block
                indented = textwrap.indent(code_body.strip(), " " * 8)
                file_contents = f'''\
from .base import QueryExpander
from langchain.chat_models import AzureChatOpenAI
from config import settings

class {class_name}(QueryExpander):
    name = "{expander_name}"

    def __init__(self):
        self.llm = AzureChatOpenAI(
            azure_endpoint=settings.OPENAI_ENDPOINT,
            api_version=settings.OPENAI_API_VERSION,
            azure_deployment=settings.LLM_MODEL,
            api_key=settings.OPENAI_API_KEY,
            temperature=0.3,
            verbose=False,
        )

    def __call__(self, query: str) -> str:
{indented}
'''
                target.write_text(file_contents, encoding="utf-8")
                st.success(f"✅ Created `{filename}` – restart the app or click Refresh to load it!")