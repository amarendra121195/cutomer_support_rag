"""Web UI for the support assistant.

    streamlit run streamlit_app.py
"""
import streamlit as st

from src.pipeline import IndexNotBuiltError, RAGPipeline
from src.utils.helpers import APIError, load_config, setup_logging

EXAMPLE_QUESTIONS = [
    "How do I connect a domain I already own?",
    "Can I accept payments while my Wix Payments account is still being verified?",
    "How do I add a contact form to my site?",
]
SOURCE_FILTERS = {
    "Everything": None,
    "Wix help center": ["wixqa"],
    "Bitext support templates": ["bitext"],
}

st.set_page_config(page_title="Support Assistant", page_icon="💬", layout="centered")


@st.cache_resource(show_spinner="Opening the knowledge base...")
def get_pipeline() -> RAGPipeline:
    """One pipeline for the whole server, so the index and the API clients are opened once."""
    config = load_config()
    setup_logging(config)
    return RAGPipeline(config)


def show_answer(message: dict) -> None:
    """Render an assistant message: the answer, what it cost, and the sources it was given."""
    # Streamlit reads $...$ as LaTeX, which would garble answers that mention prices.
    st.markdown(message["content"].replace("$", "\\$"))
    if message.get("usage"):
        st.caption(message["usage"])
    if not message.get("sources"):
        return
    with st.expander(f"Sources ({len(message['sources'])})"):
        for number, source in enumerate(message["sources"], start=1):
            title = source["title"].replace("[", "\\[").replace("]", "\\]")
            heading = f"[{title}]({source['url']})" if source["url"] else title
            st.markdown(f"**\\[{number}\\]** {heading}")
            st.caption(f"{source['source']} · similarity {source['score']:.2f}")
            st.text(source["text"])     # plain text: knowledge-base content is shown, never interpreted


def answer(pipeline: RAGPipeline, question: str, top_k: int, sources: list[str] | None) -> dict:
    """Run the pipeline and turn the result into a message that can be kept in the session."""
    result = pipeline.answer(question, top_k=top_k, sources=sources)
    response = result.response
    usage = f"{response.model} · {response.input_tokens:,} input / {response.output_tokens:,} output tokens"
    if response.cost_usd is not None:
        usage += f" · ${response.cost_usd:.4f}"
    return {
        "role": "assistant",
        "content": result.answer,
        "usage": usage,
        "sources": [
            {
                "title": chunk.metadata["title"],
                "url": chunk.metadata.get("url"),
                "source": chunk.metadata["source"],
                "score": chunk.score,
                "text": chunk.text,
            }
            for chunk in result.sources
        ],
    }


pipeline = get_pipeline()
config = pipeline.config
st.session_state.setdefault("messages", [])

with st.sidebar:
    st.header("Settings")
    source_filter = st.radio("Search in", list(SOURCE_FILTERS))
    top_k = st.slider("Sources per answer", min_value=1, max_value=10, value=config["retrieval"]["top_k"])
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()
    st.divider()
    st.caption(
        f"Chat model: `{config['llm']['model']}`  \n"
        f"Embeddings: `{config['embeddings']['model']}`  \n"
        f"Indexed chunks: {pipeline.store.count():,}"
    )

st.title("Support Assistant")
st.caption(
    "Answers are written from the retrieved help-center articles and cite them as [1], [2]. "
    "Each question is answered on its own; earlier messages are not remembered."
)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            show_answer(message)
        else:
            st.markdown(message["content"].replace("$", "\\$"))

question = st.chat_input("Ask a support question")
examples = st.empty()
if not st.session_state.messages:
    with examples.container():
        st.write("Try one of these:")
        for example in EXAMPLE_QUESTIONS:
            if st.button(example):
                question = example

if question:
    examples.empty()    # the suggestions make way for the conversation
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question.replace("$", "\\$"))
    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching the knowledge base and writing an answer..."):
                reply = answer(pipeline, question, top_k, SOURCE_FILTERS[source_filter])
        except (APIError, IndexNotBuiltError) as error:
            st.error(str(error))
            st.stop()
        show_answer(reply)
    st.session_state.messages.append(reply)
