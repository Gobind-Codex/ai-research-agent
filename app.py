import tempfile
from pathlib import Path

import streamlit as st

from src.pdf_loader import extract_pages
from src.chunking import chunk_pages
from src.vector_store import index_chunks
from src.research_agent import answer_question
from src.verification import validate_citations

st.set_page_config(
    page_title="AI Research Agent",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 AI Research Agent")
st.caption("Explore your research papers with local AI and evidence-based answers.")

# Keep conversation and indexing information across reruns.
if "messages" not in st.session_state:
    st.session_state.messages = []

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = {}


with st.sidebar:
    st.header("📚 Document workspace")

    uploaded_files = st.file_uploader(
        "Upload research PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Start with small, text-based PDFs.",
    )

    if st.button("Index uploaded PDFs", use_container_width=True):
        if not uploaded_files:
            st.warning("Upload at least one PDF first.")
        else:
            for uploaded_file in uploaded_files:
                # Keep the uploaded filename, but use a safe temporary path.
                with tempfile.TemporaryDirectory() as temp_dir:
                    pdf_path = Path(temp_dir) / uploaded_file.name
                    pdf_path.write_bytes(uploaded_file.getvalue())

                    try:
                        with st.spinner(f"Processing {uploaded_file.name}..."):
                            pages = extract_pages(pdf_path)
                            chunks = chunk_pages(pages)

                            if not chunks:
                                st.warning(
                                    f"No selectable text found in "
                                    f"{uploaded_file.name}."
                                )
                                continue

                            count = index_chunks(chunks)

                            st.session_state.indexed_files[uploaded_file.name] = {
                                "pages": len(pages),
                                "chunks": count,
                            }

                        st.success(
                            f"{uploaded_file.name}: "
                            f"{len(pages)} pages, {count} chunks indexed."
                        )

                    except Exception as error:
                        st.error(f"Could not process {uploaded_file.name}: {error}")

    st.divider()
    st.subheader("Indexed this session")

    if st.session_state.indexed_files:
        for filename, info in st.session_state.indexed_files.items():
            st.write(f"**{filename}**")
            st.caption(f"{info['pages']} pages · {info['chunks']} chunks")
    else:
        st.caption("Upload and index a PDF to begin.")

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# Main chat area
st.subheader("💬 Ask your research question")

# Display previous messages.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("View retrieved sources"):
                for source in message["sources"]:
                    st.markdown(
                        f"**[{source['id']}] {source['source']}** — "
                        f"Page {source['page_number']}"
                    )
                    st.caption(
                        source["text"][:1000]
                        + ("..." if len(source["text"]) > 1000 else "")
                    )

            if message.get("validation") is not None:
                if not message["validation"]["has_citations"]:
                    st.warning("This answer contains no source citations.")
                elif not message["validation"]["valid"]:
                    st.warning(
                        "Invalid source IDs: "
                        + ", ".join(message["validation"]["invalid_ids"])
                    )


question = st.chat_input("Ask something about your research papers...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching documents and generating an answer..."):
                result = answer_question(question, top_k=4)
                answer = result["answer"]
                sources = result["sources"]
                validation = validate_citations(answer, sources)

            st.markdown(answer)

            if sources:
                with st.expander("View retrieved sources", expanded=True):
                    for source in sources:
                        st.markdown(
                            f"**[{source['id']}] {source['source']}** — "
                            f"Page {source['page_number']}"
                        )
                        st.caption(source["text"][:1000])

            if not validation["has_citations"]:
                st.warning("This answer contains no source citations.")
            elif not validation["valid"]:
                st.warning(
                    "Invalid source IDs: " + ", ".join(validation["invalid_ids"])
                )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "validation": validation,
                }
            )

        except Exception as error:
            error_message = f"Request failed: {error}"
            st.error(error_message)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": error_message,
                    "sources": [],
                }
            )


# Export the conversation as a text research report.
if st.session_state.messages:
    report_lines = ["AI Research Agent — Research Notes", ""]

    for message in st.session_state.messages:
        role = message["role"].upper()
        report_lines.extend(
            [
                f"{role}:",
                message["content"],
                "",
            ]
        )

        if message.get("sources"):
            report_lines.append("Sources:")
            for source in message["sources"]:
                report_lines.append(
                    f"- [{source['id']}] {source['source']}, "
                    f"page {source['page_number']}"
                )
            report_lines.append("")

    st.download_button(
        "Download research notes",
        data="\n".join(report_lines),
        file_name="research_notes.txt",
        mime="text/plain",
    )
