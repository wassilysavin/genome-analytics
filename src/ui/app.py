import logging
from typing import Any, Optional

import streamlit as st

from src.ui.logic import AppLogic
from src.constants.constants import *

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s", force=True)
logger = logging.getLogger(__name__)


def main() -> None:
    app_logic = AppLogic()

    st.set_page_config(page_title="genome-insights", page_icon="🧬", layout="wide")

    st.title("🧬 genome-insights")

    _initialize_session_state(app_logic)

    _render_sequence_input()
    _render_analysis_results()
    _render_chat_interface()


def _initialize_session_state(app_logic: AppLogic) -> None:
    if "app_logic" not in st.session_state:
        st.session_state.app_logic = app_logic

    if "analysis_complete" not in st.session_state:
        st.session_state.analysis_complete = False
    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = None
    if "gene_names" not in st.session_state:
        st.session_state.gene_names = []
    if "workflow_results" not in st.session_state:
        st.session_state.workflow_results = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


def _render_sequence_input() -> None:
    # st.markdown("## Genome Analysis")
    st.markdown("Upload your DNA sequence to identify genes.")

    uploaded_file = st.file_uploader(
        "Upload DNA sequence file",
        type=["fasta", "fa", "txt"],
        help="Upload a FASTA file or paste DNA sequence directly",
    )

    sequence_text = st.text_area(
        "Or paste DNA sequence here", height=UI_TEXTAREA_HEIGHT, placeholder="ATCGATCGATCG..."
    )

    sequence = _get_sequence_from_input(uploaded_file, sequence_text)

    if sequence:
        _display_sequence_info(sequence)
        _render_analyze_button(sequence)


def _get_sequence_from_input(uploaded_file: Optional[Any], sequence_text: str) -> Optional[str]:
    sequence: Optional[str] = None

    if uploaded_file:
        sequence = uploaded_file.read().decode("utf-8")
        if sequence.startswith(">"):
            lines = sequence.split("\n")
            sequence = "".join(lines[FASTA_HEADER_SKIP_LINES:])
    elif sequence_text:
        sequence = sequence_text.strip()

    if sequence:
        sequence = "".join(c for c in sequence.upper() if c in VALID_DNA_CHARS_UI)

    return sequence


def _display_sequence_info(sequence: str) -> bool:
    if len(sequence) < MIN_SEQUENCE_LENGTH_BP:
        st.warning(f"Sequence too short. Please provide at least {MIN_SEQUENCE_LENGTH_BP} base pairs.")
        return False

    st.info(f"Sequence length: {len(sequence):,} base pairs")
    return True


def _render_analyze_button(sequence: str) -> None:
    if st.button("Analyze", type="primary"):
        _run_analysis(sequence)


def _run_analysis(sequence: str) -> None:
    app_logic = st.session_state.app_logic

    with st.spinner("Analyzing your sequence..."):
        try:
            analysis_results = app_logic.analyze_sequence(sequence)

            if analysis_results.success:
                st.success("Gene analysis completed.")

                gene_names = app_logic.extract_gene_names_from_results(analysis_results.results)

                if gene_names:
                    summary = app_logic.generate_summary(
                        sequence, gene_names, analysis_results.results, {"success": False}
                    )

                    workflow_results = {
                        "success": True,
                        "analysis_results": analysis_results,
                        "summary": summary,
                        "errors": [],
                    }

                    st.session_state.analysis_complete = True
                    st.session_state.analysis_results = analysis_results
                    st.session_state.gene_names = gene_names
                    st.session_state.workflow_results = workflow_results

                    st.rerun()
                else:
                    st.warning("No genes identified in the sequence")
            else:
                st.error(f"Analysis failed: {analysis_results.error or UNKNOWN_ERROR}")

        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")


def _render_analysis_results() -> None:
    if not st.session_state.analysis_complete or not st.session_state.workflow_results:
        return

    results = st.session_state.workflow_results

    if (
        results["analysis_results"]
        and results["analysis_results"].success
        and st.session_state.gene_names
    ):
        st.success(f"Found {len(st.session_state.gene_names)} genes.")
        gene_display = ", ".join(st.session_state.gene_names[:MAX_GENES_INLINE_DISPLAY])
        if len(st.session_state.gene_names) > MAX_GENES_INLINE_DISPLAY:
            gene_display += f" and {len(st.session_state.gene_names) - MAX_GENES_INLINE_DISPLAY} more"
        st.info(f"**Genes found:** {gene_display}")

    if results.get("summary"):
        st.markdown("### Summary & Insights")
        st.markdown(results["summary"])

    if st.button("Clear Results", type="secondary"):
        _clear_analysis_results()


def _clear_analysis_results() -> None:
    st.session_state.analysis_complete = False
    st.session_state.analysis_results = None
    st.session_state.gene_names = []
    st.session_state.workflow_results = None
    st.rerun()


def _render_chat_interface() -> None:
    if not st.session_state.gene_names:
        return

    st.markdown("### Chat")
    st.write("Ask anything about genomics, DNA, proteins, or biology in general!")

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input("Ask me anything about genomics, genes, or biology..."):
        _handle_chat_input(prompt)


def _handle_chat_input(prompt: str) -> None:
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"), st.spinner("Thinking..."):
        try:
            app_logic = st.session_state.app_logic
            response = app_logic.get_chat_response(prompt, st.session_state.gene_names)

            st.session_state.chat_history.append({"role": "assistant", "content": response})

            st.write(response)

        except Exception as e:
            error_msg = f"Chat failed: {str(e)}"
            st.error(error_msg)
            st.session_state.chat_history.append({"role": "assistant", "content": error_msg})


if __name__ == "__main__":
    main()
