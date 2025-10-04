import logging
from typing import Any

from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.tools import tool

from ..prompts.prompt_manager import PromptManager
from ..tools.bio.sequence_chunker import chunk_large_sequence, process_chunked_sequence
from ..tools.gene.gene_identification import (
    analyze_dna_sequence,
    generate_detailed_insights,
    generate_gene_summary,
    identify_genes_complete,
    search_databases_for_genes,
)
from .llm_factory import create_llm

logger = logging.getLogger(__name__)


class GeneAgent:
    def __init__(self):
        self.llm = create_llm()
        self.tools = self._setup_tools()
        self.agent_executor = self._create_agent()

    def _setup_tools(self) -> list[Any]:
        return [
            analyze_dna_sequence,
            search_databases_for_genes,
            generate_gene_summary,
            generate_detailed_insights,
            identify_genes_complete,
            chunk_large_sequence,
            process_chunked_sequence,
        ]

    def _create_agent(self) -> AgentExecutor:
        prompt_manager = PromptManager()
        prompt_template = PromptTemplate.from_template(prompt_manager.format_gene_agent_prompt())
        agent = create_react_agent(llm=self.llm, tools=self.tools, prompt=prompt_template)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            max_iterations=10,
            handle_parsing_errors=True,
            return_intermediate_steps=True,
            early_stopping_method="generate",
        )
        return agent_executor

    def identify_genes(self, sequence: str) -> dict[str, Any]:
        try:
            if len(sequence) > 10000:
                try:
                    chunk_size = 2000
                    direct_result = process_chunked_sequence.invoke(
                        {"sequence": sequence, "chunk_size": chunk_size}
                    )
                    return {
                        "success": True,
                        "sequence_length": len(sequence),
                        "tool_result": direct_result,
                        "tools_used": ["process_chunked_sequence"],
                        "method": "Direct invocation to avoid prompt truncation for large inputs",
                    }
                except Exception as tool_err:
                    logger.error(f"Direct process_chunked_sequence invocation failed: {tool_err}")
                    return {"success": False, "error": str(tool_err)}

            else:
                input_text = self.prompt_manager.format_gene_agent_complete_analysis_prompt(sequence)

            result = self.agent_executor.invoke({"input": input_text})

            return {
                "success": True,
                "sequence_length": len(sequence),
                "agent_output": result["output"],
                "intermediate_steps": result.get("intermediate_steps", []),
                "tools_used": self._extract_tools_used(result.get("intermediate_steps", [])),
                "method": "LangChain ReAct Agent with modular bioinformatics tools",
            }

        except Exception as e:
            logger.error(f"Gene Agent analysis failed: {e}")
            return {"success": False, "error": str(e)}

    def _extract_tools_used(self, intermediate_steps: list) -> list[str]:
        tools_used = []
        try:
            for step in intermediate_steps:
                if len(step) >= 1 and hasattr(step[0], "tool"):
                    tools_used.append(step[0].tool)
        except Exception:
            pass
        return tools_used

    def ask_question(self, question: str) -> dict[str, Any]:
        try:
            result = self.agent_executor.invoke({"input": question})

            return {
                "success": True,
                "question": question,
                "answer": result["output"],
                "tools_used": self._extract_tools_used(result.get("intermediate_steps", [])),
                "method": "LangChain ReAct Agent Q&A",
            }

        except Exception as e:
            logger.error(f"Gene Agent question failed: {e}")
            return {"success": False, "error": str(e), "question": question}
