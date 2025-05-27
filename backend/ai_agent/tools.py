"""LangChain tools for the Legal AI agent."""

from __future__ import annotations

from typing import List

from langchain.tools import BaseTool
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOpenAI

from .vector_db import MilvusVectorDB


class KeywordGeneratorTool(BaseTool):
    name = "generate_keywords"
    description = "Generate semantic keywords for a legal query"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        prompt = PromptTemplate(
            template="Extract important legal keywords from the query: {query}",
            input_variables=["query"],
        )
        self.chain = LLMChain(llm=ChatOpenAI(temperature=0), prompt=prompt)

    def _run(self, query: str) -> List[str]:
        response = self.chain.run(query=query)
        keywords = [k.strip() for k in response.split(",") if k.strip()]
        return keywords

    async def _arun(self, query: str) -> List[str]:
        return self._run(query)


class ConstitutionSearchTool(BaseTool):
    name = "search_db_constitution"
    description = "Search the Constitution vector database"

    def __init__(self, vector_db: MilvusVectorDB, **kwargs):
        super().__init__(**kwargs)
        self.vector_db = vector_db

    def _run(self, query: str) -> List[dict]:
        return self.vector_db.search(query)

    async def _arun(self, query: str) -> List[dict]:
        return self._run(query)


class IPCSearchTool(BaseTool):
    name = "search_db_penal_code"
    description = "Search the IPC vector database"

    def __init__(self, vector_db: MilvusVectorDB, **kwargs):
        super().__init__(**kwargs)
        self.vector_db = vector_db

    def _run(self, query: str) -> List[dict]:
        return self.vector_db.search(query)

    async def _arun(self, query: str) -> List[dict]:
        return self._run(query)


class PredictPunishmentTool(BaseTool):
    name = "predict_punishment_from_case"
    description = (
        "Predict punishment based on case description. "
        "Returns likely sentence and relevant IPC sections."
    )

    def __init__(self, llm=None, **kwargs):
        super().__init__(**kwargs)
        self.llm = llm or ChatOpenAI(temperature=0)

    def _run(self, query: str) -> str:
        prompt = (
            "Given the case description, predict likely punishment and relevant IPC sections. "
            "Return concise text. Query: " + query
        )
        return self.llm.invoke(prompt)

    async def _arun(self, query: str) -> str:
        return self._run(query)

