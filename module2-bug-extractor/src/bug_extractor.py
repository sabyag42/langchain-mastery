import os
from pathlib import Path
from typing import Literal

from langchain_core.prompts import ChatMessagePromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


class BugReport(BaseModel):
    """Structured Bug Report from raw text"""

    title: str = Field(description="A clear, concise title for the bug in under 10 words")

    severity: Literal["low", "medium", "high", "critical"] = Field(
        description="Severity level based on user impact. critical=blocks all users, "
                    "high=major feature broken, medium=partial impact, low=minor issue"
    )

    browser: str = Field(
        description="Browser where the bug occurs. Use 'unknown' if not mentioned"
    )

    os: str = Field(
        description="Operating system where bug occurs. Use 'unknown' if not mentioned"
    )

    steps_to_reproduce: str = Field(
        description="Clear steps to reproduce the bug based on the description"
    )

    expected_result: str = Field(
        description="What should have happened according to normal behaviour"
    )

    actual_result: str = Field(
        description="What actually happened — the bug behaviour"
    )


def build_chain():
    """
    Builds and returns the bug extraction chain.
    
    Flow:
      raw bug text → prompt + format instructions → LLM → PydanticOutputParser → BugReport object
    """

     # --- THE PARSER ---
    # We create the parser FIRST this time — before the prompt.
    # Why? Because we need to call parser.get_format_instructions()
    # and inject those instructions INTO the prompt template below.
    # This is the key difference from Module 1 where parser came last.

    parser = PydanticOutputParser(pydantic_object=BugReport)

    # --- THE LLM ---
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0,openai_api_key=os.getenv("OPENAI_API_KEY"))

    # --- THE PROMPT ---
    # Notice the {format_instructions} placeholder.
    # This is where the full JSON schema gets injected at runtime.
    # parser.get_format_instructions() generates something like:
    #
    #   "The output should be formatted as a JSON instance that conforms
    #    to the JSON schema below:
    #    {
    #      'title': {'description': 'A clear concise title...', 'type': 'string'},
    #      'severity': {'description': 'Severity level...', 'enum': [...]}
    #      ...
    #    }
    #    Return ONLY the JSON. No explanation. No markdown."
    #
    # You never write this manually — the parser generates it from
    # your BugReport Pydantic model automatically. This is the magic.
    prompt = ChatMessagePromptTemplate.from_messages([("system","You are an expert QA engineer and bug triage specialist. "
         "Extract structured bug report information from the raw text provided.\n\n"
         "IMPORTANT: {format_instructions}"),("human","Here is the raw bug report text:\n\n{bug_text}")])
    

     # --- THE CHAIN ---
    # Same LCEL pipe as Module 1, but notice the difference:
    #
    # Module 1:  prompt | llm | StrOutputParser()
    #                              ↑ extracts plain string
    #
    # Module 2:  prompt | llm | parser
    #                              ↑ validates JSON + returns BugReport object
    #
    # The pipe is identical. Only the parser changes.
    # This is the power of LCEL — swap any component, rest stays the same.

    chain = prompt | llm | parser

    return chain,parser

"""
Let me break down everything new here.

---

**Why we create `parser` BEFORE `prompt`**

In Module 1 the order was: LLM → Prompt → Parser. We could define them in any order because `StrOutputParser` had no dependencies.

`PydanticOutputParser` is different. It has something the prompt needs — `format_instructions`. The prompt template has a `{format_instructions}` placeholder that must be filled with the JSON schema the parser generates. So the parser must exist before we can build the prompt.

This is the dependency:

parser.get_format_instructions()
            ↓
    injected into prompt as
    {format_instructions}
            ↓
      sent to the LLM
            ↓
    LLM returns JSON matching the schema
            ↓
  parser validates + returns BugReport

"""






