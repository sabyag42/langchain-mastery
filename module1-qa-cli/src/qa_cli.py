import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# Always finds .env relative to this file's location
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)


def build_chain():
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful expert in {topic}. "
                   "Answer clearly and concisely in 3-5 sentences."),
        ("human", "{question}")
    ])

    output_parser = StrOutputParser()

    chain = prompt | llm | output_parser

    return chain


def main():
    print("=== AI Q&A CLI Tool ===")
    print("Type 'exit' to quit\n")

    chain = build_chain()

    topic = input("What topic should I be an expert in? "
                  "(e.g. Python, AWS, Playwright): ").strip()

    while True:
        question = input(f"\nYour question about {topic}: ").strip()

        if question.lower() == "exit":
            print("Bye!")
            break

        if not question:
            continue

        print("\nThinking...\n")

        response = chain.invoke({
            "topic": topic,
            "question": question
        })

        print(f"Answer: {response}")
        print("-" * 50)


if __name__ == "__main__":
    main()

# *****************Explanation of the .invoke method*****************
"""
**`chain.invoke({"topic": topic, "context": context, "question": question})`**

This single line fires the entire pipeline. Let's trace exactly what happens under the hood when Python executes this:
```
Step 1 — LangChain calls prompt.invoke({"topic": ..., "context": ..., "question": ...})
         Pydantic validates that both keys exist and are strings
         Returns: [SystemMessage(content="..."), HumanMessage(content="...")]

Step 2 — LangChain calls llm.invoke([SystemMessage(...), HumanMessage(...)])
         langchain_openai serializes the Pydantic messages to JSON
         Makes an HTTPS POST to api.openai.com/v1/chat/completions
         OpenAI processes and returns a response JSON
         langchain_openai deserializes it back into an AIMessage Pydantic object
         Returns: AIMessage(content="A locator is...", response_metadata={...})

Step 3 — LangChain calls output_parser.invoke(AIMessage(...))
         StrOutputParser reads .content from the AIMessage object
         Returns: "A locator is..." (plain Python string)

Step 4 — .invoke() returns that plain string to your variable `response`

"""
"""
🎯 Interview Angles — what you can now confidently talk about
"What is LCEL and why does LangChain use it?"
LCEL is LangChain Expression Language — it uses the | pipe operator to compose Runnable objects into a RunnableSequence. Each component implements the Runnable interface which guarantees .invoke(), .stream(), and .ainvoke() methods. This means any chain you build with LCEL automatically supports streaming and async without changing your pipeline definition.
"How does LangChain communicate with OpenAI?"
langchain_openai's ChatOpenAI class is a wrapper around the openai Python SDK. When .invoke() is called, it serializes LangChain's Pydantic message objects (SystemMessage, HumanMessage) into the JSON format OpenAI's API expects, makes an HTTPS POST to /v1/chat/completions, receives the response JSON, and deserializes it back into an AIMessage Pydantic object.
"Why use Pydantic in LangChain?"
LangChain uses Pydantic for three things — runtime validation of inputs and outputs, serialization to JSON for tracing and logging (LangSmith), and type safety across the entire chain pipeline. Every message, every prompt, and every chain component is a Pydantic model underneath.
"What is the difference between .invoke(), .stream(), and .ainvoke()?"
.invoke() — synchronous, waits for the full response before returning. .stream() — synchronous but returns a generator that yields tokens as they arrive (better UX for long responses). .ainvoke() — async version of invoke, used with asyncio when you need to run multiple chains concurrently without blocking. All three are part of the Runnable interface that every LCEL component implements.

"""
