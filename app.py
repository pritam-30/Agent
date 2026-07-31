import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

from tools import rag_tool, note_tool, TOOLS


load_dotenv()

# =====================================================
# Gemini Client
# =====================================================

gen_client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


# =====================================================
# Planner Prompt
# =====================================================

PLANNER_PROMPT = """
You are an intelligent AI assistant with access to external tools.

Your job is to decide whether a tool is required before answering.

Rules:

1. If the user's request can be answered using your own knowledge, answer directly without calling any tool.

2. If the user's request depends on information contained in the user's uploaded or indexed documents
(such as resumes, interview guides, PDFs, notes, job descriptions, or other files),
call retrieve_documents.

3. If the user asks to save, remember, record, or create a note, reminder,
todo, or event, call save_note.

4. You may call multiple tools if necessary.

5. After receiving tool results, determine whether another tool is needed.

6. Continue until you have enough information to produce the final answer.

7. Never invent document contents. If document information is required,
always use retrieve_documents.

8. Never call tools unnecessarily.

9. When a tool has provided sufficient information,
use it to answer the user instead of calling the same tool again.
"""


# =====================================================
# Planner
# =====================================================

def planner(contents):
    """
    Send the conversation to Gemini and return the response.
    """

    return gen_client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=PLANNER_PROMPT,
            tools=[
                rag_tool,
                note_tool,
            ],
        ),
    )


# =====================================================
# Agent Loop
# =====================================================

MAX_ITERATIONS = 10


def run_agent(user_message: str):

    contents = [
        {
            "role": "user",
            "parts": [
                {
                    "text": user_message
                }
            ],
        }
    ]

    for iteration in range(MAX_ITERATIONS):

        print(f"\n========== Iteration {iteration + 1} ==========")

        response = planner(contents)

        response_content = response.candidates[0].content
        parts = response_content.parts

        # -------------------------------------------------
        # Look for function calls anywhere in the response.
        # -------------------------------------------------

        function_calls = [
            part.function_call for part in parts if part.function_call]

        # -------------------------------------------------
        # Final Answer
        # -------------------------------------------------

        if not function_calls:

            print("\nGemini Final Response:\n")
            return response.text

        response_parts = []

        for fc in function_calls:
            tool_name = fc.name
            args = dict(fc.args)
            print(f"\nCalling Tool : {tool_name}")
            print(f"Arguments    : {args}")

            tool = TOOLS.get(tool_name)
            if tool is None:
                raise ValueError(f"Unknown tool: {tool_name}")

            result = tool(**args)
            print("\nTool Result:")
            print(result)

            response_parts.append(
                types.Part.from_function_response(
                    name=tool_name, response={"result": result})
            )

        # -------------------------------------------------
        # Append Gemini's function call
        # -------------------------------------------------

        contents.append(response_content)

        # -------------------------------------------------
        # Append Tool Response
        # -------------------------------------------------

        contents.append(
            types.Content(
                role="user",
                parts=response_parts
            )
        )

        # -------------------------------------------------
        # Debug Conversation State
        # -------------------------------------------------

        print("\nConversation State:\n")

        for content in contents:
            print(content)
            print()

    return "Maximum number of iterations reached."


# =====================================================
# Entry Point
# =====================================================

if __name__ == "__main__":

    while True:

        user_input = input("\nYou: ")

        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        answer = run_agent(user_input)

        print(f"\nAssistant: {answer}")
