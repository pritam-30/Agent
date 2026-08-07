import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from Eval import evaluate_rag
from tools import rag_tool, note_tool
from retrieval.retrieve import TOOLS
from utils import timer, show_notes
load_dotenv()

# =====================================================
# Gemini Client
# =====================================================

gen_client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

ENABLE_EVALUATION = os.getenv("ENABLE_EVALUATION", "False").lower() == "true"

# =====================================================
# Planner Prompt
# =====================================================

PLANNER_PROMPT = """
You are an intelligent AI assistant with access to external tools.

Your job is to determine whether external tools are needed to answer the user's request and to use them efficiently.

Rules:

0. If the user's request can be answered accurately using your own knowledge and the existing conversation context, answer directly without calling any tool.

1. If the user's request depends on information contained in the user's uploaded or indexed documents (such as PDFs, notes, resumes, interview guides, job descriptions, or other files), call retrieve_documents.

2. If the user asks to save, remember, record, create, or update a note, reminder, todo, or event, call save_note.

3. If a tool requires information produced by another tool, wait until that tool has completed before calling it. Otherwise, if multiple tools are independent, return all required function calls in a single response.

4. If multiple tools are required and they are independent, return ALL required function calls in the SAME response.

5. Only defer a tool call to a later planning step if it depends on the output of a previous tool.

6. After receiving tool results, determine whether additional tools are genuinely required. If not, produce the final answer immediately.

7. Never invent document contents. If document information is required, always use retrieve_documents.

8. Never call tools unnecessarily.

9. Never call the same tool more than once with the same arguments unless new information or a changed user request makes another call necessary.

10. When a tool has already provided sufficient information, use that information to answer the user instead of calling the same tool again.

11. Use the existing conversation history. If a previous tool response already contains the required information, reuse it instead of retrieving it again.

12. Treat each new user message as a fresh request unless it clearly refers to previous conversation or previously retrieved information.

13. When all required information has been gathered, provide a complete and helpful final answer.

14. When calling retrieve_documents, preserve the user's intent, technical terms, names, acronyms, and important keywords. Do not over-rewrite or over-expand the search query unless necessary to resolve ambiguity.

15. If retrieve_documents has been used, treat the retrieved document content as the primary source of truth. Base the answer on the retrieved content. If the retrieved information is insufficient, explicitly state that the documents do not contain enough information instead of filling gaps with your own knowledge.

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

contents = [
    {
        "role": "model",
        "parts": [
            {
                "text": "Hello! I am your intelligent AI assistant. How can I help you today?"
            }
        ],
    }
]


def run_agent(user_message: str):

    retrieved_chunks = None

    with timer("Total latency"):
        contents.append(
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=user_message)
                ]
            )
        )

        for iteration in range(MAX_ITERATIONS):

            print(f"\n========== Iteration {iteration + 1} ==========")

            with timer("Planner"):
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
                contents.append(response_content)
                final_answer = "".join(
                    part.text
                    for part in parts
                    if getattr(part, "text", None)
                ).strip()

                if not final_answer:
                    finish_reason = response.candidates[0].finish_reason
                    raise RuntimeError(
                        f"Gemini returned no usable text (finish_reason: {finish_reason})")

                try:
                    if ENABLE_EVALUATION and retrieved_chunks is not None:
                        evaluate_rag(
                            query=user_message,
                            answer=final_answer,
                            retrieved_chunks=retrieved_chunks,
                        )
                except Exception as e:
                    print(f"Evaluation failed: {e}")

                return final_answer

            response_parts = []

            for fc in function_calls:
                tool_name = fc.name
                args = dict(fc.args)
                print(f"\nCalling Tool : {tool_name}")
                print(f"Arguments    : {args}")

                tool = TOOLS.get(tool_name)
                if tool is None:
                    raise ValueError(f"Unknown tool: {tool_name}")

                with timer(tool_name):
                    result = tool(**args)
                if tool_name == "retrieve_documents":
                    retrieved_chunks = result
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
        if user_input.lower() == "show notes":
            show_notes()
            continue

        answer = run_agent(user_input)

        print(f"\nAssistant: {answer}")
