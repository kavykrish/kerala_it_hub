import os

from dotenv import load_dotenv
from groq import Groq


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

api_key = os.getenv(
    "GROQ_API_KEY"
)

if not api_key:
    raise ValueError(
        "GROQ_API_KEY not found. "
        "Please check your .env file."
    )


# ============================================================
# GROQ CLIENT
# ============================================================

client = Groq(
    api_key=api_key
)


MODEL_NAME = "openai/gpt-oss-20b"


# ============================================================
# RAG ANSWER GENERATOR
# ============================================================

def generate_rag_answer(
    query: str,
    retrieved_results: list
):
    """
    Generate an answer using retrieved RAG context.

    The LLM answers only from the retrieved
    information and avoids unsupported claims.
    """

    # --------------------------------------------------------
    # Validate query
    # --------------------------------------------------------

    if not query:

        return {
            "answer": "Please provide a question.",
            "sources": []
        }


    # --------------------------------------------------------
    # Validate retrieved results
    # --------------------------------------------------------

    if not retrieved_results:

        return {
            "answer": (
                "I could not find relevant course "
                "information for this question."
            ),
            "sources": []
        }


    # ========================================================
    # PREPARE RETRIEVED CONTEXT
    # ========================================================

    context_parts = []

    sources = []

    seen_sources = set()

    source_index = 1


    for result in retrieved_results:

        content = result.get(
            "content",
            ""
        )

        source_title = result.get(
            "source_title",
            ""
        )

        source_url = result.get(
            "source_url",
            ""
        )


        # ----------------------------------------------------
        # Skip empty content
        # ----------------------------------------------------

        if not content:
            continue


        # ----------------------------------------------------
        # Create unique source key
        # ----------------------------------------------------

        source_key = source_url.strip()


        # ----------------------------------------------------
        # First chunk from a source
        # ----------------------------------------------------

        if (
            source_key
            and source_key not in seen_sources
        ):

            seen_sources.add(
                source_key
            )


            context_parts.append(
                f"""
SOURCE {source_index}

Title:
{source_title}

URL:
{source_url}

Content:
{content}
"""
            )


            sources.append({
                "title": source_title,
                "url": source_url
            })


            source_index += 1


        # ----------------------------------------------------
        # Additional chunk from the same source
        # ----------------------------------------------------

        else:

            context_parts.append(
                f"""
ADDITIONAL CONTENT

Title:
{source_title}

URL:
{source_url}

Content:
{content}
"""
            )


    # --------------------------------------------------------
    # Combine all retrieved content
    # --------------------------------------------------------

    context = "\n".join(
        context_parts
    )


    # ========================================================
    # SYSTEM PROMPT
    # ========================================================

    system_prompt = """
You are Kerala IT Hub, an AI-powered IT course
and institute information assistant.

Your job is to answer questions about IT and
technology courses in Kerala.

Use ONLY the information provided in the
retrieved context.

IMPORTANT RULES:

1. Do not invent course names, fees, durations,
   eligibility requirements, locations, admission
   details, curriculum details, or other course
   information.

2. Treat the retrieved information as evidence
   from the listed sources, not as complete
   information about every IT course in Kerala.

3. Never use words such as:
   "all", "only", "every", "none", "best",
   "most", or "the complete list"
   unless the retrieved information explicitly
   supports that claim.

4. If the question asks for a list of courses,
   present the courses found in the retrieved
   results.

   Clearly indicate that the list is based on
   the retrieved sources and may not represent
   every course available in Kerala.

5. If a course or institute was not found in the
   retrieved results, do not claim that it
   does not exist.

6. If the retrieved information does not contain
   enough evidence to answer the question,
   clearly say that the information was not found
   in the retrieved sources.

7. Do not combine eligibility, fees, duration,
   curriculum, or other details from different
   institutes and present them as if they belong
   to the same institute.

8. When information comes from multiple institutes,
   clearly associate each detail with the correct
   institute or course.

9. Do not treat marketing claims such as
   "best", "No.1", "guaranteed job", "100%
   placement", or similar promotional statements
   as verified facts.

10. Preserve uncertainty.

    If the source says that fees, duration,
    eligibility, or other details depend on
    the batch, program, or conditions, do not
    provide a fixed value.

11. Prefer specific information over general
    assumptions.

12. Keep the answer concise and easy to understand.

13. When listing courses, institutes, or programs,
    mention the associated institute or provider
    whenever that information is available.

14. Do not claim that the retrieved list is
    complete unless the source explicitly states
    that it is complete.

15. Do not infer information that is not directly
    supported by the retrieved content.

16. At the end of the answer, provide a
    "Sources" section containing the relevant
    source titles and URLs.

17. When presenting information from a source,
    do not change its meaning or create facts
    that are not present in the retrieved content.
"""


    # ========================================================
    # USER PROMPT
    # ========================================================

    user_prompt = f"""
Question:

{query}


Retrieved course information:

{context}


Using only the retrieved course information,
answer the question accurately.

Remember:

- Do not invent information.
- Do not assume missing information.
- Clearly distinguish between different institutes.
- If the results are incomplete, say that they
  are based on the retrieved sources.
- Do not claim that a course does not exist simply
  because it was not found in the retrieved results.
"""


    # ========================================================
    # GROQ API CALL
    # ========================================================

    response = client.chat.completions.create(
        model=MODEL_NAME,

        messages=[
            {
                "role": "system",
                "content": system_prompt
            },

            {
                "role": "user",
                "content": user_prompt
            }
        ],

        temperature=0
    )


    # ========================================================
    # EXTRACT ANSWER
    # ========================================================

    answer = (
        response
        .choices[0]
        .message
        .content
    )


    # ========================================================
    # RETURN RESULT
    # ========================================================

    return {
        "answer": answer,
        "sources": sources
    }