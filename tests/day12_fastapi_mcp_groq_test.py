import asyncio
import json
import os
import sys

from fastapi import FastAPI
from fastapi.testclient import TestClient


# =========================================================
# PROJECT ROOT
# =========================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(
    0,
    PROJECT_ROOT
)


# =========================================================
# IMPORT FASTAPI APP
# =========================================================

from backend.main import app


# =========================================================
# MCP IMPORTS
# =========================================================

from mcp import (
    ClientSession,
    StdioServerParameters
)

from mcp.client.stdio import (
    stdio_client
)


# =========================================================
# GROQ RAG GENERATOR
# =========================================================

from backend.rag_generator import (
    generate_rag_answer
)


# =========================================================
# TEST QUESTION
# =========================================================

QUESTION = (
    "What Python topics are covered "
    "in Data Science courses in Kerala?"
)


# =========================================================
# MCP RETRIEVAL FUNCTION
# =========================================================

async def retrieve_from_mcp(
    question: str
):

    print(
        "\nStarting MCP server..."
    )


    # -----------------------------------------------------
    # MCP SERVER PARAMETERS
    # -----------------------------------------------------

    server_params = StdioServerParameters(

        command=sys.executable,

        args=[
            "-m",
            "mcp_server.course_search_server"
        ]
    )


    # -----------------------------------------------------
    # START MCP SERVER
    # -----------------------------------------------------

    async with stdio_client(
        server_params
    ) as (read, write):

        print(
            "MCP server started."
        )


        # -------------------------------------------------
        # CREATE MCP CLIENT SESSION
        # -------------------------------------------------

        async with ClientSession(
            read,
            write
        ) as session:

            await session.initialize()

            print(
                "MCP connection initialized."
            )


            # -------------------------------------------------
            # CALL RETRIEVAL TOOL
            # -------------------------------------------------

            print(
                "\nCalling MCP retrieval tool..."
            )


            result = await session.call_tool(

                "retrieve_course_information",

                arguments={

                    "query": question,

                    "max_results": 5,

                    "top_k": 5

                }
            )


            print(
                "MCP retrieval completed."
            )


            # -------------------------------------------------
            # EXTRACT RESULT
            # -------------------------------------------------

            for content in result.content:

                if hasattr(
                    content,
                    "text"
                ):

                    try:

                        data = json.loads(
                            content.text
                        )

                        return data

                    except json.JSONDecodeError:

                        continue


    return None


# =========================================================
# MAIN TEST
# =========================================================

async def main():

    print("=" * 80)

    print(
        "KERALA IT HUB - DAY 12 FASTAPI + MCP + GROQ TEST"
    )

    print("=" * 80)


    # =====================================================
    # STEP 1 — TEST FASTAPI DIRECTLY
    # =====================================================

    print(
        "\nSTEP 1: TESTING FASTAPI"
    )

    print("-" * 80)


    client = TestClient(
        app
    )


    response = client.get(
        "/health"
    )


    print(
        "FastAPI status code:",
        response.status_code
    )


    print(
        "FastAPI response:",
        response.json()
    )


    if response.status_code != 200:

        print(
            "\n❌ FastAPI health check failed."
        )

        return


    print(
        "\n✅ FastAPI health check passed."
    )


    # =====================================================
    # STEP 2 — TEST MCP RETRIEVAL
    # =====================================================

    print(
        "\n" + "=" * 80
    )

    print(
        "STEP 2: TESTING MCP RETRIEVAL"
    )

    print(
        "=" * 80
    )


    print(
        "\nQuestion:"
    )

    print(
        QUESTION
    )


    retrieved_data = await retrieve_from_mcp(
        QUESTION
    )


    if not retrieved_data:

        print(
            "\n❌ MCP retrieval returned no data."
        )

        return


    print(
        "\nSources checked:",
        retrieved_data.get(
            "sources_checked",
            0
        )
    )


    print(
        "Chunks created:",
        retrieved_data.get(
            "chunks_created",
            0
        )
    )


    retrieved_results = (
        retrieved_data.get(
            "retrieved_results",
            []
        )
    )


    print(
        "Relevant chunks:",
        len(
            retrieved_results
        )
    )


    if not retrieved_results:

        print(
            "\n❌ No relevant chunks retrieved."
        )

        return


    print(
        "\n✅ MCP retrieval passed."
    )


    # =====================================================
    # STEP 3 — SEND RETRIEVED DATA TO GROQ
    # =====================================================

    print(
        "\n" + "=" * 80
    )

    print(
        "STEP 3: TESTING GROQ RAG GENERATION"
    )

    print(
        "=" * 80
    )


    answer_result = generate_rag_answer(

        query=QUESTION,

        retrieved_results=retrieved_results

    )


    if not answer_result:

        print(
            "\n❌ Groq generation failed."
        )

        return


    answer = answer_result.get(
        "answer",
        ""
    )


    if not answer:

        print(
            "\n❌ Empty Groq answer."
        )

        return


    print(
        "\n✅ Groq answer generated."
    )


    print(
        "\n" + "-" * 80
    )

    print(
        "GENERATED ANSWER"
    )

    print(
        "-" * 80
    )


    print(
        "\n" + answer
    )


    # =====================================================
    # STEP 4 — SIMULATE FASTAPI RESPONSE
    # =====================================================

    print(
        "\n" + "=" * 80
    )

    print(
        "STEP 4: FASTAPI RESPONSE STRUCTURE"
    )

    print(
        "=" * 80
    )


    final_response = {

        "status": "success",

        "question": QUESTION,

        "answer": answer,

        "sources": answer_result.get(
            "sources",
            []
        )

    }


    print(
        "\nFinal API response:"
    )


    print(
        json.dumps(
            final_response,
            indent=2,
            ensure_ascii=False
        )
    )


    # =====================================================
    # FINAL STATUS
    # =====================================================

    print(
        "\n" + "=" * 80
    )

    print(
        "DAY 12 FASTAPI + MCP + GROQ TEST COMPLETE"
    )

    print(
        "=" * 80
    )


    print("""
FastAPI
   ↓
MCP Client
   ↓
MCP Server
   ↓
Web RAG
   ↓
Retrieved Context
   ↓
Groq
   ↓
API Response

✅ PIPELINE TEST SUCCESSFUL
""")


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    asyncio.run(
        main()
    )