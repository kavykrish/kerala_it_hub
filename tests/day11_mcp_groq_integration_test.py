import asyncio
import json
import os
import sys


# =========================================================
# PROJECT ROOT
# =========================================================

project_root = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(
    0,
    project_root
)


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
# GROQ IMPORT
# =========================================================

from backend.rag_generator import (
    generate_rag_answer
)


# =========================================================
# MAIN
# =========================================================

async def main():

    print("=" * 80)
    print(
        "KERALA IT HUB - DAY 11 MCP + RAG + GROQ TEST"
    )
    print("=" * 80)


    # =====================================================
    # START MCP SERVER
    # =====================================================

    server_params = StdioServerParameters(

        command=sys.executable,

        args=[
            "-m",
            "mcp_server.course_search_server"
        ]
    )


    print("\nSTEP 1: STARTING MCP SERVER")
    print("-" * 80)


    async with stdio_client(
        server_params
    ) as (read, write):

        print(
            "✅ MCP server started."
        )


        # =================================================
        # CONNECT TO MCP SERVER
        # =================================================

        async with ClientSession(
            read,
            write
        ) as session:

            print(
                "\nSTEP 2: INITIALIZING MCP CONNECTION"
            )
            print("-" * 80)


            await session.initialize()


            print(
                "✅ MCP connection initialized."
            )


            # =================================================
            # USER QUESTION
            # =================================================

            query = (
                "What Python topics are covered "
                "in Data Science courses in Kerala?"
            )


            print(
                "\n" + "=" * 80
            )

            print(
                "STEP 3: USER QUESTION"
            )

            print(
                "=" * 80
            )

            print(
                "\nQuestion:"
            )

            print(
                query
            )


            # =================================================
            # CALL MCP RAG TOOL
            # =================================================

            print(
                "\n" + "=" * 80
            )

            print(
                "STEP 4: MCP RAG RETRIEVAL"
            )

            print(
                "=" * 80
            )


            print(
                "\nCalling retrieve_course_information()..."
            )


            mcp_result = await session.call_tool(

                "retrieve_course_information",

                arguments={

                    "query": query,

                    "max_results": 5,

                    "top_k": 5

                }
            )


            print(
                "\n✅ MCP retrieval completed."
            )


            # =================================================
            # EXTRACT JSON RESULT
            # =================================================

            retrieved_data = None


            for content in mcp_result.content:

                if hasattr(
                    content,
                    "text"
                ):

                    try:

                        retrieved_data = json.loads(
                            content.text
                        )

                    except json.JSONDecodeError:

                        print(
                            "Could not parse MCP result as JSON."
                        )


            if not retrieved_data:

                print(
                    "\n❌ No retrieval data received."
                )

                return


            # =================================================
            # DISPLAY RETRIEVAL SUMMARY
            # =================================================

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

            print(
                "Relevant chunks:",
                len(
                    retrieved_data.get(
                        "retrieved_results",
                        []
                    )
                )
            )


            # =================================================
            # SEND RETRIEVED DATA TO GROQ
            # =================================================

            print(
                "\n" + "=" * 80
            )

            print(
                "STEP 5: GENERATING GROUNDED ANSWER"
            )

            print(
                "=" * 80
            )


            retrieved_results = (
                retrieved_data.get(
                    "retrieved_results",
                    []
                )
            )


            print(
                "\nSending retrieved context to Groq..."
            )


            answer_result = generate_rag_answer(

                query=query,

                retrieved_results=retrieved_results

            )


            print(
                "\n✅ Groq answer generated."
            )


            # =================================================
            # DISPLAY FINAL ANSWER
            # =================================================

            print(
                "\n" + "=" * 80
            )

            print(
                "FINAL AI ANSWER"
            )

            print(
                "=" * 80
            )


            print(
                "\n"
                + answer_result["answer"]
            )


            # =================================================
            # DISPLAY SOURCES
            # =================================================

            print(
                "\n" + "=" * 80
            )

            print(
                "RETRIEVED SOURCES"
            )

            print(
                "=" * 80
            )


            for source in answer_result[
                "sources"
            ]:

                print(
                    f"\n- {source['title']}"
                )

                print(
                    f"  {source['url']}"
                )


            # =================================================
            # FINAL PIPELINE
            # =================================================

            print(
                "\n" + "=" * 80
            )

            print(
                "DAY 11 MCP + RAG + GROQ PIPELINE"
            )

            print(
                "=" * 80
            )


            print("""
User Question
      ↓
MCP Client
      ↓
MCP Server
      ↓
Web Search
      ↓
Course Webpages
      ↓
Text Cleaning
      ↓
Chunking
      ↓
Hybrid Retrieval
      ↓
Relevant Course Context
      ↓
Groq LLM
      ↓
Grounded Answer
      ↓
Sources
""")


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    asyncio.run(main())