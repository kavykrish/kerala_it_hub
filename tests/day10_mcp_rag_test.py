import asyncio
import sys
import os


# =========================================================
# PROJECT ROOT
# =========================================================

project_root = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
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
# MAIN
# =========================================================

async def main():

    print("=" * 80)
    print("KERALA IT HUB - DAY 10 MCP RAG TEST")
    print("=" * 80)


    # =====================================================
    # START SERVER
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
        # CONNECT
        # =================================================

        async with ClientSession(
            read,
            write
        ) as session:

            print(
                "\nSTEP 2: INITIALIZING CONNECTION"
            )
            print("-" * 80)


            await session.initialize()


            print(
                "✅ MCP connection initialized."
            )


            # =================================================
            # LIST TOOLS
            # =================================================

            print(
                "\nSTEP 3: CHECKING MCP TOOLS"
            )
            print("-" * 80)


            tools_result = (
                await session.list_tools()
            )


            print(
                "Number of tools:",
                len(tools_result.tools)
            )


            for tool in tools_result.tools:

                print(
                    f"✅ {tool.name}"
                )


            # =================================================
            # TEST RAG RETRIEVAL
            # =================================================

            print(
                "\n" + "=" * 80
            )

            print(
                "STEP 4: TESTING RAG RETRIEVAL"
            )

            print(
                "=" * 80
            )


            query = (
                "What Python topics are covered "
                "in Data Science courses in Kerala?"
            )


            print(
                "\nQuery:"
            )

            print(
                query
            )


            print(
                "\nCalling MCP RAG tool..."
            )


            result = await session.call_tool(

                "retrieve_course_information",

                arguments={

                    "query": query,

                    "max_results": 5,

                    "top_k": 5

                }
            )


            print(
                "\n✅ RAG retrieval completed."
            )


            # =================================================
            # DISPLAY RESULT
            # =================================================

            print(
                "\n" + "=" * 80
            )

            print(
                "RETRIEVED COURSE INFORMATION"
            )

            print(
                "=" * 80
            )


            for content in result.content:

                if hasattr(
                    content,
                    "text"
                ):

                    print(
                        content.text
                    )


            # =================================================
            # SECOND QUERY
            # =================================================

            print(
                "\n" + "=" * 80
            )

            print(
                "STEP 5: TESTING MACHINE LEARNING QUERY"
            )

            print(
                "=" * 80
            )


            query = (
                "Does the course include "
                "machine learning?"
            )


            print(
                "\nQuery:"
            )

            print(
                query
            )


            result = await session.call_tool(

                "retrieve_course_information",

                arguments={

                    "query": query,

                    "max_results": 5,

                    "top_k": 5

                }
            )


            print(
                "\n✅ Retrieval completed."
            )


            for content in result.content:

                if hasattr(
                    content,
                    "text"
                ):

                    print(
                        content.text
                    )


    # =====================================================
    # FINAL
    # =====================================================

    print(
        "\n" + "=" * 80
    )

    print(
        "DAY 10 MCP RAG TEST FINISHED"
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":

    asyncio.run(main())