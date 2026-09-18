import asyncio
import sys
import os


# =========================================================
# PROJECT ROOT
# =========================================================

project_root = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, project_root)


# =========================================================
# MCP CLIENT IMPORTS
# =========================================================

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


# =========================================================
# MAIN TEST
# =========================================================

async def main():

    print("=" * 80)
    print("KERALA IT HUB - DAY 8 MCP COMPLETE TEST")
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

    async with stdio_client(server_params) as (
        read,
        write
    ):

        print("✅ MCP server process started.")


        # =================================================
        # CONNECT TO SERVER
        # =================================================

        async with ClientSession(
            read,
            write
        ) as session:

            print("\nSTEP 2: CONNECTING TO MCP SERVER")
            print("-" * 80)

            await session.initialize()

            print("✅ MCP connection initialized.")


            # =================================================
            # LIST TOOLS
            # =================================================

            print("\nSTEP 3: LISTING MCP TOOLS")
            print("-" * 80)

            tools_result = await session.list_tools()

            print(
                "Number of tools:",
                len(tools_result.tools)
            )

            for tool in tools_result.tools:

                print("\nTool name:")
                print(tool.name)

                print("\nDescription:")
                print(tool.description)


            # =================================================
            # TEST TOOL 1: SEARCH
            # =================================================

            print("\n" + "=" * 80)
            print("STEP 4: TESTING SEARCH TOOL")
            print("=" * 80)

            query = "Data Science courses in Kochi"

            print("\nQuery:")
            print(query)

            search_result = await session.call_tool(
                "search_kerala_courses",
                arguments={
                    "query": query,
                    "max_results": 5
                }
            )

            print("\n✅ Search tool executed successfully.")

            print("\nSearch results:")

            for content in search_result.content:

                if hasattr(content, "text"):
                    print(content.text)


            # =================================================
            # TEST TOOL 2: FETCH PAGE
            # =================================================

            print("\n" + "=" * 80)
            print("STEP 5: TESTING PAGE READER TOOL")
            print("=" * 80)

            test_url = (
                "https://www.rogersoft.com/"
                "course/data-science-training"
            )

            print("\nURL:")
            print(test_url)

            page_result = await session.call_tool(
                "fetch_course_page",
                arguments={
                    "url": test_url
                }
            )

            print("\n✅ Page reader tool executed successfully.")

            print("\nExtracted webpage content:")

            for content in page_result.content:

                if hasattr(content, "text"):

                    text = content.text

                    # Display only first 5000 characters
                    print(text[:5000])

                    print(
                        "\n...[content truncated for testing]..."
                    )


    # =====================================================
    # COMPLETE
    # =====================================================

    print("\n" + "=" * 80)
    print("DAY 8 MCP COMPLETE TEST FINISHED")
    print("=" * 80)

    print("""
MCP PIPELINE TESTED:

MCP Client
    ↓
MCP Server
    │
    ├── search_kerala_courses()
    │       ↓
    │   Web Search
    │
    └── fetch_course_page()
            ↓
        Page Reader
""")


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    asyncio.run(main())