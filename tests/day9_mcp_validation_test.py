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
# MCP IMPORTS
# =========================================================

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


# =========================================================
# MAIN TEST
# =========================================================

async def main():

    print("=" * 80)
    print("KERALA IT HUB - DAY 9 MCP VALIDATION TEST")
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

        print("✅ MCP server started.")


        # =================================================
        # CONNECT TO SERVER
        # =================================================

        async with ClientSession(
            read,
            write
        ) as session:

            print("\nSTEP 2: INITIALIZING MCP CONNECTION")
            print("-" * 80)

            await session.initialize()

            print("✅ MCP connection initialized.")


            # =================================================
            # LIST TOOLS
            # =================================================

            print("\nSTEP 3: CHECKING AVAILABLE TOOLS")
            print("-" * 80)

            tools_result = await session.list_tools()

            print(
                "Number of tools:",
                len(tools_result.tools)
            )

            for tool in tools_result.tools:

                print(
                    f"✅ {tool.name}"
                )


            # =================================================
            # TEST 1
            # =================================================

            print("\n" + "=" * 80)
            print("TEST 1: DATA SCIENCE COURSES IN KOCHI")
            print("=" * 80)

            query = "Data Science courses in Kochi"

            print("\nQuery:")
            print(query)

            result = await session.call_tool(
                "search_kerala_courses",
                arguments={
                    "query": query,
                    "max_results": 5
                }
            )

            print("\nResult:")

            for content in result.content:

                if hasattr(content, "text"):
                    print(content.text[:3000])


            # =================================================
            # TEST 2
            # =================================================

            print("\n" + "=" * 80)
            print("TEST 2: PYTHON COURSES IN TRIVANDRUM")
            print("=" * 80)

            query = "Python courses in Trivandrum"

            print("\nQuery:")
            print(query)

            result = await session.call_tool(
                "search_kerala_courses",
                arguments={
                    "query": query,
                    "max_results": 5
                }
            )

            print("\nResult:")

            for content in result.content:

                if hasattr(content, "text"):
                    print(content.text[:3000])


            # =================================================
            # TEST 3
            # =================================================

            print("\n" + "=" * 80)
            print("TEST 3: CYBERSECURITY COURSES IN KERALA")
            print("=" * 80)

            query = "Cybersecurity courses in Kerala"

            print("\nQuery:")
            print(query)

            result = await session.call_tool(
                "search_kerala_courses",
                arguments={
                    "query": query,
                    "max_results": 5
                }
            )

            print("\nResult:")

            for content in result.content:

                if hasattr(content, "text"):
                    print(content.text[:3000])


            # =================================================
            # TEST 4
            # =================================================

            print("\n" + "=" * 80)
            print("TEST 4: EMPTY QUERY")
            print("=" * 80)

            result = await session.call_tool(
                "search_kerala_courses",
                arguments={
                    "query": "",
                    "max_results": 5
                }
            )

            print("\nResult:")

            for content in result.content:

                if hasattr(content, "text"):
                    print(content.text)


            # =================================================
            # TEST 5
            # =================================================

            print("\n" + "=" * 80)
            print("TEST 5: INVALID URL")
            print("=" * 80)

            invalid_url = "https://this-is-not-a-real-course-page-12345.com"

            print("\nURL:")
            print(invalid_url)

            result = await session.call_tool(
                "fetch_course_page",
                arguments={
                    "url": invalid_url
                }
            )

            print("\nResult:")

            for content in result.content:

                if hasattr(content, "text"):
                    print(content.text)


            # =================================================
            # TEST 6
            # =================================================

            print("\n" + "=" * 80)
            print("TEST 6: VALID COURSE WEBPAGE")
            print("=" * 80)

            valid_url = (
                "https://www.rogersoft.com/"
                "course/data-science-training"
            )

            print("\nURL:")
            print(valid_url)

            result = await session.call_tool(
                "fetch_course_page",
                arguments={
                    "url": valid_url
                }
            )

            print("\nResult:")

            for content in result.content:

                if hasattr(content, "text"):

                    text = content.text

                    print(text[:5000])

                    print(
                        "\n...[content truncated]..."
                    )


            # =================================================
            # FINAL
            # =================================================

            print("\n" + "=" * 80)
            print("DAY 9 MCP VALIDATION TEST COMPLETED")
            print("=" * 80)

            print("""
MCP VALIDATION FLOW:

User Query
    ↓
MCP Client
    ↓
MCP Server
    ↓
search_kerala_courses()
    ↓
Web Search

OR

MCP Client
    ↓
MCP Server
    ↓
fetch_course_page()
    ↓
Webpage
    ↓
Extracted Course Content
""")


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    asyncio.run(main())