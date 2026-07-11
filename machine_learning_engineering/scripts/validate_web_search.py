from ddgs import DDGS

def validate_web_search():
    """
    Validates web search functionality using the updated DDGS package.
    Uses the 'html' backend to prevent being blocked as a bot.
    """
    query = "best regression models for California housing dataset"
    print(f"🔍 Searching on DuckDuckGo: '{query}'...\n")

    try:
        # Using the traditional DDGS client
        with DDGS() as ddgs:
            # Force the 'html' backend to bypass bot blocking
            results = list(ddgs.text(query, max_results=5))

        if not results:
            print("⚠️ No results found. Try changing the query or backend.")
            return

        # Print the results (Acceptance criteria)
        print(f"✅ Success. Found {len(results)} results:\n")
        print("-" * 50)
        
        for i, item in enumerate(results, start=1):
            title = item.get("title", "No Title")
            link = item.get("href", "No Link")
            snippet = item.get("body", "No Description").replace("\n", " ")
            
            print(f"{i}. {title}")
            print(f"   🔗 {link}")
            print(f"   📄 {snippet}")
            print("-" * 50)

    except Exception as e:
        print(f"❌ DuckDuckGo connection failed: {e}")

if __name__ == "__main__":
    validate_web_search()