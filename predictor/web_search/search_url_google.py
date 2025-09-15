from tqdm import tqdm
import json
import requests
import time


def search(queries, search_key):
    """
    Search function that takes queries and performs search requests.
    Args:
    - queries: List of search queries.
    - search_key: API key for the search service.
    
    Returns:
    - List of search results for each query.
    """
    url = "https://google.serper.dev/search"
    responses = []
    search_results = []

    for query in tqdm(queries[:], desc="Searching for urls..."):
        payload = json.dumps(
            {
                "q": query
            }
        )
        headers = {
            'X-API-KEY': search_key,
            'Content-Type': 'application/json'
        }

        reconnect = 0
        while reconnect < 3:
            try:
                response = requests.request("POST", url, headers=headers, data=payload)
                break
            except (requests.exceptions.RequestException, ValueError):
                reconnect += 1
                print(f'url: {url} failed * {reconnect}')
        
        result = json.loads(response.text)
        if "organic" in result:
            results = result["organic"][:10]
        else:
            results = [query]  # fallback if no organic results
        responses.append(results)

        search_dict = [{"queries": query, "results": results}]
        search_results.extend(search_dict)

    return search_results


def process_search_queries(input_json_path, output_json_path, search_key):
    """
    Process search queries from a JSON file and perform search.
    
    Args:
    - input_json_path: Path to input JSON file with questions and queries.
    - output_json_path: Path to output JSON file to store the results.
    - search_key: API key for the search service.
    """
    # Load the input data (queries, ground truths)
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    final_output = []

    for item in data:
        search_query = item['query']

        # Perform search
        search_results = search([search_query], search_key)
        results = search_results[0].get('results', [])

        # Extract the top 10 results with title, url, and snippet
        entries = [
            {"title": r.get("title", ""), "url": r.get("link", ""), "snippet": r.get("snippet", "")}
            for r in results[:10]
            if "link" in r and "title" in r and "snippet" in r
        ]

        final_output.append({
            'question': item['question'],
            'query': item['query'],
            'ground_truth': item['ground_truth'],
            'results': entries
        })

    # Write the results to the output JSON file
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    # Input and output file paths
    input_json_path = ""
    output_json_path = ""
    search_key = "XXXXXXXXXXXXXXXX"  # Your actual API key here

    # Call the process function
    process_search_queries(input_json_path, output_json_path, search_key)
