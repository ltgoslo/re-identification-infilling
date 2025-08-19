import requests
import json
import tqdm

if __name__ == '__main__':
    headers = {'Accept': 'application/json'}
    articles = []
    for i in tqdm.trange(1, 4423):  # 4422 is the number of pages with 500 items per page
        try:
            worked = False
            while not worked:
                try:
                    response = requests.get(f"https://api.wp1.openzim.org/v1/projects/Biography/articles?page={i}&numRows=500", headers=headers)
                    data = response.json()
                    worked = True
                except Exception:
                    print("Retrying...")
            for art in data['articles']:
                articles.append(art)
        # If the request fails (404) then print the error.
        except requests.exceptions.HTTPError as error:
            print(error)
        print(f"Total length of articles parsed: {len(articles)}")

        if len(articles) % 20000 == 0:
            with open(f'../datasets/bios_{len(articles)}.json', 'w') as fout:
                json.dump(articles, fout)

    with open(f'../datsets/bios_{len(articles)}.json', 'w') as fout:
        json.dump(articles, fout)

    with open("../datasets/article_titles.txt", "w") as fout:
        for article in articles:
            fout.write(article["article"] + "\n")
