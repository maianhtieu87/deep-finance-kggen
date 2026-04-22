import json, os

path = r'd:\deep-finance-kggen\data\interim\kg_embeddings\news_embeddings_finbert.json'
print(f'File size: {os.path.getsize(path) / 1024:.1f} KB')

with open(path, 'r', encoding='utf-8') as f:
    data = json.load(f)

dates = sorted(data.keys())
print(f'Total dates: {len(dates)}')

if not dates:
    print('>>> FILE IS EMPTY <<<')
else:
    print(f'Date range: {dates[0]} --> {dates[-1]}')
    sample_date = dates[len(dates)//2]
    tickers = data[sample_date]
    print(f'Tickers on {sample_date}: {list(tickers.keys())}')
    if tickers:
        sample_ticker = list(tickers.keys())[0]
        vec = tickers[sample_ticker]
        print(f'Vector dim [{sample_ticker}]: {len(vec)}')
        print(f'Sample values: {[round(x,4) for x in vec[:5]]}')

    total_pairs = sum(len(v) for v in data.values())
    all_dims = set(len(v) for d in data.values() for v in d.values() if v)
    print(f'Total (date, ticker) pairs: {total_pairs}')
    print(f'Dims found in file: {all_dims}')
