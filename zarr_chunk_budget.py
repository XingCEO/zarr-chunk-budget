"""Estimate chunk read amplification against your actual slice workload."""

import math


def analyze(data):
    shape=data['shape']; item=data['item_bytes']; candidates=data['candidates']; queries=data['queries']
    if not shape or any(type(x) is not int or x<=0 for x in shape): raise ValueError('shape must contain positive integers')
    if type(item) is not int or item<=0: raise ValueError('item_bytes must be a positive integer')
    penalty=float(data.get('request_penalty_bytes',0))
    if not math.isfinite(penalty) or penalty<0: raise ValueError('invalid request penalty')
    parsed=[]
    for q in queries:
        start,stop=q['start'],q['stop']; weight=float(q.get('weight',1))
        if len(start)!=len(shape) or len(stop)!=len(shape): raise ValueError('query dimensionality mismatch')
        if any(type(a) is not int or type(b) is not int or not 0<=a<=b<=n for a,b,n in zip(start,stop,shape)): raise ValueError('invalid slice bounds')
        if not math.isfinite(weight) or weight<0: raise ValueError('weight must be nonnegative')
        parsed.append((start,stop,weight))
    reports=[]
    for chunks in candidates:
        if len(chunks)!=len(shape) or any(type(c) is not int or c<=0 for c in chunks): raise ValueError('invalid chunk shape')
        requests=fetched=logical=0
        for start,stop,weight in parsed:
            logical+=math.prod(b-a for a,b in zip(start,stop))*item*weight
            if any(a==b for a,b in zip(start,stop)): continue
            counts=[]; extents=[]
            for a,b,c,n in zip(start,stop,chunks,shape):
                first,last=a//c,(b-1)//c
                counts.append(last-first+1)
                extents.append(min(n,(last+1)*c)-first*c)
            requests+=math.prod(counts)*weight
            fetched+=math.prod(extents)*item*weight
        reports.append({'chunks':chunks,'weighted_requests':requests,'weighted_fetched_bytes':fetched,'weighted_logical_bytes':logical,'read_amplification':fetched/logical if logical else None,'score':fetched+penalty*requests,'total_array_chunks':math.prod((n+c-1)//c for n,c in zip(shape,chunks))})
    reports.sort(key=lambda r:(r['score'],r['chunks']))
    return {'best':reports[0]['chunks'] if reports else None,'candidates':reports,'model':'uncompressed, no cache, unit-stride rectangular reads'}


def main():
    """Read one UTF-8 JSON document and emit one machine-readable report."""
    import argparse
    import json
    import sys
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="JSON input path, or - for stdin")
    args = parser.parse_args()
    try:
        raw = sys.stdin.read() if args.input == "-" else Path(args.input).read_text(encoding="utf-8")
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("input must be a JSON object")
        result = analyze(data)
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    except (OSError, ValueError, KeyError, TypeError, OverflowError) as error:
        print(f"Invalid input: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
