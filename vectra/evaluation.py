from typing import List, Dict, Any

async def evaluate_test_set(client, test_set: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    report: List[Dict[str, Any]] = []
    for item in test_set:
        res = await client.query_rag(item['question'])
        ctx = "\n".join([s.get('summary','') for s in res.get('sources', [])])
        fp = f"Rate 0-1: Is the following Answer derived only from the Context?\nContext:\n{ctx}\n\nAnswer:\n{res.get('answer') if isinstance(res.get('answer'), str) else str(res.get('answer'))}"
        rp = f"Rate 0-1: Does the Answer correctly answer the Question?\nQuestion:\n{item['question']}\n\nAnswer:\n{res.get('answer') if isinstance(res.get('answer'), str) else str(res.get('answer'))}"
        faith = 0.0; rel = 0.0
        try:
            faith = max(0.0, min(1.0, float(str(await client.llm.generate(fp, 'You return a single number between 0 and 1.')))))
        except Exception:
            faith = 0.0
        try:
            rel = max(0.0, min(1.0, float(str(await client.llm.generate(rp, 'You return a single number between 0 and 1.')))))
        except Exception:
            rel = 0.0
        report.append({ 'question': item['question'], 'expectedGroundTruth': item.get('expectedGroundTruth',''), 'faithfulness': faith, 'relevance': rel })
    return report
