import argparse
import asyncio
import json
import os
from .config import RAGConfig
from .webconfig_server import start as start_webconfig

def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return RAGConfig.model_validate(json.load(f))

async def main_async():
    parser = argparse.ArgumentParser(prog='vectra')
    parser.add_argument('cmd', choices=['ingest','query','webconfig', 'dashboard'])
    parser.add_argument('target', nargs='?')
    parser.add_argument('--config')
    parser.add_argument('--stream', action='store_true')
    args = parser.parse_args()
    if args.cmd == 'webconfig':
        cfg_path = args.config or os.path.join(os.getcwd(), 'vectra-config.json')
        start_webconfig(cfg_path, 'webconfig')
        await asyncio.Event().wait()
        return

    if args.cmd == 'dashboard':
        cfg_path = args.config or os.path.join(os.getcwd(), 'vectra-config.json')
        start_webconfig(cfg_path, 'dashboard')
        await asyncio.Event().wait()
        return
    
    if not args.config:
        raise SystemExit('--config is required for ingest/query')
        
    # Lazy import to avoid hard dependencies for webconfig
    from .core import RAGClient
    
    cfg = load_config(args.config)
    client = RAGClient(cfg)
    if args.cmd == 'ingest':
        await client.ingest_documents(os.path.abspath(args.target))
        print('Ingestion complete')
    else:
        res = await client.query_rag(args.target, None, args.stream)
        if args.stream:
            async for chunk in res:
                t = chunk.get('delta') if isinstance(chunk, dict) else str(chunk)
                print(t, end='')
            print()
        else:
            print(json.dumps(res, ensure_ascii=False))

def main():
    asyncio.run(main_async())
