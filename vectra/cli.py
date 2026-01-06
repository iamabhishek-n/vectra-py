import argparse
import asyncio
import json
import os
from .config import VectraConfig, SessionType
from .webconfig_server import start as start_webconfig
from .telemetry import telemetry

def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return VectraConfig.model_validate(json.load(f))

async def main_async():
    parser = argparse.ArgumentParser(prog='vectra')
    parser.add_argument('cmd', choices=['ingest','query','webconfig', 'dashboard'])
    parser.add_argument('target', nargs='?')
    parser.add_argument('--config')
    parser.add_argument('--stream', action='store_true')
    args = parser.parse_args()
    if args.cmd == 'webconfig':
        # Telemetry is tricky here without full config load, but we try
        telemetry.init() 
        telemetry.track('cli_command_used', {'command': 'webconfig', 'flags': []})
        cfg_path = args.config or os.path.join(os.getcwd(), 'vectra-config.json')
        start_webconfig(cfg_path, 'webconfig')
        await asyncio.Event().wait()
        return

    if args.cmd == 'dashboard':
        telemetry.init()
        telemetry.track('cli_command_used', {'command': 'dashboard', 'flags': []})
        cfg_path = args.config or os.path.join(os.getcwd(), 'vectra-config.json')
        start_webconfig(cfg_path, 'dashboard')
        await asyncio.Event().wait()
        return
    
    if not args.config:
        raise SystemExit('--config is required for ingest/query')
        
    # Lazy import to avoid hard dependencies for webconfig
    from .core import VectraClient
    
    cfg = load_config(args.config)
    cfg.session_type = SessionType.CLI
    
    # Telemetry init happens inside VectraClient, but we want to track CLI command too
    # We can rely on VectraClient init or do it here explicitly.
    # VectraClient init handles enabling/disabling based on config.
    # So we should probably do it after client creation?
    # But track needs initialized telemetry.
    
    client = VectraClient(cfg)
    
    telemetry.track('cli_command_used', {
        'command': args.cmd,
        'flags': ['--stream'] if args.stream else []
    })

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
            
    telemetry.flush()

def main():
    asyncio.run(main_async())
