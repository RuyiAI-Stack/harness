"""Run independent RabbitMQ relay, recovery and typed workers."""
import argparse
import asyncio
import json
import signal
from pathlib import Path

from codex_agent.runtime_config import runtime_config
from codex_agent.tasks.store import JobStore
from codex_agent.messaging.broker import Broker
from codex_agent.messaging.relay import publish_once
from .consumer import execute_delivery


async def serve(args):
    root = args.workspace.resolve()
    config = runtime_config()
    if not config.queue.enabled or (config.repoRoot and Path(config.repoRoot).resolve() != root):
        raise ValueError("enable queue and select the configured workspace")
    jobs = JobStore(root)
    if args.role == 'reconcile':
        if not args.confirm_no_active_execution or not args.job or not args.note:
            raise ValueError('reconcile requires --job, --note and --confirm-no-active-execution')
        print(json.dumps(jobs.reconcile(args.job, args.note)))
        return
    if args.role == 'execute':
        from .handlers import execute_job
        execute_job(root, args.job, args.owner)
        return
    if args.role == 'recover':
        while True:
            result = await asyncio.to_thread(jobs.recover)
            if result:
                print(json.dumps(result), flush=True)
            if args.once:
                return
            await asyncio.sleep(5)
    while True:
        try:
            async with Broker(jobs) as broker:
                if args.role == 'relay':
                    while True:
                        sent = await publish_once(jobs, broker)
                        if args.once:
                            return
                        if not sent:
                            await asyncio.sleep(.5)
                else:
                    async with broker.queues[args.role].iterator() as deliveries:
                        async for delivery in deliveries:
                            await execute_delivery(jobs, delivery, args.role)
                            if args.once:
                                return
        except Exception as error:
            print(json.dumps({'status': 'unavailable', 'component': args.role,
                              'error_type': type(error).__name__}), flush=True)
            if args.once:
                raise SystemExit(1) from None
            await asyncio.sleep(3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('role', choices=['relay', 'recover', 'agent', 'validation', 'memory', 'execute', 'reconcile'])
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--job')
    parser.add_argument('--owner')
    parser.add_argument('--note')
    parser.add_argument('--confirm-no-active-execution', action='store_true')
    args = parser.parse_args()
    if args.role == 'execute' and (not args.job or not args.owner):
        parser.error('execute requires job and owner')
    try:
        async def controlled():
            task = asyncio.current_task()
            asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, task.cancel)
            await serve(args)
        asyncio.run(controlled())
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass


if __name__ == '__main__':
    main()
