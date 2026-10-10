"""Polling transactional outbox. Network waits never hold a database transaction."""
import asyncio


async def publish_once(jobs, broker):
    event = await asyncio.to_thread(jobs.claim_publication)
    if event is None:
        return False
    try:
        await broker.publish(event)
    except Exception as error:
        # The exception message can contain a credential-bearing AMQP URL.
        await asyncio.to_thread(jobs.publication_result, event, type(error).__name__)
    else:
        await asyncio.to_thread(jobs.publication_result, event)
    return True
