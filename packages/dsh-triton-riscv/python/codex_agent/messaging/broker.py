"""One workspace-scoped durable queue per worker role, with confirmed publishing."""
import json
import os
from urllib.parse import urlsplit

import aio_pika
from pamqp.commands import Basic

from codex_agent.runtime_config import runtime_config


class Broker:
    def __init__(self, jobs):
        self.jobs = jobs
        self.prefix = f"triton.{jobs.db.cache_namespace}.{jobs.db.workspace_id}"
        self.connection = None

    async def __aenter__(self):
        cfg = runtime_config().queue
        url = os.environ.get(cfg.urlEnv, "")
        if not cfg.enabled or urlsplit(url).scheme not in {"amqp", "amqps"}:
            raise ValueError(f"Enable queue and set {cfg.urlEnv} to an AMQP URL")
        try:
            self.connection = await aio_pika.connect(url, timeout=5)
            self.channel = await self.connection.channel(publisher_confirms=True, on_return_raises=True)
            await self.channel.set_qos(prefetch_count=1)
            self.exchange = await self.channel.declare_exchange(self.prefix, aio_pika.ExchangeType.DIRECT, durable=True)
            dead = await self.channel.declare_exchange(self.prefix + ".dead", aio_pika.ExchangeType.FANOUT, durable=True)
            queue = await self.channel.declare_queue(self.prefix + ".dead", durable=True)
            await queue.bind(dead)
            self.queues = {}
            for kind in ("agent", "validation", "memory"):
                queue = await self.channel.declare_queue(self.prefix + "." + kind, durable=True,
                    arguments={"x-dead-letter-exchange": dead.name})
                await queue.bind(self.exchange, routing_key=kind)
                self.queues[kind] = queue
            return self
        except Exception:
            if self.connection is not None:
                await self.connection.close()
            raise RuntimeError("RabbitMQ unavailable or topology incompatible; check service/configuration") from None

    async def __aexit__(self, *_):
        if self.connection is not None:
            await self.connection.close()

    async def publish(self, event):
        message = aio_pika.Message(json.dumps(self.jobs.envelope(event)).encode(),
            content_type="application/json", delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            message_id=event["id"], type=event["kind"])
        confirmed = await self.exchange.publish(message, routing_key=event["kind"], mandatory=True, timeout=5)
        if not isinstance(confirmed, Basic.Ack):
            raise RuntimeError("RabbitMQ did not confirm publication")
