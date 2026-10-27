"""Administrative account commands; never exposed as model tools."""
import argparse
from pathlib import Path

from .config import load_config
from .store import Accounts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("serve")
    sub.add_parser("invite")
    for name in ("disable", "enable"):
        sub.add_parser(name).add_argument("username")
    template = sub.add_parser("template")
    template.add_argument("--source", type=Path, required=True)
    template.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "template":
        from .workspaces import freeze_template
        freeze_template(args.source, args.destination)
        print("Template created; review its contents before enabling registration")
        return
    if not args.config:
        parser.error("--config is required")
    config = load_config(args.config)
    if args.command == "serve":
        import uvicorn
        from .api import create_gateway
        # Public HTTPS should terminate at a configured reverse proxy, not this loopback service.
        uvicorn.run(create_gateway(config), host="127.0.0.1", port=config.listen_port,
                    access_log=False, proxy_headers=False)
    else:
        accounts = Accounts(config)
        try:
            if args.command == "invite":
                print(accounts.invite())
            else:
                accounts.set_status(args.username.strip().lower(), args.command == "enable")
                print("Account status updated; previous login sessions revoked")
        finally:
            accounts.close()


if __name__ == "__main__":
    main()
