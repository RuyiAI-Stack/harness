export function failExpPromptText(operation: string): string {
  return `## Network Failure

Failed operation: ${operation}

When a dependency download, clone, submodule update, or package install fails because the network is unavailable or unreliable:

1. Stop retrying after the first clearly network-related failure.
2. Tell the user which operation failed and ask whether they have an HTTP/HTTPS or SOCKS proxy.
3. If they provide one, confirm the scope (temporary command, current shell, or project config) before applying it. Reuse existing proxy environment variables when present; do not persist credentials or edit global Git configuration without permission.
4. If no proxy is available, offer offline/local-cache options and report exactly what remains uninstalled.`
}
