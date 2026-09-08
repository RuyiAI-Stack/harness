# dsh-env-builder

[中文](README.zh.md) | English

Purpose: Manage ephemeral environment directories and GitHub component installs for Singularity / workload tools.

Package: `@dangosys/dsh-env-builder`

Dependencies: tools, webServer

config.yaml: none

### Tools

1. env_list: List environment/projectN dirs and components.

2. env_register_component: Register a cloned and built repo in the manifest (after bash install).

3. env_remove_component: Remove a component from an environment.

4. env_set_component_status: Set status ready | modified | installing.

5. env_select: Select the active environment.

6. env_select_repo: Select the active repo inside the selected environment.

7. env_ensure_component: Ensure the selected repo component is cloned; return local path.

8. env_bind_component_session: Bind a session id to a component.

9. env_attach_session / env_detach_session: Attach / detach sessions on an environment.

10. env_rename: Rename an environment id.

11. env_set_running: Set the Live running flag.

12. env_reset: git reset --hard && git clean -fd on every component.

13. env_delete: Delete an environment directory and manifest entry.

Environment creation for singularity mode is owned by `@dangosys/dsh-singularity-agent` (`env_create`).

### Web APIs

Prefix: `/integrations/env-builder`

1. GET `/environments`: List environments.

2. GET `/selected`: Current selected environment and repo.

3. POST `/selected-repo`: Select repo (body: `{ repo }`).

4. DELETE `/environments/:id`: Delete environment.

5. POST `/environments/:id/select|reset|running|rename`: Select / reset / running / rename.

6. POST|DELETE `/environments/:id/components`: Remove component (DELETE only; install via agent bash + env_register_component).

7. POST `/environments/:id/component-status|component-session`: Component status / session bind.

8. POST|DELETE `/environments/:id/sessions`: Attach / detach session.

### Service state

1. ctx.envBuilder.store: manifest and project dirs under `$RUYI_ROOT/environment`
