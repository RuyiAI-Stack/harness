from codex_agent.test_planner import plan_tests


def targets():
    return [
        {"id":"pytest::a", "name":"a", "kind":"pytest", "path":"python/a.py", "source_files":["python/a.py"], "likely_area":"general", "command":"pytest python/a.py"},
        {"id":"build::core", "name":"core", "kind":"build", "path":"lib/CMakeLists.txt", "source_files":["lib/core.cpp"], "dependencies":[], "command":"cmake --build . --target core"},
        {"id":"build::app", "name":"app", "kind":"build", "path":"app/CMakeLists.txt", "source_files":["app/main.cpp"], "dependencies":["core"], "command":"cmake --build . --target app"},
        {"id":"pytest::b", "name":"b", "kind":"pytest", "path":"python/b.py", "source_files":["python/b.py"], "likely_area":"general", "command":"pytest python/b.py"},
    ]


def test_quick_maps_source_and_dependency_closure():
    plan = plan_tests(["lib/core.cpp"], targets(), tier="quick")
    assert plan.tier == "quick"
    assert [t.id for t in plan.targets] == ["build::core", "build::app"]
    assert "dependency impact" in plan.targets[1].reasons
    assert plan.parallel_groups


def test_standard_adds_same_area_tests_and_history_regression():
    plan = plan_tests(["python/a.py", "python/b.py"], targets(), tier="standard", history=[{"id":"pytest::b", "status":"failed"}])
    assert plan.tier == "standard"
    assert {t.id for t in plan.targets} == {"pytest::a", "pytest::b"}
    assert plan.history_regressions == ["pytest::b"]
    assert plan.targets[1].regression


def test_auto_promotes_build_config_to_full():
    plan = plan_tests(["CMakeLists.txt"], targets())
    assert plan.tier == "full"
    assert len(plan.targets) == len(targets())
