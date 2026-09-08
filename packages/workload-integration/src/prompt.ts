export function playbookText(repoPath: string): string {
  const modelsRoot = `${repoPath}/bb-tests/workloads/src/ModelTest/e2e/models/models`
  return `# Buckyball ModelTest e2e workload（初始模型适配）

把用户的模型链接，变成 ${modelsRoot}/<X>/ 下的一个新 workload，并完成三处注册。
本阶段只做模型侧：在本地把该模型跑出与参考一致的正确结果。不绑 chip/core，
不跑 rushB / 仿真 / FPGA，也不做 buddy-opt / buddy-translate / buddy-llc 等 lowering。

开始前先调用 \`env_ensure_component({ name: "buckyball" })\`，使用当前选中环境的 checkout（${repoPath}）。

## 修改范围

不超过 ${modelsRoot}/<X>/，外加下面三处注册。

## 要写的东西

对照已有同类模型，通常是：

- \`import-*.py\`（或同类 importer）
- 一个 \`*-main.cpp\` driver
- \`CMakeLists.txt\`
- 必要资产（如图、vocab）
- \`HANDOFF.md\`

三处注册（细节见 \`buckyball_existing_models_index\`）：

1. \`e2e/models/CMakeLists.txt\` 的 MODEL reset 列表
2. \`e2e/models/models/CMakeLists.txt\` 的 \`MODEL_<X>_DIR\` + \`if(MODEL_<X>) add_subdirectory\`
3. \`bbdev/api/steps/workload/01_build_event.step.py\` 的 \`MODEL_CMAKE\`

不要写：\`archs/buckyball/\`、\`quant/\`、\`trace/\`、\`specs/*\` + \`*Runner\`（RAX）、
per-core 编译器、bbdev \`MODEL_TARGETS\` / \`MODEL_LAYOUT\`。参考里若有这些，跳过。

## 环境

统一使用buckyball目录的nix环境，不要使用其他环境。nix 根目录位于 ${repoPath}。

## 步骤

1. \`env_ensure_component({ name: "buckyball" })\`
2. \`buckyball_model_info\` → 弄清层类型 / shape / dtype / checkpoint。
3. \`buckyball_existing_models_index\` → 选同类参考，搭目录并注册。
4. 用官方/原始实现准备一个固定输入，得到正确的全输出参考（不要只比 argmax）。
5. 在本地跑通该模型：默认 CPU；若单次推理超过 10s（例如一张图），改用本地 GPU。
   输出必须与参考在约定容差内一致。
6. 写 \`HANDOFF.md\`，再跑 \`buckyball_workload_audit\`，必须 \`ACCEPT\`。
7. 在 https://github.com/buddy-compiler/buddy-examples 仓库开 PR 交付。

## HANDOFF.md

放在 ${modelsRoot}/<X>/HANDOFF.md。这是给评审/后续阶段看的交接说明：本 workload
写了什么、怎么复现、本地怎么跑通、还有哪些已知限制。每条断言引用命令输出或
文件路径，不要写空话。

五个英文标题（audit 按此检查）：

1. Local Run：设备（CPU/GPU）、命令、耗时
2. Build Binding：用过的 configure/build 命令（如有）

## 交付（开 PR）

\`ACCEPT\` 之后，在 \`${repoPath}\`（Buckyball 仓库）开 PR：

- 基于最新 main 建分支，只包含本 workload 写集与三处注册
- 标题/正文写清：模型链接、本地跑通设备（CPU/GPU）、audit 结果
- 用 \`gh pr create\` 创建；推送/开 PR 失败就停并报告，不要假装已交付`
}
