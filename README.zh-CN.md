# StrideLens 中文快速开始

[English](README.md)

## 快速开始与支持状态

### 问题
展示真实 CPU 张量是否共享存储、是否实际重叠，以及能否执行 view，而不导出张量值或原始地址。

### 安装
```sh
python -m pip install .
```

### 成功命令
```sh
python -m stridelens demo --out demo.html --json demo.json
```
预期：合成张量演示完成并写入本地 HTML/JSON 证据；退出码 0。

### 故意失败命令
```sh
python -m stridelens render missing-report.json --out should-not-exist.html
```
预期：缺失输入被拒绝，以非零退出码结束，且不发布报告。

### 证据与覆盖边界
证据使用真实 CPU 张量、直接标量地址 oracle、已知重叠/不相交用例、畸形报告、CLI 子进程和可选真实浏览器检查。结论受枚举预算限制；未知不会被当作不重叠。这不是性能分析器或 GPU 测量工具。

### 支持状态
版本 0.1.1 支持 Python 3.10+ 和精确的 CPU strided float32 torch.Tensor 输入。本地验证环境为 Python 3.12.3 与 torch 2.12.0+cpu；渲染已保存证据时 PyTorch 可选。维护为尽力而为，不提供 SLA。

### 限制
不支持 CUDA、稀疏、量化、meta、共轭/负位标志和张量子类。默认最多 32 个张量、16 个轴，每个张量最多枚举 4,096 个元素。名称和形状仍可能属于敏感元数据。
