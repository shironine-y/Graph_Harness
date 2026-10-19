# 基于事理图谱的AppWorld Harness

## 1. 安装

要求 Python >= 3.11，以及另行安装、下载好数据并可正常运行的 AppWorld。请遵循所用 AppWorld 版本的官方安装方式，在其工作虚拟环境中安装本项目依赖；本项目不会安装完整 AppWorld 数据或改写其源码。

在本目录执行：

```bash
python -m pip install -r requirements.txt
```

## 2. 配置信息

仓库提供模板 `config.example.json`，首次使用时复制为本地私有配置 `config.json`，再按自己的环境编辑：

```bash
cp config.example.json config.json
```

Windows PowerShell 使用 `Copy-Item config.example.json config.json`。

必须修改：

- `runtime.python_executable`：AppWorld 虚拟环境的解释器路径；也支持 PATH 中的 `python` 命令。正式运行推荐指定完整虚拟环境路径，不解引用其符号链接。
- `runtime.appworld_root`：AppWorld 仓库根目录，须包含 `src/appworld`、`data/datasets` 及任务数据。示例的 `../appworld` 仅是占位布局。
- `model.name/base_url`：实际模型名与 OpenAI 兼容服务地址。
- `neo4j`：实际数据库连接及图谱版本；关闭 `enabled` 可运行无图谱建议对照。

相对路径按配置文件所在目录解析，启动器支持 `~` 和环境变量展开。脚本入口默认来自本发布目录；不要指向旧版本后端。

Linux 凭据示例：

```bash
export MODEL_API_KEY='YOUR_SERVICE_KEY'
# 本地无鉴权模型服务也需提供非空占位值，例如 EMPTY。
export NEO4J_PASSWORD='YOUR_DATABASE_PASSWORD'
```

Windows PowerShell 使用 `$env:MODEL_API_KEY`、`$env:NEO4J_PASSWORD`。启动器不会自动加载 `.env`。默认通过 `model.api_key_env` 和 `neo4j.password_env` 读取凭据；

## 3. 启动与任务选择

```bash
python start_appworld_online_pilot.py --check
python start_appworld_online_pilot.py

# 等价 Shell 入口。
bash start_appworld_online_pilot.sh
```

`--check` 检查解释器、依赖、任务集、后端参数和已启用的 Neo4j 连接，不调用模型生成或执行任务；它不能保证模型推理可用。`run.mode="plan"` 只预览选择与参数，不执行任务，但可以查询图谱、服务及写启动清单。`runtime.background=true` 在实际在线模式启动独立后台进程；启动成功不表示任务已完成。

只需修改 `run.dataset` 即可切换任务集，例如 `dev`、`test_normal`、`test_challenge`、任意未来分组名；可带 `.txt` 后缀，不接受路径。文件须位于：

```text
<appworld_root>/data/datasets/<dataset>.txt
```

每行一个任务 ID，不能重复。`run.test_tasks="all"` 选择全文件；整数表示固定种子随机抽样数量。非空 `run.test_task_ids` 优先于数量，按给定顺序执行并检查文件成员关系。

## 4. 输出结果

默认启动日志位于 `outputs/appworld_launches/`，每次运行的报告位于 `outputs/<dataset>_<run_id>/`。实际路径以终端打印和启动清单为准：

- `plan.json`：任务、数据集文件哈希、预算、模型采样和特性开关。
- `task_results.jsonl`：每个已结束任务一行，含官方评价计数、耗时与审计路径。
- `report.json`：数据集汇总、官方评价及报告一致性错误。
- `config_snapshot.json`：脱敏后的有效配置。
- `launch_*.result.json`：退出码及结束时间。

AppWorld 原生任务 Trace、数据库及 `logs/harness_events.jsonl` 位于 `<appworld_root>/experiments/outputs/<experiment_name>/tasks/`，并不全部放在 Harness 报告目录。

Agent 标记完成与官方通过分开统计。测点通过率由通过数除以总测点数计算；任务耗时之和不等于并行实验的墙钟时间。默认仍调用官方评价。

## 5. 自定义任务集

使用者可自行按组整理任务 ID，每个 TXT 文件中每行一个 ID，组内不能重复。将文件放到 AppWorld 数据集目录顶层，例如：

```text
<appworld_root>/data/datasets/MyGroup.txt
```

在现有完整配置中，将 `run` 部分的相关字段改为下面的值，不要用该片段覆盖整个配置文件：

```json
{
  "run": {
    "dataset": "MyGroup",
    "test_tasks": "all",
    "test_task_ids": []
  }
}
```

保留 `run.run_official_evaluation=true`，然后使用正常入口启动：

```bash
python start_appworld_online_pilot.py --config config.json
```

非空 `test_task_ids` 会覆盖全组选择，因此运行完整分组时应清空该列表。等待该组任务执行与官方评价结束后，再切换下一组；若需要重复实验，可再次启动，每次会创建独立输出。可为各组分别准备自己的私有配置文件，并通过 `--config` 指定；不要在运行过程中修改正在使用的配置。



本仓库仅发布Harness代码与配置模板；AppWorld、任务数据及图谱需另行准备。真实凭据、私有配置和运行输出不应提交。