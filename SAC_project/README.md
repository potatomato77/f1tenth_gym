# SAC 赛车项目

后续的基线、残差控制、SAC 训练、评测和对应测试均放在本目录。
仿真器作为已有依赖使用；正式基线不导入 `examples/waypoint_follow.py`。

## 纯追踪基线

在项目根目录，使用现有虚拟环境运行：

```powershell
.\.venv\Scripts\python.exe -m SAC_project.baseline
```

默认参数为 Spielberg、赛道参考速度曲线（`--speed-mode profile --vgain 1`）、前视距离 1.2 m、种子
12345、ST/RK4、100 Hz、无渲染、关闭未参与控制的 LiDAR，墙接触保持开启。
先完成出场圈，再统计三个完整圈；最多运行 100000 个物理 step。
程序返回 0 表示成功，1 表示碰撞、超时或跟踪停止。
运行结束后自动生成 PNG 性能图，Windows 与无显示器的 Linux 均可使用。
绘图依赖已列在 `SAC_project/requirements.txt`，若当前环境尚未安装：

```powershell
.\.venv\Scripts\python.exe -m pip install -r SAC_project/requirements.txt
```

Windows 桌面显示路径与当前目标点：

```powershell
.\.venv\Scripts\python.exe -m SAC_project.baseline --render
```

短验证及参数调整：

```powershell
.\.venv\Scripts\python.exe -m SAC_project.baseline --speed-mode fixed --speed 2 --laps 1
.\.venv\Scripts\python.exe -m SAC_project.baseline --speed-mode profile --vgain 1 --laps 1
.\.venv\Scripts\python.exe -m SAC_project.baseline --speed-mode fixed --speed 2 --reference centerline --laps 1
.\.venv\Scripts\python.exe -m pytest -q SAC_project/tests
```

也可直接运行 `SAC_project/baseline.py`。Linux 服务器使用相同的模块入口，
不加 `--render` 时无需显示环境。地图缺失时仍会按仿真器规则下载。

## 控制契约

`PurePursuitController.plan(cog_pose)` 接受 `[质心 x, 质心 y, 车身 yaw]`。
它先通过 `rear = cog - lr * [cos(yaw), sin(yaw)]` 换到后轴，再完成投影、
精确圆与线段求交、目标相对位置和曲率计算。路径显式闭合，连续重复点会移除。

正常跟踪使用沿路径前进顺序的第一个前视圆精确交点。偏离路径、位于前视圆
之外时，从最近投影沿路径前进一个前视距离取得恢复目标。两种模式统一使用
`atan(2 * wheelbase * target_lateral / actual_target_distance**2)`。
目标不在车身前方、路径过远或无有效交点时，输出零转角、零速目标并报告原因。
零速目标通过车辆内部控制器实现制动，不意味着实际速度瞬间归零。

`PursuitCommand.action` 是独立 float32 数组，形状 `(2,)`，顺序固定为
`[目标转角 rad, 目标速度 m/s]`。发送原生环境时补成 `action[None, :]`。
`limit_action()` 显式限制转角到车辆允许范围、速度到 `[0, v_max]`，后续残差
叠加后也应经过同一入口。
本基线的车辆参数固定，未启用域随机化；后续启用随机限制时须同步设计限幅契约。

## 两种速度模式

- `--speed-mode profile`（默认）：读取所选参考线的 `vxs`，在后轴最近投影所在的
  线段两端速度之间做线性插值，再乘以 `--vgain`。闭合段插值回首点；移除重复路径点
  时同步移除对应速度，保证位置与速度对齐。速度取自车辆当前位置，不是前视目标点。
- `--speed-mode fixed --speed 2`：全程使用指定速度，适合低速几何验证。

默认 `--reference raceline` 的速度来源为
`maps/Spielberg/Spielberg_raceline.csv` 的 `vx_mps` 列，当前范围约 4.51–8 m/s。
`--reference centerline` 则使用 centerline 自己的 `vxs`；由中心线文件构造时通常是
加载器生成的固定 1 m/s，不会自动套用另一条 racing line 的速度表。
`--speed` 只作用于 fixed 模式，`--vgain` 只作用于 profile 模式。
这些值均是目标速度，实际速度由仿真器内部控制器和车辆动力学产生。

控制器类直接构造时仍默认 fixed，便于独立几何测试；`run_baseline()` 和命令行入口
默认 profile。两种模式均保留物理限幅，遇到无效跟踪目标时仍发出停车目标。

## 结果与计时

每次运行在 `runs/` 下生成独立目录，也可以用 `--output-dir` 指定：

- `trajectory.csv`：每步运动后的质心／后轴位置、实际速度和转角、两种跟踪误差，
  最近参考点、位置范数误差、速度／航向／转向误差，以及产生该步运动的目标点、
  原始转角、原始期望速度、限幅后命令、限幅标志、模式和碰撞信息。
- `reference_path.csv`：实际使用的闭合参考折线、折线弧长及各点的期望速度。
- `summary.json`：成功状态、完整圈时间、各类误差 RMS/MAE/P95/最大值/均值、
  速度范围、限幅比例、关键配置和图表文件名。
- `performance.png`：六个面板，显示速度对比、速度误差、位置范数误差、带符号路径误差、
  转向对比和航向误差。
- `speed_tracking.png`：期望速度与实际速度在同一张图上；发生速度限幅时另外显示限幅后的目标。
- `position_error.png`：后轴、质心分别到参考折线的最近距离。
- `speed_error.png`：实际速度减去该步施加的目标速度。运行超过 2 秒时同时提供
  全程图与 2 秒之后的局部放大图，避免起步大误差遮住后续细节；统计仍保留全部样本。
- `trajectory.png`：参考路径、后轴轨迹和质心轨迹叠加，标记首末采样点或碰撞点。

使用 `--no-plots` 可以仅输出 CSV/JSON。对于本版产生的运行目录，可以不重跑仿真而重新绘图：

```powershell
.\.venv\Scripts\python.exe -m SAC_project.reporting SAC_project/runs/你的运行目录
```

图中文字使用英文和 SI 单位，避免服务器缺少中文字体；CSV/JSON 中的角度为 rad，
图中的转向和航向角转换为 deg。图表使用 Agg 离屏绘制，不弹出额外窗口。

`count_partial_first_lap=False` 将首次过线作为计时起点，随后才计完整圈。
`full_lap_times_s` 从环境的圈计时读取，不从 reward 累加；总仿真时间包含出场圈。
误差是对所选参考折线的距离，正号表示路径左侧，汇总覆盖全部 step、包含出场圈。
后轴是本控制器的跟踪参考点；质心误差独立报告，不等同于默认 centerline Frenet 误差。

`position_error_norm = ||rear_position - closest_reference_point||_2`，始终非负；
它不是到前视目标点的距离，也不是按时间对齐的完整轨迹跟踪误差。
`rear_error`、`cog_error` 保留左右方向的符号。
`speed_error = actual_speed - command_speed`；其中 command_speed 是经过物理限幅的目标。
`reference_speed` 则保存限幅前的期望速度，另有 `reference_speed_error` 显示对原始期望的误差。
`heading_error` 是车身 yaw 减去后轴投影所在参考线段的朝向，并回绕到 `[-pi, pi)`。

一行数据的时间戳来自 `info["sim_time"]`，对应当前物理步结束时刻。
位置、实际速度、实际转角和路径误差取步后状态；目标速度、目标转角及前视目标点是
该步开始时计算并施加的命令，不会误用下一步的新命令。速度误差统计包含起步加速过程。

本地用途是调试与验证；正式 SAC 训练放在租用的服务器。

## 残差环境封装

已新增 `residual_env.py`、`features.py` 和 `rewards.py`。首版策略只输出一维归一化转向
残差，默认最大修正 0.05 rad；纵向保持与基线相同。观察默认 25 维，包含后轴误差、
运动状态、基线动作、最近动作历史与局部路径预瞄。每个策略 step 对应一个 0.01 s 物理步。

完整设计、特征顺序、奖励公式和时间对齐说明见
[残差环境导读](RESIDUAL_ENV_GUIDE_ZH.md)。正式训练前先运行：

```powershell
.\.venv\Scripts\python.exe -m SAC_project.validate_residual --speed-mode profile --laps 3
.\.venv\Scripts\python.exe -m SAC_project.validate_residual --speed-mode fixed --speed 2 --laps 1
```

验证器要求可选的 SB3 checker 依赖，见 `requirements-rl.txt`；它不运行 SAC 优化。
输出位于 `runs/residual_validation_.../`，包含零残差对照 CSV／PNG、随机动作短回放和
带配置的 JSON。零残差比较物理动作、七维状态、计圈与终止；任务奖励使用新定义，
不要求与原生 survival reward 相等。

## SAC 训练与评测

`train.py` 用 SB3 SAC 训练残差策略，`evaluate.py` 在相同任务上比较学习策略与
零残差基线。两者共享 `rl_common.py` 中的任务配置，训练结束会把完整配置写入
`train_config.json`，评测可直接从运行目录恢复同一套特征、奖励和动作契约。

### 本地链路验收

```powershell
# 64 步、16 更新起步、小 buffer；只验证链路，不代表学习效果
.\.venv\Scripts\python.exe -m SAC_project.train --smoke
```

`--smoke` 会强制使用 CPU、单环境、`learning_starts=16`、`batch_size=16`、
`buffer_size=256`、`total_timesteps=64`，并在 32 和 64 步保存 checkpoint。
正常输出应包含 `n_updates` 大于零、loss 有限，以及 `sac_final.zip`、
`replay_buffer.pkl`、`logs/progress.csv` 和 `train_config.json`。

断点恢复：

```powershell
.\.venv\Scripts\python.exe -m SAC_project.train --smoke `
  --resume SAC_project/runs/上次运行/sac_final.zip `
  --resume-replay SAC_project/runs/上次运行/replay_buffer.pkl
```

训练步数会从 checkpoint 的 `num_timesteps` 继续累加，而不是从零开始。
恢复要求当前任务配置与 checkpoint 的观察／动作空间一致；跨平台不保证逐位一致。

### 正式训练

Linux 服务器、无渲染：

```bash
python -m SAC_project.train --total-timesteps 500000 --n-envs 8 --vec-env subproc \
  --checkpoint-freq 50000 --device auto
```

本地或快速调试可保持 `--n-envs 1 --vec-env dummy`。`SubprocVecEnv` 在 Windows 使用
spawn，入口已加 `multiprocessing.freeze_support()`，环境工厂是模块顶层函数。
Linux 上可用 fork，但任务的数值复现仍应以固定种子和行为指标为准。

默认 SAC 参数针对 100 Hz 决策：`gamma=0.999`、每环境步一次梯度更新、
`learning_starts=10000`。奖励权重、残差幅度、特征顺序等在训练期间应保持冻结；
修改后需要新开运行目录。服务器首次运行前记得把 `maps/` 下使用的地图拷过去，
`maps/` 不随 git 或 Docker 镜像分发。

关键输出：

- `sac_final.zip` 和 `sac_checkpoint_*_steps.zip`：SB3 模型。
- `replay_buffer.pkl`：最终 replay buffer，用于恢复训练。
- `train_config.json`：完整任务配置、特征顺序与尺度、git commit、依赖版本。
- `logs/progress.csv`：总奖励、损失和 `residual/*` 任务指标。

### 策略评测

```powershell
# 自动读取运行目录中的配置和 sac_final.zip
.\.venv\Scripts\python.exe -m SAC_project.evaluate `
  --run-dir SAC_project/runs/sac_xxx --episodes 5

# 指定模型；显式参数会覆盖运行目录中的环境配置
.\.venv\Scripts\python.exe -m SAC_project.evaluate `
  --model SAC_project/runs/sac_xxx/sac_final.zip --episodes 5

# 本地观看行驶过程（默认 CPU 推理）
.\.venv\Scripts\python.exe -m SAC_project.evaluate `
  --run-dir SAC_project/runs/sac_xxx --episodes 1 --render

# 保存每步轨迹
.\.venv\Scripts\python.exe -m SAC_project.evaluate `
  --run-dir SAC_project/runs/sac_xxx --episodes 5 --trajectory
```

每个种子先跑零残差基线，再跑学习策略，两者使用同一环境、同一初态和同一奖励。
输出目录包含 `summary.json`、`episodes.csv`，使用 `--trajectory` 时另有
`trajectories/*.csv`。指标包括成功率、碰撞／跟踪停止／超时率、后轴误差 RMS 与 P95、
航向误差、残差均值与峰值、限幅比例、完整圈时间、回合回报和推理延迟。
失败回合保留在统计中；圈时间只汇总已完成圈，并同时报告成功率。

`evaluate.py` 默认 `--device cpu`，保证回放稳定；`--device auto/cuda` 需要对应的
PyTorch 构建。服务器无显示器时不要加 `--render`；本地 5060 目前安装的是 CPU 版
Torch，模型可直接加载，但推理仍在 CPU。

## 当前验证记录

此前固定低速验证：Spielberg、种子 12345、fixed 2 m/s、前视 1.2 m，完成三个完整圈，无碰撞、无限幅：
完整圈时间分别为 169.17、169.16、169.17 s。后轴误差 RMS 约 0.91 cm，
质心误差 RMS 约 0.93 cm；误差统计包含出场圈，不能与此前高速示例直接比较。
本记录为 Windows 上的无渲染运行，未据此宣称真实图形窗口已验证。

本版速度曲线与图表验证：Spielberg、前视 1.2 m、profile、vgain=1，种子
12345／42／2026 各完成三个完整圈，无碰撞；圈时约 45.55–45.56 s。
种子 12345 的后轴位置范数 RMSE 约 0.0927 m，速度 RMSE 约 0.3323 m/s，
统计均包含起步与出场圈。fixed 2 m/s 对照完成一个完整圈，圈时约 169.17 s。
这些是静态网格起点下的本地验证，尚未代表随机扰动或跨地图鲁棒性。
结果及五张 PNG 均保存在 `runs/validation_speed_modes_20261018_112142/` 的各次运行目录。

残差环境接口验证：项目测试 **62 passed**，SB3 checker 通过。Spielberg、种子 12345，
profile 模式逐步比较 18256 步并完成三个完整圈；fixed 2 m/s 比较 33812 步并完成一个
完整圈；fixed 10 m/s 时两套环境均在第 1170 步撞墙。三种场景的物理动作与七维标准状态
均逐位一致，最大差值为 0。每种模式另外执行 256 步随机残差接口验证，无训练更新。
详见 `runs/residual_interfaces_20261018_114557/`；profile 目录还记录了每项奖励分量。

SAC 链路 smoke 验证（2026-10-21，Windows CPU）：`--smoke` 完成 64 个环境步和 48 次
梯度更新，loss 有限，保存了 checkpoint、最终模型与 replay buffer；从 64 步 checkpoint
恢复后继续到 128 步，`num_timesteps` 正确累加。`evaluate.py` 用该未训练模型与零残差
基线各跑 1 个回合，两者均完成三个完整圈，圈时 45.55 s，基线后轴误差 RMS 与既有记录
一致（0.0927 m）。另外验证了 Windows 上 `SubprocVecEnv` 双环境训练、短回合截断时
replay 的 `dones`/`timeouts` 同时置位，以及 fixed 10 m/s 碰撞时只有 `dones` 置位
（第 1170、2345 步，无 timeout）。新增配置测试后项目为 **65 passed**。
这验证的是训练、保存、恢复、加载和评测链路，不代表 SAC 已学到改善。
