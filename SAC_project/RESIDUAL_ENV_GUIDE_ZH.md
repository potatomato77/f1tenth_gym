# 残差环境：观察、动作、奖励与零残差验收

首版目标是已知地图、可获得仿真状态的单车路径跟踪。纯追踪产生基本动作，策略只补充
转向修正，纵向继续使用同一份参考速度方案。当前实现与验证不启动 SAC 训练。

## 1. 环境封装到底做什么

```text
策略看到 25 维观察，输出 z ∈ [-1, 1]
  -> 当前状态对应的纯追踪动作 + 转向残差
  -> 显式物理限幅
  -> SingleAgentWrapper 转换动作形状
  -> 原有 F110Env 推进一个物理步
  -> 更新纯追踪动作、策略观察、任务奖励和诊断
```

代码分工：

- `baseline.py`：已有纯追踪控制器，保持其几何和速度方案。
- `features.py`：观察的固定顺序、归一化、局部路径预瞄。
- `rewards.py`：进度解缠和各项任务奖励。
- `residual_env.py`：Gymnasium 生命周期、动作合成、历史与终止处理。
- `validate_residual.py`：两个独立模拟器的零残差对照和随机动作短验证。
- `tests/test_residual_env.py`：无需下载地图的合成场景与契约测试。

首版一个策略 step 对应一个 0.01 s 的物理 step，即 100 Hz，没有 action repeat。
后续改变决策频率时，应同时重新核对基线对照、历史、奖励和折扣的时间尺度。

## 2. 动作：一个修正旋钮

策略动作空间为 `Box(-1, 1, shape=(1,), dtype=float32)`：

```text
requested_residual = 0.05 rad * z
steering_cmd = clip(base_action[0] + requested_residual, s_min, s_max)
speed_cmd = base_action[1]
```

这里的 `base_action` 已经通过基线的限幅；合成后再次调用同一个 `limit_action()`。
默认最大修正 0.05 rad，约 2.86°，是待训练验证的超参数，可以用 `ResidualConfig` 调整。
例如基线 0.10 rad、z=0.4，则请求修正 0.02 rad，最终目标为 0.12 rad（若未触发限幅）。

正转角代表向左转，负转角代表向右转。动作是**目标转角**，不是立即写入实际车轮角度。
实际转角和速度仍由已有执行器控制与车辆动力学产生。形状错误、越界或非有限动作会
在推进物理状态之前被拒绝，封装不会偷偷裁剪一个无效的策略动作。

日志区分：

- `policy_action`：策略输出的归一化 z，后续 replay buffer 应保存这个动作。
- `requested_steering_residual_rad`：期望添加的修正量。
- `effective_steering_residual_rad`：限幅后最终转角减去基线转角。
- `physical_action`：实际发送给仿真器的 `[目标转角, 目标速度]`。

例如基线已经接近转角上限，策略继续输出正修正，实际增加量会小于 0.05 rad。
有限残差和限幅只限制指令大小，不构成避免碰撞的保证。
若基线因路径过远、目标在后方或无有效前视点进入 stop 模式，残差不会覆盖其停车命令。

## 3. 观察：默认 25 维，顺序固定

所有位置误差、投影和预瞄位置都使用基线的后轴参考点：

```text
rear = CoG - lr * [cos(yaw), sin(yaw)]
```

`std_state` 的位置是质心，不能将它直接当后轴，也不能重复减去 lr。
后轴到所选参考折线的最近投影定义横向误差，左侧为正。航向误差是
`wrap(body_yaw - reference_segment_yaw)`；这里使用折线段的方向，与控制几何一致。

| 下标 | 特征 | 默认归一化尺度／范围 |
|---|---|---|
| 0 | 后轴带符号路径距离 | max_reacquire_distance，默认 3 m |
| 1–2 | 航向误差 sin、cos | 本身在 [-1,1] |
| 3 | 实际速度 | max(abs(v_min), abs(v_max))，默认 20 m/s |
| 4 | 当前基线目标速度 | 同上；使用限幅后的可执行目标 |
| 5 | 实际车轮转角 | max(abs(s_min), abs(s_max))，默认 0.4189 rad |
| 6 | 横摆角速度 | 默认 10 rad/s |
| 7–8 | 质心侧偏角 beta 的 sin、cos | 本身在 [-1,1] |
| 9 | 当前基线目标转角 | 与实际车轮转角相同 |
| 10 | 上次策略残差 z | [-1,1] |
| 11–12 | 上次最终转角、速度命令 | 与各自物理量相同 |
| 13–15 | 路径前方 1 m 的局部 x、局部 y、目标速度 | 坐标默认 11 m；速度 20 m/s |
| 16–18 | 前方 2 m 的同类信息 | 同上 |
| 19–21 | 前方 4 m 的同类信息 | 同上 |
| 22–24 | 前方 8 m 的同类信息 | 同上 |

完整名称可读取 `env.feature_names`；尺度与配置可保存 `env.describe()`。
坐标尺度取“最远预瞄距离 + 最大重捕获距离”，默认 8+3=11 m。

前方点按参考折线的**弧长**取样，不是任取后面四行 CSV。设世界坐标偏移为 dx、dy：

```text
local_x =  cos(yaw) * dx + sin(yaw) * dy
local_y = -sin(yaw) * dx + cos(yaw) * dy
```

这样相似弯道即使在地图的不同位置、朝向不同，策略也能看到相似的局部几何。
预瞄速度使用同一条参考线的速度插值与物理限幅；固定速度模式下四个预瞄速度也固定。

使用 sin/cos 是为了避免 +179° 与 -179° 被误当成相差 358°。
beta 则描述速度方向相对车身朝向的偏差，用于呈现 ST 模型的侧偏运动。
上次命令可帮助理解最近的执行过程，但一帧历史不保证长延迟任务具有完整 Markov 状态。

最终观察为独立的 float32 数组，统一裁剪到 [-1,1]，超过尺度的分量数记录为
`feature_clip_count`。原始状态与误差保留在 info 中，不会因观察裁剪而被修改。
策略不读取车辆真实附着系数或奖励分量；目前使用已知地图和仿真状态，并不包含感知学习。

## 4. 奖励：每一项都有明确意义

默认公式：

```text
r = progress_used_m
    - dt * [4*(ey/0.5m)^2 + (ephi/0.5rad)^2
            + 0.1*z^2 + 0.05*(z-z_previous)^2]
    - 20 * I(collision or tracking_stop)
```

| 项 | 作用 |
|---|---|
| progress | 鼓励沿参考路径前进；倒退产生负值 |
| lateral | 减少后轴路径偏离 |
| heading | 减少车身朝向与路径朝向偏差 |
| residual | 减少没有收益的额外干预 |
| smoothness | 减少相邻决策之间的残差突变 |
| failure | 碰撞或基线失去有效跟踪目标的单次失败成本 |

进度由**相同参考折线的后轴投影**计算，不直接复用原生 info 中的中心线进度。
过首尾连接处时，将 s 的差值回绕到 [-L/2, L/2)，避免绕一圈后突然得到约 -L 的奖励。
最近投影偶尔可能切换到另一段路径，因此用于奖励的进度限制在
`±2 * speed_scale * dt` 内。原值与使用值分别存为 `progress_raw_m`、`progress_used_m`，
发生限制时有 `progress_clipped` 标志，便于检查异常投影。

持续惩罚乘 dt，失败成本只扣一次。终止后必须 reset，不能继续 step 并反复扣碰撞成本。
速度误差作为观察和诊断，本阶段不单独设置纵向跟踪奖励，因为策略没有速度残差动作。

一个可手算的例子：dt=0.01 s、前进 0.08 m、ey=0.1 m、ephi=0、z=0.4、上次 z=0.2：

```text
progress   = +0.08
lateral    = -0.0016
heading    =  0
residual   = -0.00016
smoothness = -0.00002
总奖励     =  0.07822（未发生失败）
```

这些权重是可解释的初始值，尚未通过训练选优。后续应结合奖励分量范围和独立评测结果调整，
不能只看总回报。原环境的默认奖励 0.01 保留为 `base_reward` 供检查，不加进任务奖励。

## 5. 时间顺序与 reset

reset 时：清空上次 z／物理命令；从当前状态计算基线动作、投影、观察，并将当前 s 作为
进度起点。出生在赛道中段不会直接获得“从零走到出生点”的进度奖励。

每个 step 的顺序：

1. 验证 z，读取当前状态对应的缓存基线动作。
2. 合成、限幅并施加物理命令，恰好推进一个原生物理步。
3. 根据步后状态计算下一次基线动作与观察；新观察中的历史是刚施加过的 z 和命令。
4. 用前后投影计算进度，以步后跟踪误差计算奖励。
5. 返回观察、任务奖励、终止／超时标志和独立的诊断副本。

`info["residual"]["base_action"]` 是产生该步运动的基线动作；
`next_base_action` 对应返回观察里的新状态。不能混用二者。
时间使用原生 `info["sim_time"]`，不读取原环境故意滞后一拍的观察时钟。

碰撞和完整圈结束保留原生 terminated；时间上限保留 truncated。
基线 runner 遇到 stop 原本会跳出循环，残差环境将同一条件显式表示为任务 terminated，
并保留 `base_terminated`、`base_truncated` 说明原生状态；不会提前引入额外的误差终止阈值。
如果碰撞与超时同一步发生，原生两个标志都会保留，不能假设它们永远互斥。

当前版本固定车辆参数并要求角度／目标速度控制。改变车辆参数、地图或观察布局时应
重新构造封装；域随机化需要后续补上每回合的控制器几何、尺度和限制同步，当前显式拒绝启用。

## 6. 零残差验收：验证的是物理行为

零残差意味着合成后的命令恰好等于基线命令。它不意味着转角、速度或任务奖励为零。
基线观察是字典，残差观察是特征向量，也不要求二者格式相同。

验收使用两个独立的 F110Env，共享只读地图缓存，具有独立状态与随机生成器，并使用相同
配置与 seed。每个物理步比较：

- 两维物理命令，包括目标速度。
- 七维标准状态：位置、实际转角、速度、yaw、yaw rate、beta。
- 碰撞、计圈、圈时、仿真时钟和终止／超时位置。
- 残差观察是否在空间内、奖励是否有限。

同时记录是否逐位一致，以及每个状态分量的最大绝对差。默认数值容差为 1e-6；
跨平台的复现应以数值容差和行为指标评估，不承诺 Windows 与 Linux 逐位一致。
奖励使用新的定义，所以原生 survival reward 与任务奖励不作为等价比较对象。

接口还检查：正负方向、非对称物理限幅、无效动作不推进状态、reset 清空历史、
返回数组不别名、真实墙接触、碰撞后禁止继续 step、SB3 checker、随机残差短回放。

## 7. 本地运行

在仓库根目录：

```powershell
# 安装可选 checker／后续 SAC 依赖（当前环境已经具备）
.\.venv\Scripts\python.exe -m pip install -r SAC_project/requirements-rl.txt

# 只运行测试，不训练
.\.venv\Scripts\python.exe -m pytest -q SAC_project/tests

# 速度曲线模式：完整出场圈之后，再验证三个完整圈
.\.venv\Scripts\python.exe -m SAC_project.validate_residual --speed-mode profile --laps 3

# 固定低速对照
.\.venv\Scripts\python.exe -m SAC_project.validate_residual --speed-mode fixed --speed 2 --laps 1
```

每次在 `SAC_project/runs/residual_validation_.../` 生成：

- `summary.json`：等价性、逐位一致性、逐分量误差、成功／碰撞情况、完整特征顺序与配置。
- `zero_residual.csv`：每步的双模拟器对照，以及六项奖励分量。
- `zero_residual_equivalence.png`：轨迹、实际速度和差值曲线。
- `random_residual.csv`：默认 256 步随机动作与奖励分量的检查记录，遇到终止立即停止。

验证程序返回 0 说明接口与等价性检查通过；不等于所有测试速度都能完成赛道。
例如固定 10 m/s 时，如果两套环境在同一步撞墙，也属于正确复现基线，
报告仍会明确 `baseline_success=false` 与 `collision=true`。

最小调用：

```python
import numpy as np
from SAC_project.residual_env import make_residual_env

env = make_residual_env(speed_mode="profile", laps=1)
try:
    obs, info = env.reset(seed=12345)
    obs, reward, terminated, truncated, info = env.step(np.zeros(1, dtype=np.float32))
    print(info["residual"]["physical_action"])
    print(info["residual"]["reward_terms"])
finally:
    env.close()
```

本阶段通过之后，再在 Linux 服务器接 SAC 的短更新、保存／恢复和正式训练。
后续学习器必须沿用这里的特征顺序、尺度、残差幅度和动作语义。

## 8. 本次实际验收

Windows 无渲染运行，Spielberg、seed=12345、前视 1.2 m：

| 速度方案 | 比较物理步数 | 原基线与零残差封装的表现 | 七维状态／两维物理动作最大差值 |
|---|---:|---|---:|
| profile、vgain=1 | 18256 | 均完成出场圈之后的三个完整圈 | 0 |
| fixed 2 m/s | 33812 | 均完成出场圈之后的一个完整圈 | 0 |
| fixed 10 m/s | 1170 | 均在 11.70 s 碰撞结束 | 0 |

项目测试 62 passed，包含 26 项新增残差环境测试；SB3 checker 通过。
每种速度方案另跑 256 步随机残差，观察、动作范围与奖励有限性检查通过。
这些是接口等价性与回放结果，SAC 尚未在本阶段训练。
结果位于 `runs/residual_interfaces_20261018_114557/`。profile 目录通过模块入口复跑，
CSV 还包含完整奖励分量，圈时为 45.55、45.56、45.55 s。

参考：[Gymnasium Env](https://gymnasium.farama.org/api/env/)、
[SB3 自定义环境与 checker](https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html)、
[Residual Reinforcement Learning](https://residualrl.github.io/)。
