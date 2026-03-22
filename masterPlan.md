# CompressAI Fork Master Plan

## 目标
面向 `8bit` 单通道灰度 `BMP` 图像，基于 `compressai` 演进出一个可持续迭代的压缩系统：

1. 第一阶段先得到可训练、可压缩、可解压、可 benchmark 的灰度有损模型。
2. 第二阶段在有损主码流之上增加残差支路，使系统具备"从压缩文件还原出无损原图"的能力。
3. 第三阶段再围绕总压缩率、编码速度、解码速度、工程可维护性做优化。

## 对当前 7 步计划的调整
当前思路的大方向是对的，但顺序需要调整，否则会把 4 类问题混在一起：

1. 数据格式问题：当前训练和评测链路默认是 RGB，不先拆出来，后面 benchmark 会失真。
2. 模型改造问题：`elic2022-official` 复制成 `neutronStar2026` 是合理的，但要先做"等价克隆"，再做灰度改造。
3. 系统设计问题：残差无损恢复要先做一个独立、可验证的基线，再去修改 loss；否则无法知道 loss 是否真的降低了总码率。
4. 工程组织问题：`benchmark.py` 应该尽早建立，但一开始只做"调度和记录"，不要一上来就承载太多编解码逻辑。

结论：

- `scripts/` 文件夹应该尽早建立。
- `neutronStar2026` 应该先做"行为等价复制"，再做灰度化。
- "残差算法"必须拆成两个阶段：
  - 先做无损恢复链路基线。
  - 再做 residual-aware loss 优化总码率。
- 性能优化必须放在功能和 benchmark 稳定之后。

## 规划原则
每个阶段都必须满足以下约束：

1. 阶段目标单一，不同时解决"模型结构 + 数据格式 + 残差设计 + 性能优化"四件事。
2. 阶段完成后必须能跑 benchmark，并生成可归档结果。
3. 每个阶段的改动范围尽量控制在少量文件内，适合一次 agent 实施，目标是单阶段实现和验证成本控制在 `500k token` 以内。
4. 优先复用现有 `compressai` 能力，避免过早重写训练、评测、熵编码基础设施。
5. 所有 benchmark 都要有固定输入集、固定 checkpoint、固定输出目录，确保横向可比。

## 建议目录规划
建议尽早统一工程入口，避免后续脚本分散：

```text
scripts/
  train_neutronstar.py
  compress_neutronstar.py
  decompress_neutronstar.py
  benchmark.py
  helpers/

artifacts/
  checkpoints/
  bitstreams/
  recon/
  benchmarks/
  logs/
```

说明：

- `scripts/benchmark.py` 负责调度，不负责承载所有核心算法。
- 训练、压缩、解压分别保留独立脚本，便于单独调试。
- benchmark 输出统一落到 `artifacts/benchmarks/`，后续阶段直接复用。

## 统一 Benchmark 约定
从第一阶段开始就统一结果格式，后面每一阶段只扩展字段，不随意改口径。

### 必备指标
- `model_name`
- `phase`
- `dataset_name`
- `checkpoint`
- `num_images`
- `bpp_main`
- `psnr_gray`
- `encode_time`
- `decode_time`
- `status`

### 残差阶段新增指标
- `residual_codec`
- `bpp_residual`
- `bpp_total`
- `lossless_restored`
- `max_abs_error`

### Benchmark 判定原则
- 有损阶段：至少要能比较 `bpp_main` 和 `psnr_gray`。
- 无损恢复阶段：必须比较 `bpp_total`，并验证 `lossless_restored == true`。
- 优化阶段：必须比较速度、显存或吞吐，不只看 PSNR。

## 分阶段实施计划

## Phase 1: 建立脚本框架与 Benchmark 基线
### 目标
先建立稳定的实验入口和结果记录方式，让后续每个阶段都能被量化验证。

### 范围
- 新建 `scripts/` 目录。
- 新建 `scripts/benchmark.py`。
- 新建训练、压缩、解压的最小包装脚本。
- 先复用现有 `compressai` 训练和评测逻辑，不重写底层算法。

### 为什么先做这个阶段
你的原计划里把 `benchmark.py` 放在第 3 步是对的，但建议和脚本目录一起作为第一阶段完成。否则后续每个改动都缺少统一验证入口。

### 交付物
- `scripts/` 目录结构成型。
- `benchmark.py` 可以调度至少一种现有模型的训练后评测或 checkpoint 评测。
- benchmark 结果能落盘成 `json` 或 `csv`。

### 可验证结果
- 在固定测试集上成功生成一份 benchmark 结果。
- 同一命令重复执行两次，输出结构一致。
- 不要求此阶段就支持灰度原生指标，可以先跑通最小链路。

### 建议 benchmark
- 基线模型：先用仓库现成、最容易走通链路的模型做 smoke benchmark。
- 输出至少包含：`bpp_main`、重建质量、编码时间、解码时间。

### 复杂度控制
- 预计改动文件数：`4-6`
- 适合作为单次 agent 任务

## Phase 2: 复制 `elic2022-official` 为 `neutronStar2026`
### 目标
把后续实验主线从上游模型中解耦出来，建立自己的模型命名、注册和加载入口。

### 范围
- 复制 `elic2022-official` 为 `neutronStar2026`。
- 保持初始行为与原始 `elic2022-official` 尽量一致。
- 接入模型注册和脚本可调用入口。

### 为什么这是独立阶段
如果直接在原始 `elic2022-official` 上改，后续很难区分"是灰度化引入的问题"还是"是你自定义结构引入的问题"。

### 交付物
- `neutronStar2026` 模型类。
- 对应的注册名和加载入口。
- 最小 smoke test：forward、compress、decompress 可以运行。

### 可验证结果
- 在相同输入上，`neutronStar2026` 和原始 `elic2022-official` 都能成功推理。
- benchmark 能记录新模型名称。
- 此阶段不追求指标提升，只追求功能等价和链路可用。

### 建议 benchmark
- 同一张或同一批图像上，对比 `elic2022-official` 与 `neutronStar2026` 的：
  - 是否成功编码和解码
  - `bpp_main`
  - 基本重建质量

### 复杂度控制
- 预计改动文件数：`3-6`
- 适合作为单次 agent 任务

## Phase 3: 灰度数据链路原生化
### 目标
让数据集、模型输入输出、评测指标都真正面向单通道灰度图，而不是把灰度图伪装成 RGB。

### 范围
- 数据集读取支持单通道灰度。
- `neutronStar2026` 首层和末层改为 `1` 通道。
- 评测链路新增灰度图指标，例如 `psnr_gray`。
- 压缩和解压脚本支持灰度图输入输出。

### 为什么这一阶段必须独立
这是整个项目后续正确性的基础。如果继续沿用 RGB 默认流程，即使 benchmark 能跑，结果也会混入无意义的 3 通道重复信息。

### 交付物
- 单通道灰度版 `neutronStar2026`。
- 灰度图数据加载与推理链路。
- benchmark 可按灰度图口径出结果。

### 可验证结果
- 输入 `8bit` 灰度 `BMP`，压缩与解压链路无报错。
- 重建输出仍是单通道灰度。
- benchmark 输出 `psnr_gray` 与 `bpp_main`。

### 建议 benchmark
- 评测集固定为你的灰度 `BMP` 测试集。
- 对比对象：
  - Phase 2 的伪 RGB 路线
  - Phase 3 的原生灰度路线

### 通过标准
- 原生灰度链路完成。
- benchmark 输出稳定。
- 结果可用于后续所有阶段直接对比。

### 复杂度控制
- 预计改动文件数：`5-8`
- 适合作为单次 agent 任务

## Phase 4: 灰度有损训练基线
### 目标
先拿到一个真正服务你场景的灰度有损主码流 baseline。

### 范围
- 建立 `train_neutronstar.py` 的可训练版本。
- 支持从头训练或从已有 checkpoint 恢复训练。
- 固定一套小规模训练配置和测试配置。

### 为什么在这里训练
只有在"模型身份稳定 + 灰度链路稳定"后，训练结果才有比较意义。

### 交付物
- 可复现的训练命令。
- 可复现的测试命令。
- 至少一个基线 checkpoint。

### 可验证结果
- 在固定 epoch 或固定 step 后，benchmark 可以给出：
  - `bpp_main`
  - `psnr_gray`
  - 编码解码耗时
- 相比未训练或随机初始化模型，指标明显改善。

### 建议 benchmark
- 选一个小测试集做快速 benchmark。
- 选一个正式测试集做阶段归档 benchmark。
- 记录训练配置摘要，便于后续对照。

### 复杂度控制
- 预计改动文件数：`3-6`
- 适合作为单次 agent 任务

## Phase 5: 残差无损恢复基线
### 目标
先验证"有损主码流 + 残差支路 = 可无损恢复"这件事在系统层面是否成立。

### 范围
- 定义残差：从原图与有损重建图之间计算残差。
- 设计一种确定性的残差存储和恢复方案。
- 先用简单、可靠、可验证的残差编码方式，不急着做学习式残差模型。

### 为什么不能直接先改 loss
如果没有一个独立的残差基线，你无法知道未来 loss 改动到底是在降低"总码率"，还是只是在牺牲主码流质量。

### 交付物
- 残差生成逻辑。
- 残差恢复逻辑。
- 端到端无损恢复验证。

### 可验证结果
- 对任意测试图像，解码后叠加残差可以完全恢复原图。
- 必须验证：
  - `lossless_restored == true`
  - `max_abs_error == 0`
- benchmark 新增：
  - `bpp_residual`
  - `bpp_total`

### 建议 benchmark
- 与 Phase 4 比较：
  - 仅主码流 `bpp_main`
  - 主码流 + 残差后的 `bpp_total`
- 这一步的 benchmark 目标不是最优，而是建立总码率口径。

### 复杂度控制
- 预计改动文件数：`4-7`
- 适合作为单次 agent 任务

## Phase 6: residual-aware loss 设计与训练

### 背景与动机

Phase 5 的 benchmark 数据揭示了关键事实：

| 指标 | Phase 5 实测值 | 占 bpp_total |
|---|---|---|
| `bpp_main` | 0.190 | 4.4% |
| `bpp_residual` | 4.085 | 95.6% |
| `bpp_total` | 4.275 | 100% |

残差码率占总码率的 **96%**。当前 Phase 4 的标准 MSE loss 优化的是"让 `x_hat` 尽量好看"，
但我们的真正目标是 **最小化 bpp_total = bpp_main + bpp_residual**（最终输出是无损恢复的原图，
不关心 `x_hat` 的视觉质量）。

### 核心理论依据

残差 `r = x_orig − round(x_hat × 255)` 通常服从 **Laplace 分布**（以 0 为中心，尖峰重尾）。
对于 Laplace(0, b)：

- 尺度参数 b = E[|r|]，即 L1 范数
- 熵 H = log₂(2eb) bits/pixel

因此：

- **MSE (L2) 是 Gaussian 残差模型下的最优估计** → 不匹配实际分布
- **L1 是 Laplacian 残差模型下的最优估计** → 更匹配实际分布

用 Phase 5 数据验证：PSNR~32 dB → RMS error ~ 6.4 像素值 → 对 Laplace 分布 b ≈ 4.5，
理论熵 H = log₂(2e × 4.5) ≈ 4.6 bpp，而实测 `bpp_residual = 4.085`（zstd 利用了空间相关性，
略低于 i.i.d. 理论值）。

### MSE vs L1 对残差压缩的影响

- **MSE** 倾向于把误差"均匀摊开"（所有像素都有小误差）→ 残差中几乎没有零值 → 熵高 → 不利于压缩
- **L1** 倾向于让更多像素误差恰好为 0，代价是少数像素误差稍大 → 残差更稀疏 → 熵低 → 更利于压缩

所以 **将 MSE 替换为 L1 是信息论上合理的第一步改进**。

### 子阶段拆分策略

Phase 6 拆成 4 个子阶段，每个子阶段独立可验证，复杂度控制在单次 agent 任务内。
后一个子阶段以前一个子阶段的结果为基础，如果某个子阶段效果不佳可以停下来分析，
不需要一次性押注在复杂设计上。

---

## Phase 6a: 实现 `ResidualAwareRDLoss`（L1 替换 MSE）

### 目标
新建 `ResidualAwareRDLoss` 类，将 distortion 从 MSE 替换为 L1，作为 bpp_residual 的可微代理。
不改模型结构，不改训练流程，只新增 criterion 类。

### 设计

当前 loss：

```
L_current = λ × 255² × MSE(x_hat, x) + bpp_main
```

新 loss：

```
L_new = λ × 255 × L1(x_hat, x) + bpp_main
```

注意尺度变化：MSE 前系数是 255²，L1 前系数是 255。lambda 的数值意义不同，需要重新校准。

### 范围
- 在 `compressai/losses/rate_distortion.py` 中新增 `ResidualAwareRDLoss` 类。
- 注册到 criterion registry。
- 在 `compressai/losses/__init__.py` 中导出。
- `forward` 返回值中保留 `mse_loss` 字段用于监控对比。

### 交付物
- `ResidualAwareRDLoss` 类（在 `compressai/losses/rate_distortion.py`）。
- `compressai/losses/__init__.py` 更新导出。

### 可验证结果
- 新 loss 类可以被 `import` 和实例化。
- `forward(output, target)` 返回字典包含 `{"loss", "bpp_loss", "mse_loss", "l1_loss", "residual_l1"}`。
- 对同一组 `(output, target)` 数据，`bpp_loss` 与原 `RateDistortionLoss` 一致（验证 bpp 计算未被改动）。

### 复杂度控制
- 预计改动文件数：`2`（`rate_distortion.py` + `__init__.py`）
- 适合作为单次 agent 任务

---

## Phase 6b: 接入训练脚本并完成首次 L1 训练

### 目标
将 `train_neutronstar.py` 改为支持选择 criterion（默认仍为 MSE，可切换到 L1），
用新 loss 训练一个 checkpoint，验证训练链路完整可用。

### 范围
- `train_neutronstar.py` 新增 `--criterion` 参数，可选 `mse`（默认）或 `residual-l1`。
- 日志输出适配新 loss 的字段（打印 `l1_loss` 和 `residual_l1`）。
- 用新 criterion 跑一轮完整训练（epoch 数与 Phase 4 相同），得到 checkpoint。

### 交付物
- 更新后的 `train_neutronstar.py`。
- 一个用 `residual-l1` criterion 训练的 checkpoint。
- 训练日志（保存到 `artifacts/logs/`）。

### 可验证结果
- `python scripts/train_neutronstar.py --criterion residual-l1 ...` 正常启动并完成训练。
- 训练过程中 loss 持续下降。
- 用 `--criterion mse`（默认）训练行为与 Phase 4 完全一致（不破坏回归）。

### lambda 调参建议
- 起始 lambda 值：`0.01`（与 Phase 4 相同数值）。
- 如果 bpp_main > 1.0，减小 lambda。
- 如果 bpp_residual 无明显下降，增大 lambda。
- 目标是找到 bpp_total 的最低点。

### 复杂度控制
- 预计改动文件数：`1`（`train_neutronstar.py`）
- 适合作为单次 agent 任务

---

## Phase 6c: 实现 `TotalBppLoss`（直接最小化 total BPP）

### 目标
设计并实现一个新的 loss 函数，**直接最小化 total BPP = neural BPP + residual BPP**，
用残差的 Laplacian 熵估计替代 L1 代理，消除 lambda 调参的需要。

### 理论依据

Phase 6a/6b 的 L1 loss 是残差熵的**线性代理**：

```
L_l1 = λ × 255 × MAE + bpp_neural
```

但真正要最小化的是 total BPP。残差在 Laplacian 模型下的熵为：

```
H(residual) = log₂(2e × MAE_uint8)   bits/pixel
```

这是 MAE 的**对数**，不是线性关系。用 L1 做代理有两个问题：

1. **需要手动调 lambda**：L1 和 bpp_neural 量纲不同，必须靠 lambda 平衡，最优值需实验搜索。
2. **梯度权重不自适应**：L1 对所有 MAE 水平给予相同梯度。但实际上 MAE 越小时，
   每降低 1 单位 MAE 节省的残差熵越多（∂H/∂b = 1/(b·ln2)），应该推得更用力。

直接用 `log₂(2e × MAE)` 作为 distortion 项，就是在直接最小化残差熵本身：

```
L_total = bpp_neural + log₂(2e × MAE_uint8)
```

两项都是 bpp 单位，**不需要 lambda**，优化器自动找 neural BPP 和 residual BPP 的最优平衡点。

### 与 L1 的梯度对比

| 当前 MAE (uint8) | L1 loss 有效梯度 | log(MAE) loss 有效梯度 |
|---|---|---|
| 5.0 | λ × 255 = 常数 | 1 / (5.0 × ln2) = 0.29 |
| 1.0 | λ × 255 = 常数 | 1 / (1.0 × ln2) = 1.44 |
| 0.5 | λ × 255 = 常数 | 1 / (0.5 × ln2) = 2.89 |
| 0.1 | λ × 255 = 常数 | 1 / (0.1 × ln2) = 14.4 |

log(MAE) 的梯度随 MAE 减小而自动加大，在低误差区间推力更强。

### 设计

```python
class TotalBppLoss(nn.Module):
    """Directly minimize total BPP = neural BPP + estimated residual entropy.

    Uses the Laplacian entropy formula H = log₂(2e·b) where b = MAE as the
    residual BPP estimate. Both terms are in bpp units, so no lambda is needed.
    Includes STE quantization to account for the uint8 rounding at inference.
    """

    def forward(self, output, target):
        N, _, H, W = target.size()
        num_pixels = N * H * W

        # Neural BPP (unchanged)
        bpp_neural = sum(
            torch.log(lk).sum() / (-math.log(2) * num_pixels)
            for lk in output["likelihoods"].values()
        )

        # STE quantization: forward does round(), backward passes through
        x_hat = output["x_hat"]
        x_hat_q = (x_hat * 255).round() / 255
        x_hat_q = x_hat + (x_hat_q - x_hat).detach()

        # Laplacian entropy estimate of quantized residual
        residual_abs = (target - x_hat_q).abs() * 255   # uint8 scale
        mae = residual_abs.mean()                        # Laplacian scale b
        bpp_residual = torch.log2(2 * math.e * (mae + 1e-6))

        # Total BPP — no lambda needed
        loss = bpp_neural + bpp_residual

        return {
            "loss": loss,
            "bpp_loss": bpp_neural,
            "bpp_residual_est": bpp_residual,
            "l1_loss": mae / 255,
            "mse_loss": F.mse_loss(output["x_hat"], target),
        }
```

### 范围
- 在 `compressai/losses/rate_distortion.py` 中新增 `TotalBppLoss` 类。
- 注册到 criterion registry。
- 在 `compressai/losses/__init__.py` 中导出。
- 在 `train_neutronstar.py` 中接入 `--criterion total-bpp` 选项。

### 交付物
- `TotalBppLoss` 类（在 `compressai/losses/rate_distortion.py`）。
- 更新后的 `compressai/losses/__init__.py`。
- 更新后的 `train_neutronstar.py`（支持 `--criterion total-bpp`）。
- 用 `total-bpp` criterion 训练的 checkpoint。
- 与 Phase 6b（L1 + lambda=1.0）的 benchmark 对比结果。

### 可验证结果
- `TotalBppLoss` 可被 import 和实例化。
- `forward(output, target)` 返回字典包含 `{"loss", "bpp_loss", "bpp_residual_est", "l1_loss", "mse_loss"}`。
- 训练过程中 `loss` 持续下降。
- `lossless_restored == true`（无损恢复仍然成功）。
- 与 Phase 6b 对比：

| 指标 | Phase 6b (L1, λ=1.0) | Phase 6c (TotalBppLoss) | 变化方向 |
|---|---|---|---|
| `bpp_main` | — | — | 可能↑（模型选择花更多 bits） |
| `bpp_residual` | — | — | ↓ 目标 |
| `bpp_total` | — | — | **↓ 主判据** |
| `lossless_restored` | true | true | 不变 |

### 决策点
- **如果 `bpp_total` 下降**：TotalBppLoss 有效，找到了比 L1+lambda 更好的操作点。
  可直接进入 Phase 6d 或 Phase 7。
- **如果 `bpp_total` 持平或上升**：检查 `bpp_neural` 和 `bpp_residual_est` 的训练曲线，
  分析是否 Laplacian 假设不够准确或梯度数值不稳定，考虑加入 clamp 或温度系数。

### 注意事项
- `mae + 1e-6` 中的 epsilon 防止 log(0)，但如果训练后期 MAE 极小可能需要调整。
- STE 的 round() 操作在 backward 时梯度直通，可能导致训练初期不稳定，
  建议监控训练曲线。如果不稳定，可以用退火策略：前 N 个 epoch 用 L1，之后切换到 TotalBppLoss。

### 复杂度控制
- 预计改动文件数：`3`（`rate_distortion.py` + `__init__.py` + `train_neutronstar.py`）
- 适合作为单次 agent 任务

---

## Phase 6d: Dead-zone L1 与量化感知改进（条件性）

### 前置条件
Phase 6c 已完成。无论 6c 结果如何，都可以进入 6d 尝试进一步改进。

### 目标
在 L1 基础上加入 **dead-zone**（量化感知）：当连续误差 `|x_hat − x| < 0.5/255` 时，
量化后残差恰好为 0，不需要任何比特。因此这个范围内的误差不应被惩罚。

### 设计

```
L_deadzone = λ × 255 × E[max(|x_hat − x| − 0.5/255, 0)] + bpp_main
```

Dead-zone 释放的模型容量可以让更多像素的残差精确为 0，进一步降低残差熵。

### 范围
- 在 `compressai/losses/rate_distortion.py` 中新增 `DeadZoneResidualLoss` 类。
- `compressai/losses/__init__.py` 中导出。
- `train_neutronstar.py` 的 `--criterion` 新增 `residual-dz` 选项。

### 交付物
- `DeadZoneResidualLoss` 类。
- 用 `residual-dz` 训练的 checkpoint。
- 与 Phase 6c 结果的 benchmark 对比。

### 可验证结果
- `lossless_restored == true`。
- 对比 Phase 6c（L1）和 Phase 5（MSE）：`bpp_total` 是否进一步下降。

### 可选的进一步探索
如果 dead-zone 效果显著，可以考虑进一步加入 STE（Straight-Through Estimator）模拟 uint8 量化：

```python
x_hat_255 = x_hat * 255
x_hat_q = x_hat_255 + (x_hat_255.round() - x_hat_255).detach()  # STE
residual = target_255 - x_hat_q.clamp(0, 255)
```

这属于可选探索，不是 Phase 6d 的硬性交付。

### 复杂度控制
- 预计改动文件数：`2-3`
- 适合作为单次 agent 任务

---

## Phase 7: 残差链路工程化集成
### 目标
把"主码流 + 残差 + 元数据"集成为一个规范化压缩流程，而不是多个临时文件拼接。

### 范围
- 统一压缩文件组织方式。
- 统一解压逻辑。
- 统一 benchmark 输入输出口径。

### 为什么这一阶段单独做
前面的阶段是验证算法与总码率；这一阶段才是把验证过的方案沉淀成稳定工程接口。

### 交付物
- 统一压缩文件格式或统一输出目录规范。
- `compress_neutronstar.py` 与 `decompress_neutronstar.py` 工程化稳定。
- benchmark 可以直接对端到端流程做验证。

### 可验证结果
- 单条命令完成压缩、解压、无损校验。
- benchmark 不再依赖人工拼接中间文件。

### 建议 benchmark
- 正式端到端 benchmark：
  - 成功率
  - `bpp_total`
  - 编码时间
  - 解码时间
  - 无损恢复正确率

### 复杂度控制
- 预计改动文件数：`4-7`
- 适合作为单次 agent 任务

## Phase 8: 性能优化
### 目标
在正确性和总码率稳定后，再做编码速度、解码速度、显存和吞吐优化。

### 范围
- 优化热点路径。
- 减少不必要的数据复制和磁盘中间产物。
- 评估是否需要替换部分脚本组织或批处理方式。

### 为什么最后做
过早做性能优化，容易把未稳定的算法逻辑固化，增加返工成本。

### 交付物
- 优化前后 benchmark 对比。
- 明确的优化收益说明。

### 可验证结果
- 在不破坏正确性的前提下：
  - 编码更快，或
  - 解码更快，或
  - 占用更低

### 建议 benchmark
- 和 Phase 7 的最终稳定版直接对比：
  - `bpp_total`
  - `encode_time`
  - `decode_time`
  - 峰值显存或内存

### 复杂度控制
- 每次只做一类优化。
- 如果优化目标不同，建议拆成多个小子阶段实施。

## 建议的实施顺序总结
最终建议按以下顺序推进：

1. `scripts/` + `benchmark.py` + 最小实验框架
2. 复制 `elic2022-official` 为 `neutronStar2026`
3. 单通道灰度原生支持
4. 灰度有损训练基线
5. 残差无损恢复基线
6. residual-aware loss 优化总码率
   - 6a. 实现 `ResidualAwareRDLoss`（L1 替换 MSE）— 只改 loss 类，2 文件
   - 6b. 接入训练脚本，完成首次 L1 训练 — 只改训练脚本，1 文件
   - 6c. 实现 `TotalBppLoss`（直接最小化 total BPP，消除 lambda）— 3 文件
   - 6d. Dead-zone L1 量化感知改进（条件性）— 2-3 文件
7. 端到端压缩流程工程化
8. 性能优化

## 每阶段的验收模板
后续每个阶段都建议统一使用下面的验收模板：

### 阶段完成判定
- 功能是否可运行
- benchmark 是否可运行
- 结果是否落盘
- 是否能与上一阶段横向比较
- 是否保留回退路径

### 阶段归档物
- 代码版本
- 配置文件
- checkpoint
- benchmark 结果
- 已知问题列表

## 不建议的做法
以下做法应尽量避免：

1. 在还未建立 benchmark 基线前，直接修改 loss。
2. 在还未完成灰度原生支持前，用 RGB 结果长期替代灰度结果。
3. 在残差总码率口径未建立前，过早讨论"最终压缩率是否更优"。
4. 在功能尚未稳定前，提前做大规模性能优化。
5. 让 `benchmark.py` 同时承载训练、压缩、解压、评测全部底层逻辑，导致后续难以维护。

## 第一批推荐实施任务
如果按 agent 小步推进，推荐优先做以下 3 个阶段：

1. Phase 1：脚本框架与 benchmark 基线
2. Phase 2：`neutronStar2026` 等价复制
3. Phase 3：单通道灰度原生支持

这 3 个阶段完成后，项目就会进入"可稳定迭代"的状态，后续训练、残差和 loss 设计都会更顺。
