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

## Phase 6 阶段总结

### 各 loss 实测对比

| 指标 | Phase 5 (MSE) | Phase 6b (L1, λ=1.0) | Phase 6c (TotalBppLoss) |
|---|---|---|---|
| `psnr_gray` | ~32 dB | **40.04 dB** | 36.37 dB |
| `bpp_main` | 0.190 | 0.218 | 0.283 |
| `bpp_residual` | 4.085 | **3.314** | 3.562 |
| `bpp_total` | 4.275 | **3.532** | 3.844 |
| 残差占比 | 95.6% | 93.8% | 92.7% |

### 关键结论

1. **L1 loss**：相比 Phase 5 的 MSE，bpp_total 下降 17.4%。
   TotalBppLoss（Phase 6c）反而全面回退，Laplacian i.i.d. 熵公式不够准确（实际残差有空间
   相关性，zstd 利用了这些相关性，i.i.d. 模型高估了真实熵）。但是依然继续选择使用TotalBppLoss, 
   这样可以保证未来模型改造以后依然可以用这套loss来做评估.

2. **残差压缩已接近理论极限**：Phase 6b 的 PSNR 40.04 dB → MAE ≈ 1.8，
   Laplacian i.i.d. 理论熵 ≈ 3.27 bpp，实测 bpp_residual = 3.314，
   zstd 已经把残差压到距离理论下界仅 0.04 bpp。

3. **继续改 loss 收益极其有限**：在当前重建质量（MAE）水平下，残差压缩已无空间。
   **降低 bpp_total 的唯一出路是降低 MAE 本身**——让网络产生更精确的重建。

4. **不同图像表现差异巨大**：bpp_total 从 2.90（PSNR 42+ dB）到 4.32（PSNR 31 dB），
   说明当前网络对某些纹理模式适配不足，网络容量未被有效利用。

---

## Phase 7: 模型与 loss 持续改进

### 背景与动机

Phase 6 的结论表明，当前瓶颈是**网络重建精度不够**：

- `bpp_residual` 占 `bpp_total` 的 94%，且残差压缩已接近理论极限
- 降低 `bpp_total` 的唯一途径是降低 MAE（让重建更准确）→ 需要更强的网络

当前 `NeutronStar2026` 直接复制自 ELIC 2022（`Elic2022Official`），存在三个结构性问题：

**问题 1："注意力"不是真正的自注意力。**
`AttentionBlock` 是 Cheng2020 的门控机制（`a ⊙ σ(b) + identity`），内部只有 3×3 卷积，
感受野局限于局部。对 1024×1024 灰度扫描图的长程空间相关性完全无法建模。

**问题 2：编解码器之间没有跳跃连接。**
解码器是纯顺序 `nn.Sequential`，所有空间细节必须从 64×64 的潜在空间恢复。
高频纹理在 16× 下采样中严重丢失，这直接导致残差大。

**问题 3：潜在空间 channel groups 为 RGB 设计。**
`groups = [16, 16, 32, 64, 192]` 来自 ELIC 原始 RGB 设计。灰度信息量约为 RGB 的 1/3，
但分组策略完全不变。最后一组 192 通道中大量通道接近零信息。

### 子阶段拆分策略

Phase 7 拆成 4 个子阶段，每个子阶段改进一个独立方面，可以独立训练和评测。
如果某个子阶段效果不佳可以回退，不影响其他改进。

---

## Phase 7a: 引入 Window-based Self-Attention

### 目标
将 `AttentionBlock`（门控机制，3×3 局部感受野）替换为 **Window-based Multi-Head
Self-Attention (W-MSA)**，赋予模型真正的全局/半全局建模能力。

### 为什么这是最高优先级

当前 `AttentionBlock` 的本质是：

```python
def forward(self, x):
    a = self.conv_a(x)     # 3 个 ResidualUnit，每个只有 3×3 conv
    b = self.conv_b(x)     # 3 个 ResidualUnit + 1×1 conv
    return a * sigmoid(b) + x   # 逐元素门控
```

这只是一个局部非线性变换，**不是注意力机制**。近年来 SOTA 图像压缩模型
（STF, LIC-TCM, ELIC-SM）都已用 Window/Swin attention 替代。对于灰度扫描图像的
大尺度重复纹理和长程相关性，真正的自注意力收益尤其大。

### 设计

新增 `WindowAttentionBlock` 替换编解码器中的 `AttentionBlock`：

```python
class WindowAttentionBlock(nn.Module):
    """Window-based multi-head self-attention with optional shifted windows.

    Applies W-MSA within non-overlapping windows of size window_size × window_size,
    alternating with SW-MSA (shifted by window_size // 2) for cross-window
    information flow. Uses relative position bias.
    """

    def __init__(self, dim, num_heads=8, window_size=8, shift=False):
        ...

    def forward(self, x):
        # x: (B, C, H, W) — 与 AttentionBlock 接口一致
        # 1. 将特征图分割为不重叠的 window_size × window_size 窗口
        # 2. 在每个窗口内做标准多头自注意力
        # 3. 合并窗口，恢复原始空间分辨率
        # 4. 残差连接
        ...
```

替换位置（编码器和解码器各 2 处）：

```
Encoder g_a:
  ... 3× ResBlock ...
  AttentionBlock(N)       →  WindowAttentionBlock(N, shift=False)
  ... 3× ResBlock ...
  AttentionBlock(M)       →  WindowAttentionBlock(M, shift=True)

Decoder g_s:
  AttentionBlock(M)       →  WindowAttentionBlock(M, shift=True)
  ... 3× ResBlock ...
  AttentionBlock(N)       →  WindowAttentionBlock(N, shift=False)
  ... 3× ResBlock ...
```

### 范围
- 在 `compressai/layers/` 中新增 `WindowAttentionBlock` 类。
- 修改 `compressai/models/neutronstar.py` 中 `NeutronStar2026` 的 `g_a` 和 `g_s`，
  用 `WindowAttentionBlock` 替换 `AttentionBlock`。
- `compressai/layers/__init__.py` 导出新类。
- 不改变模型注册名、接口签名和 latent codec。

### 交付物
- `WindowAttentionBlock` 类。
- 更新后的 `NeutronStar2026`（使用新注意力）。
- 用 L1 loss（Phase 6b 最优 criterion）从头训练的 checkpoint。

### 可验证结果
- `forward`、`compress`、`decompress` 链路无报错。
- `lossless_restored == true`（无损恢复仍然成功）。
- 与 Phase 6b 对比：

| 指标 | Phase 6b (旧 Attention) | Phase 7a (W-MSA) | 期望方向 |
|---|---|---|---|
| `bpp_main` | 0.218 | — | 可能略↑ |
| `bpp_residual` | 3.314 | — | **↓ 主判据** |
| `bpp_total` | 3.532 | — | **↓** |
| `psnr_gray` | 40.04 | — | ↑ |

### 注意事项
- `window_size` 必须能整除特征图尺寸。对于 1024×1024 输入，经 2 次 stride-2 后
  为 256×256（第一个 AttentionBlock 位置），`window_size=8` 可以整除。
  经 4 次 stride-2 后为 64×64（第二个 AttentionBlock 位置），同样可以整除。
- 如果输入图不是 `window_size` 的整倍数，需要加 padding 逻辑。
- 参数量会增加（attention 的 QKV 投影），但主要增长在 self-attention 层，
  相对于 9 个 ResBlock 的总参数量占比不大。
- 建议先用小 epoch 验证训练稳定性，再做完整训练。

### 复杂度控制
- 预计改动文件数：`3`（新增 attention 层 + 修改模型 + 更新 `__init__.py`）
- 适合作为单次 agent 任务

### Phase 7a 实测结果

| 指标 | Phase 6b (旧 Attention, L1) | Phase 7a (W-MSA, TotalBppLoss) | 变化 |
|---|---|---|---|
| `psnr_gray` | 40.04 dB | **42.35 dB** | +2.3 dB |
| `bpp_main` | 0.218 | **0.139** | -36% |
| `bpp_residual` | 3.314 | **3.135** | -5.4% |
| `bpp_total` | 3.532 | **3.274** | **-7.3%** |

W-MSA 在所有指标上全面优于旧的 conv-based gating 机制。但训练 loss（~1.8）
与实测 bpp_total（3.274）之间存在 1.8× 偏差，说明 `TotalBppLoss` 的 Laplacian
i.i.d. 熵估计不够准确，这直接推动了 Phase 7b 的 loss 改进。

---

## Phase 7b: 可微分零阶经验熵替换 TotalBppLoss

### 背景与动机

Phase 7a 的 benchmark 揭示了 `TotalBppLoss` 的关键缺陷：训练 loss 与实测
`bpp_total` 之间存在巨大偏差。

| 指标 | 训练 loss 中的值 | Benchmark 实测值 | 偏差倍数 |
|---|---|---|---|
| `bpp_neural` (理论熵) vs `bpp_main` (实际码流) | ~0.09 | 0.139 | 1.5× |
| `bpp_residual_est` (Laplacian) vs `bpp_residual` (zstd) | ~1.72 | 3.135 | 1.8× |
| `loss` vs `bpp_total` | ~1.81 | 3.274 | 1.8× |

`bpp_residual_est` 的偏差是主要来源。当前公式 `log₂(2e × MAE_uint8)` 做了
两个不成立的假设：

1. **连续分布假设**：实际残差是整数值（-20, -1, 0, 1, 2, ...）。当 MAE 接近 1
   时，连续 Laplacian 熵公式严重低估离散分布的熵。
2. **全局同分布假设**：一个全局 MAE 代表所有像素，但不同区域残差分布差异巨大
   （纹理区 MAE 大，平坦区 MAE ≈ 0），全局平均后信息被抹掉。

实际的 zstd 压缩做的是接近**零阶经验熵**（每个整数值的出现频率决定编码长度），
所以最直接的改进是：**在 loss 中直接估计零阶经验熵**。

### 核心理论

零阶经验熵定义为：

```
H₀ = -Σ p(k) × log₂(p(k))
```

其中 p(k) 是残差值 k 在整幅图中出现的概率（频率）。这比 Laplacian 公式更准确：

| 特性 | Laplacian 公式 `log₂(2eb)` | 零阶经验熵 `H₀` |
|---|---|---|
| 分布假设 | 假设 Laplace 分布 | 无分布假设，直接从数据估计 |
| 离散/连续 | 连续近似 | 自然处理离散值 |
| 多模态/偏态 | 无法捕捉 | 自动捕捉 |
| 稀疏性（大量 0） | 低估 0 值的编码收益 | 正确反映 0 值的高概率 |
| 与 zstd 的关系 | 粗略下界 | 接近 zstd 实际表现（≈ FSE/Huffman 部分） |

### 难点：可微性

直方图（histogram）本身不可微——离散化操作会切断梯度。解决方案：
**软直方图（soft histogram）**，用高斯核将每个残差值平滑地分配到相邻 bin，
使梯度可以通过。

### 设计

在 `compressai/losses/rate_distortion.py` 中新增 `SoftEntropyBppLoss` 类，
替换 `TotalBppLoss`：

```python
class SoftEntropyBppLoss(nn.Module):
    """Minimize total BPP = neural BPP + differentiable zero-order entropy
    of the quantized residual.

    Uses a Gaussian-kernel soft histogram to estimate the empirical PMF of
    integer residual values, then computes H = -Σ p log₂(p).  Both terms
    are in bpp units, so no lambda is needed.
    """

    def __init__(self, bin_range=20, sigma=0.5):
        super().__init__()
        self.bin_range = bin_range   # cover residuals in [-R, R]
        self.sigma = sigma

    def _soft_zero_order_entropy(self, residual_uint8):
        """Differentiable zero-order entropy of integer-valued residuals.

        Memory-efficient: loops over 2R+1 bins with O(N) per iteration
        instead of materializing an O(N × bins) tensor.
        """
        R = self.bin_range
        r = residual_uint8.reshape(-1)
        bins = torch.arange(-R, R + 1, dtype=r.dtype, device=r.device)
        inv_2sigma2 = -1.0 / (2 * self.sigma ** 2)

        log_counts = torch.stack([
            torch.logsumexp((r - b).square() * inv_2sigma2, dim=0)
            for b in bins
        ])

        log_pmf = log_counts - torch.logsumexp(log_counts, dim=0)
        pmf = log_pmf.exp()
        entropy = -(pmf * log_pmf).sum() / math.log(2)
        return entropy

    def forward(self, output, target):
        N, _, H, W = target.size()
        num_pixels = N * H * W

        bpp_loss = sum(
            (torch.log(lk).sum() / (-math.log(2) * num_pixels))
            for lk in output["likelihoods"].values()
        )

        x_hat = output["x_hat"]
        x_hat_q = (x_hat * 255).round() / 255
        x_hat_q = x_hat + (x_hat_q - x_hat).detach()   # STE

        residual_uint8 = (target - x_hat_q) * 255

        bpp_residual_est = self._soft_zero_order_entropy(residual_uint8)

        loss = bpp_loss + bpp_residual_est

        mae = residual_uint8.abs().mean()
        return {
            "loss": loss,
            "bpp_loss": bpp_loss,
            "bpp_residual_est": bpp_residual_est,
            "l1_loss": mae / 255,
            "mse_loss": F.mse_loss(x_hat, target),
        }
```

### 关键参数

- **`bin_range`**（默认 20）：覆盖残差 [-R, R] 的整数范围。Phase 7a 的 PSNR 42 dB
  对应 MAE ≈ 0.7 uint8，残差绝大部分在 [-5, 5] 内，R=20 已足够覆盖尾部。
  R 越小循环越少越快。
- **`sigma`**（默认 0.5）：高斯核宽度。sigma=0.5 使相邻整数 bin 有适度重叠，
  保证梯度平滑。太小（< 0.3）梯度接近 STE 不稳定，太大（> 1.0）模糊分布形状。

### 性能开销分析

- 循环次数：`2 × bin_range + 1 = 41` 次
- 每次循环：一个逐元素平方 + `logsumexp`，都是 O(N) 的 GPU 操作
- 内存：O(N) 常驻 + O(41) 的 log_counts 向量，**不需要** O(N × bins) 的大矩阵
- 对于 256×256 patch、batch=32：N = 2M 像素，41 次简单向量操作，
  预计增加 **5-15%** 的训练时间

### 范围
- 在 `compressai/losses/rate_distortion.py` 中新增 `SoftEntropyBppLoss` 类。
- 注册到 criterion registry。
- 在 `compressai/losses/__init__.py` 中导出。
- 修改 `scripts/train_neutronstar.py` 使用新 loss（替换 `TotalBppLoss`）。
- **不改模型结构**，模型仍为 Phase 7a 的 W-MSA 版 `NeutronStar2026`。
- 需要先回退 Phase 7b（skip connections）对 `neutronstar.py` 的改动，
  恢复到 Phase 7a 的纯 W-MSA 架构。

### 交付物
- `SoftEntropyBppLoss` 类。
- 更新后的 `compressai/losses/__init__.py`。
- 更新后的 `train_neutronstar.py`。
- 用新 loss 从头训练的 checkpoint。
- 与 Phase 7a（TotalBppLoss）的 benchmark 对比。

### 可验证结果
- `SoftEntropyBppLoss` 可被 import 和实例化。
- `forward(output, target)` 返回字典包含
  `{"loss", "bpp_loss", "bpp_residual_est", "l1_loss", "mse_loss"}`。
- 训练过程中 loss 持续下降。
- `lossless_restored == true`。
- 与 Phase 7a 对比：

| 指标 | Phase 7a (TotalBppLoss) | Phase 7b (SoftEntropyBppLoss) | 期望 |
|---|---|---|---|
| `bpp_main` | 0.139 | — | 可能略变 |
| `bpp_residual` | 3.135 | — | **↓ 主判据** |
| `bpp_total` | 3.274 | — | **↓** |
| `psnr_gray` | 42.35 | — | 可能略变 |
| loss 与 bpp_total 的偏差 | 1.8× | — | **↓ 接近 1.0×** |

### 决策点
- **如果 bpp_total 下降且 loss-bpp_total 偏差缩小**：新 loss 有效，
  说明更准确的梯度信号让优化器找到了更好的操作点。继续使用。
- **如果 bpp_total 持平**：说明 Laplacian 公式虽然不准确，但梯度方向
  足够好，更准确的 H₀ 并没有改变最优点。可以考虑调整 `sigma` 或 `bin_range`。
- **如果训练不稳定**：`sigma` 过小导致梯度尖锐。增大 `sigma` 或使用
  warmup 策略（前 N epoch 用 TotalBppLoss，之后切换到 SoftEntropyBppLoss）。

### 复杂度控制
- 预计改动文件数：`3`（`rate_distortion.py` + `__init__.py` + `train_neutronstar.py`）
  + `1`（回退 `neutronstar.py` 中的 skip connection 改动）
- 适合作为单次 agent 任务

---

## Phase 7c: 潜在空间 channel groups 重新设计

### 目标
重新设计潜在空间的 channel groups 分组策略，使其适配灰度图像的信息分布，
提升上下文模型效率。

### 当前问题

当前 `groups = [16, 16, 32, 64, 192]`（sum = M = 320）来自 ELIC 为 RGB 设计的原始配置。

灰度单通道图像的信息量约为 RGB 的 1/3，但：
- 最后一组 192 通道过大，大量通道接近零信息，上下文模型为它们浪费了计算
- 前两组各 16 通道过小，承载信息不足，限制了后续组的上下文质量
- bpp_main 仅 0.22（320 通道中实际只有约 0.22 × 64² = 900 bits 有效信息），
  说明通道利用率极低

### 设计

**策略：减小 M，重新分配 groups，增多分组数量**

候选方案（需要实验对比）：

| 方案 | M | groups | 分组数 | 说明 |
|---|---|---|---|---|
| 当前 | 320 | [16, 16, 32, 64, 192] | 5 | ELIC 原始 RGB |
| A | 192 | [16, 16, 32, 32, 32, 32, 32] | 7 | 更均匀、更多组 |
| B | 256 | [16, 16, 32, 32, 64, 96] | 6 | 适度减小、更均匀 |
| C | 192 | [24, 24, 24, 24, 48, 48] | 6 | 灰度定制 |

选择原则：
- `sum(groups) == M`
- 分组更均匀，避免最后一组过大
- 更多的组数 → 每组有更多上下文历史 → 上下文模型更精准
- M 不宜太小，否则信息瓶颈太紧

### 范围
- 修改 `NeutronStar2026.__init__` 中的 `groups` 默认值和 M 默认值。
- 上下文模型（`channel_context`、`spatial_context`、`param_aggregation`）
  会自动适配新的 groups（它们已经用循环生成），无需额外改代码。
- 超先验 h_a / h_s 的通道数可能需要随 M 调整。

### 交付物
- 调整 groups/M 后的 `NeutronStar2026`。
- 至少 2 个方案的训练 checkpoint。
- 各方案的 benchmark 对比。

### 可验证结果
- 训练链路正常。
- `lossless_restored == true`。
- 对比不同 groups 方案的 `bpp_main`、`bpp_residual`、`bpp_total`。
- 选出 `bpp_total` 最低的方案。

### 注意事项
- 改变 M 后旧 checkpoint 不再兼容，必须从头训练。
- 建议先用小 epoch（如 20）快速筛选方案，再对最优方案做完整训练。
- 如果 Phase 7a 和 7b 的改动已经稳定，7c 应在它们之上做（累积改进）。

### 复杂度控制
- 预计改动文件数：`1`（`neutronstar.py`）
- 适合作为单次 agent 任务（每个方案独立训练）

---

## Phase 7d: 网络架构改造（待规划）

### 目标
探索网络结构层面的改进，降低重建误差（MAE），从而降低 `bpp_residual`。

### 待讨论方向
- 编解码器跳跃连接（需解决推理时跳跃特征不可用的问题，
  方案包括：将跳跃特征编码到码流中作为多尺度超先验、
  或仅在解码器内部添加 DenseNet 风格的跳连）
- 解码器容量增强（更深/更宽的解码器）
- 多尺度潜在空间
- 其他 SOTA 图像压缩架构的设计元素

### 前置条件
- Phase 7b（loss 改进）和 Phase 7c（channel groups）的结果已稳定
- 在改进后的 loss 和 groups 基础上做架构探索，确保变量控制

### 复杂度控制
- 待 Phase 7b/7c 完成后根据 benchmark 数据具体规划

---

## 建议的实施顺序总结

### 已完成阶段

1. ~~Phase 1：脚本框架与 benchmark 基线~~
2. ~~Phase 2：`neutronStar2026` 等价复制~~
3. ~~Phase 3：单通道灰度原生支持~~
4. ~~Phase 4：灰度有损训练基线~~
5. ~~Phase 5：残差无损恢复基线~~
6. ~~Phase 6：residual-aware loss 优化总码率~~
   - ~~6a. 实现 `ResidualAwareRDLoss`（L1 替换 MSE）~~
   - ~~6b. 接入训练脚本，完成首次 L1 训练~~
   - ~~6c. 实现 `TotalBppLoss`~~

### 进行中

7. Phase 7：模型与 loss 持续改进
   - ~~7a. 引入 Window-based Self-Attention — 替换 4 处 AttentionBlock，3 文件~~
   - 7b. 可微分零阶经验熵替换 TotalBppLoss — 新增 SoftEntropyBppLoss，3 文件
   - 7c. 潜在空间 channel groups 重新设计 — 调整 M 和 groups，1 文件
   - 7d. 网络架构改造（待规划）

### 后续可选方向（待 Phase 7 结果确定后规划）

- 端到端压缩流程工程化（统一压缩文件格式）
- 性能优化（编解码速度、显存）
- 训练流程优化（数据增强、学习率调度、更大 patch）
- 学习式残差编码（如果 zstd 成为新瓶颈）

## 每阶段的验收模板

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
