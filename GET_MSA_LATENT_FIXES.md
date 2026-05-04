# get_msa_latent.py 修复总结

## 修复内容

### 1️⃣ 修复张量维度错误（Permute Bug）

**问题**：
```python
# 原代码的错误处理
z_local = z_local.permute(1, 0)  # 错误的维度操作
```

**原因**：
- `z_local` 的形状为 `(batch_size, T', latent_dim)` 
- 使用 `permute(1, 0)` 在 3D 张量上只会交换前两个维度
- 这导致形状变为 `(T', batch_size, latent_dim)`，无法与 `reference_end_latent` 进行正确的 cat 操作

**修复方案**：
```python
# 对于 reference_end_latent（bs=1 的特殊情况）
reference_end_latent = z_local.squeeze(0)  # (1, 1, latent_dim) → (1, latent_dim)

# 对于主循环中的 z_local_i
z_local_i = z_local[i:i+1]  # (1, T', latent_dim)
z_local_i = z_local_i.squeeze(0)  # (1, T', latent_dim) → (T', latent_dim)
```

**结果**：
✅ 维度一致，可正确执行 `torch.cat([z_local_i, reference_end_latent], dim=0)`

---

### 2️⃣ 修复 Batch Size 遍历问题

**问题**：
```python
# 原代码的强制假设
save_path = pjoin(args.latent_dir, name[0] + '.npy')  # 只处理第一个样本
np.save(save_path, latent_np)  # batch 中其他样本被忽略或覆盖
```

**原因**：
- 代码假设 `batch_size == 1`，只处理 `name[0]`
- 实际 DataLoader 可能返回多个样本
- 导致数据丢失和文件覆盖

**修复方案**：
```python
# 新增内层循环，逐个处理 batch 中的每个样本
for i in range(bs):
    z_local_i = z_local[i:i+1]  # 提取第 i 个样本的特征
    sample_name = name[i] if isinstance(name, (list, tuple)) else name
    latent_save_path = pjoin(args.latent_dir, sample_name + '.npy')
    np.save(latent_save_path, latent_np)
    processed_count += 1  # 样本计数

# 错误统计也要正确
except Exception as e:
    skipped_count += bs  # 跳过整个 batch
```

**结果**：
✅ 完整支持 batch 处理（bs > 1）
✅ 每个样本独立保存，不再丢失或覆盖

---

### 3️⃣ 分离保存局部特征 (z_local) 与全局语义特征 (h_cls)

**问题**：
- 原代码只保存 `z_local`（局部物理特征）
- 忽略了 `h_cls`（全局语义特征）
- 生成阶段无法进行语义检索引导

**方案**：
```python
# 初始化两个输出目录
args.h_cls_dir = pjoin('./humanml3d_272/h_cls_latents_msa_vae', args.exp_name)

# 在循环中分别保存两种特征
# 1. 保存 z_local（局部特征）
latent_save_path = pjoin(args.latent_dir, sample_name + '.npy')
np.save(latent_save_path, latent_np)

# 2. 保存 h_cls（全局语义特征）
h_cls_i = h_cls[i].cpu().detach().numpy()
h_cls_save_path = pjoin(args.h_cls_dir, sample_name + '.npy')
np.save(h_cls_save_path, h_cls_i)
```

**输出目录结构**：
```
./humanml3d_272/
├── t2m_latents_msa_vae/
│   └── <exp_name>/
│       ├── 000000.npy          ← z_local（局部特征）+ reference
│       ├── 000002.npy
│       └── ...
└── h_cls_latents_msa_vae/
    └── <exp_name>/
        ├── 000000.npy          ← h_cls（全局语义特征）
        ├── 000002.npy
        └── ...
```

**结果**：
✅ 分离存储局部和全局特征
✅ 后续生成模型可以加载 h_cls 进行检索引导

---

## 验证清单

- [x] 语法检查通过 (`python -m py_compile`)
- [x] 张量维度处理正确（squeeze 而非 permute）
- [x] Batch 循环完整（for i in range(bs)）
- [x] h_cls_dir 初始化和创建
- [x] 双输出目录：latent_dir 和 h_cls_dir
- [x] 日志更新反映两种特征的保存位置

---

## 使用示例

```bash
# 运行脚本
python get_msa_latent.py \
    --dataname t2m_272 \
    --resume_pth ./Experiments/MSA_VAE_t2m_272/model.pth \
    --exp_name extract_v1

# 查看输出
ls -lh ./humanml3d_272/t2m_latents_msa_vae/extract_v1/ | head -10
ls -lh ./humanml3d_272/h_cls_latents_msa_vae/extract_v1/ | head -10

# 验证特征维度
python -c "import numpy as np; z = np.load('./humanml3d_272/t2m_latents_msa_vae/extract_v1/000000.npy'); h = np.load('./humanml3d_272/h_cls_latents_msa_vae/extract_v1/000000.npy'); print(f'z_local: {z.shape}, h_cls: {h.shape}')"
```

---

**修复完成日期**: 2026-03-18
**验证环境**: Windows VSCode Remote-SSH → Linux Server
