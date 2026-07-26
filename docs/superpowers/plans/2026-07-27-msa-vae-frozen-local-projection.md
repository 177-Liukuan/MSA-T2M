# MSA-VAE Frozen Local Projection Phase-2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an auditable opt-in Phase-2 mode that freezes the Phase-1 `local_proj` parameters while allowing local-alignment gradients to update the CNN encoder.

**Architecture:** Put reusable phase trainability and optimizer-group construction in `utils/msa_vae_training.py`, then make `train_msa_vae.py` call those helpers after loading the Phase-1 checkpoint. Expose the behavior through an opt-in CLI flag and a default-off HumanML launcher environment variable, and record the resolved boolean in checkpoint metadata.

**Tech Stack:** Python 3.8.11, PyTorch 2.4.1, `unittest`, Bash, Accelerate 1.0.1.

## Global Constraints

- Use the existing `mgpt` Conda environment; do not upgrade dependencies.
- Preserve the 272-D HumanML representation, causal 16-D latent convention, model tensor names, and checkpoint state-dict shapes.
- The new behavior is opt-in and defaults to disabled in direct Python and root-shell invocations.
- Do not change BABEL launchers or the completed 2026-07-26 pilot contract.
- Do not add a detach, `torch.no_grad()`, copied latent, or second projection to the local-alignment path.
- Do not start full training, evaluation, or GPU jobs while validating this implementation.
- Preserve unrelated untracked files and third-party worktrees.

---

### Task 1: Phase trainability and optimizer helpers

**Files:**
- Modify: `utils/msa_vae_training.py`
- Modify: `tests/test_msa_vae_training_entrypoint.py`

**Interfaces:**
- Consumes: an `nn.Module` exposing either `model.msa_vae` or the MSA core directly, with `cnn_encoder`, `cnn_decoder`, `decode_proj`, and `local_proj` child modules.
- Produces: `configure_msa_vae_trainability(model, phase, freeze_phase2_local_proj=False) -> dict[str, bool]`.
- Produces: `build_phase2_optimizer_param_groups(model, lr, cnn_lr_scale) -> list[dict[str, object]]`.
- Produces: checkpoint metadata field `training_args.freeze_phase2_local_proj: bool`.

- [ ] **Step 1: Add failing module-state, gradient-flow, optimizer, and metadata tests**

Extend imports in `tests/test_msa_vae_training_entrypoint.py`:

```python
from utils.msa_vae_training import (
    build_phase2_optimizer_param_groups,
    configure_msa_vae_trainability,
)
```

Add a small real module fixture and tests:

```python
class TinyMSACore(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_encoder = nn.Linear(3, 2, bias=False)
        self.cnn_decoder = nn.Linear(2, 3, bias=False)
        self.decode_proj = nn.Linear(2, 2, bias=False)
        self.trans_encoder = nn.Linear(2, 2, bias=False)
        self.trans_decoder = nn.Linear(2, 2, bias=False)
        self.global_proj = nn.Linear(2, 4, bias=False)
        self.local_proj = nn.Linear(2, 4, bias=False)


class TinyMSAWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.msa_vae = TinyMSACore()


class PhaseTrainabilityTest(unittest.TestCase):
    def test_phase2_frozen_local_projection_preserves_requested_module_states(self):
        model = TinyMSAWrapper()

        state = configure_msa_vae_trainability(
            model,
            phase=2,
            freeze_phase2_local_proj=True,
        )

        self.assertFalse(state["local_proj"])
        for name in (
            "cnn_encoder",
            "cnn_decoder",
            "decode_proj",
            "trans_encoder",
            "trans_decoder",
            "global_proj",
        ):
            self.assertTrue(state[name], msg=name)
        self.assertTrue(
            all(
                parameter.requires_grad
                for parameter in model.msa_vae.cnn_encoder.parameters()
            )
        )
        self.assertTrue(
            all(
                not parameter.requires_grad
                for parameter in model.msa_vae.local_proj.parameters()
            )
        )

    def test_local_loss_crosses_frozen_projection_and_optimizer_excludes_it(self):
        torch.manual_seed(123)
        model = TinyMSAWrapper()
        configure_msa_vae_trainability(
            model,
            phase=2,
            freeze_phase2_local_proj=True,
        )
        groups = build_phase2_optimizer_param_groups(
            model,
            lr=1e-3,
            cnn_lr_scale=0.1,
        )
        optimizer = torch.optim.SGD(groups)
        local_parameter_ids = {
            id(parameter)
            for parameter in model.msa_vae.local_proj.parameters()
        }
        optimizer_parameter_ids = {
            id(parameter)
            for group in groups
            for parameter in group["params"]
        }
        self.assertTrue(local_parameter_ids.isdisjoint(optimizer_parameter_ids))

        encoder_before = model.msa_vae.cnn_encoder.weight.detach().clone()
        projection_before = model.msa_vae.local_proj.weight.detach().clone()
        inputs = torch.ones(4, 3)
        mu = model.msa_vae.cnn_encoder(inputs)
        prediction = model.msa_vae.local_proj(mu)
        loss = (prediction - torch.ones_like(prediction)).square().mean()
        loss.backward()

        encoder_grad = model.msa_vae.cnn_encoder.weight.grad
        self.assertIsNotNone(encoder_grad)
        self.assertTrue(torch.isfinite(encoder_grad).all())
        self.assertGreater(encoder_grad.abs().sum().item(), 0.0)
        self.assertIsNone(model.msa_vae.local_proj.weight.grad)

        optimizer.step()
        self.assertFalse(
            torch.equal(
                encoder_before,
                model.msa_vae.cnn_encoder.weight.detach(),
            )
        )
        torch.testing.assert_close(
            projection_before,
            model.msa_vae.local_proj.weight.detach(),
        )

    def test_phase0_and_phase1_preserve_existing_trainability_contracts(self):
        model = TinyMSAWrapper()
        for parameter in model.parameters():
            parameter.requires_grad = False

        phase0 = configure_msa_vae_trainability(model, phase=0)
        self.assertTrue(all(phase0.values()))

        phase1 = configure_msa_vae_trainability(model, phase=1)
        for name in ("cnn_encoder", "cnn_decoder", "decode_proj"):
            self.assertFalse(phase1[name], msg=name)
        for name in (
            "trans_encoder",
            "trans_decoder",
            "global_proj",
            "local_proj",
        ):
            self.assertTrue(phase1[name], msg=name)

    def test_missing_required_module_and_empty_optimizer_group_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "local_proj"):
            configure_msa_vae_trainability(
                nn.Linear(2, 2),
                phase=2,
                freeze_phase2_local_proj=True,
            )

        model = TinyMSAWrapper()
        model.msa_vae.trans_encoder = nn.Identity()
        model.msa_vae.trans_decoder = nn.Identity()
        model.msa_vae.global_proj = nn.Identity()
        model.msa_vae.local_proj = nn.Identity()
        configure_msa_vae_trainability(model, phase=2)
        with self.assertRaisesRegex(ValueError, "top"):
            build_phase2_optimizer_param_groups(
                model,
                lr=1e-3,
                cnn_lr_scale=0.1,
            )
```

Add `"freeze_phase2_local_proj": False` to
`CheckpointMetadataTest._args()`, then assert:

```python
self.assertFalse(
    metadata["training_args"]["freeze_phase2_local_proj"]
)

enabled = build_msa_checkpoint_metadata(
    self._args(phase=2, freeze_phase2_local_proj=True)
)
self.assertTrue(
    enabled["training_args"]["freeze_phase2_local_proj"]
)
```

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```bash
conda run -n mgpt python -m unittest -v \
  tests.test_msa_vae_training_entrypoint.PhaseTrainabilityTest \
  tests.test_msa_vae_training_entrypoint.CheckpointMetadataTest.test_checkpoint_payload_preserves_state_and_metadata
```

Expected: import failure because
`configure_msa_vae_trainability` and
`build_phase2_optimizer_param_groups` do not exist.

- [ ] **Step 3: Implement the trainability helper**

Add to `utils/msa_vae_training.py`:

```python
MSA_REQUIRED_PHASE_MODULES = (
    'cnn_encoder',
    'cnn_decoder',
    'decode_proj',
    'local_proj',
)
MSA_CNN_MODULES = (
    'cnn_encoder',
    'cnn_decoder',
    'decode_proj',
)


def _msa_core(model):
    core = getattr(model, 'msa_vae', model)
    missing = [
        name
        for name in MSA_REQUIRED_PHASE_MODULES
        if not hasattr(core, name)
    ]
    if missing:
        raise ValueError(
            'MSA-VAE model is missing required modules: '
            + ', '.join(missing)
        )
    return core


def configure_msa_vae_trainability(
        model, phase, freeze_phase2_local_proj=False):
    """Apply the phase trainability contract after checkpoint loading."""
    if phase not in (0, 1, 2):
        raise ValueError(f'unsupported phase: {phase}')
    if freeze_phase2_local_proj and phase != 2:
        raise ValueError(
            'freeze_phase2_local_proj is valid only for Phase 2'
        )
    core = _msa_core(model)
    for parameter in core.parameters():
        parameter.requires_grad = True
    if phase == 1:
        for name in MSA_CNN_MODULES:
            for parameter in getattr(core, name).parameters():
                parameter.requires_grad = False
    elif phase == 2 and freeze_phase2_local_proj:
        for parameter in core.local_proj.parameters():
            parameter.requires_grad = False
    return {
        name: all(
            parameter.requires_grad
            for parameter in module.parameters()
        )
        for name, module in core.named_children()
    }
```

For parameterless modules such as `nn.Identity`, `all([])` deliberately
reports `True`.

- [ ] **Step 4: Implement trainable-only Phase-2 optimizer groups**

Add to `utils/msa_vae_training.py`:

```python
def build_phase2_optimizer_param_groups(model, lr, cnn_lr_scale):
    """Build trainable-only top/CNN parameter groups for Phase 2."""
    core = _msa_core(model)
    cnn_parameter_ids = {
        id(parameter)
        for name in MSA_CNN_MODULES
        for parameter in getattr(core, name).parameters()
    }
    cnn_params = []
    top_params = []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) in cnn_parameter_ids:
            cnn_params.append(parameter)
        else:
            top_params.append(parameter)
    if not top_params:
        raise ValueError('Phase 2 top optimizer group has no parameters')
    if not cnn_params:
        raise ValueError('Phase 2 CNN optimizer group has no parameters')
    return [
        {'params': top_params, 'lr': float(lr)},
        {
            'params': cnn_params,
            'lr': float(lr) * float(cnn_lr_scale),
        },
    ]
```

Add `'freeze_phase2_local_proj'` to
`TRAINING_IDENTITY_FIELDS`.

- [ ] **Step 5: Run Task 1 tests and verify GREEN**

Run:

```bash
conda run -n mgpt python -m unittest -v \
  tests.test_msa_vae_training_entrypoint.PhaseTrainabilityTest \
  tests.test_msa_vae_training_entrypoint.CheckpointMetadataTest
```

Expected: all selected tests pass, including the real backward/optimizer
step.

- [ ] **Step 6: Commit Task 1**

```bash
git add utils/msa_vae_training.py \
  tests/test_msa_vae_training_entrypoint.py
git commit -m "feat: define frozen local projection training contract"
```

---

### Task 2: CLI and training-entrypoint integration

**Files:**
- Modify: `options/option_msa_vae.py`
- Modify: `train_msa_vae.py`
- Modify: `tests/test_msa_vae_full_sequence_launchers.py`
- Modify: `tests/test_msa_vae_training_entrypoint.py`

**Interfaces:**
- Consumes: `configure_msa_vae_trainability` and
  `build_phase2_optimizer_param_groups` from Task 1.
- Produces: CLI flag `--freeze-phase2-local-proj` with destination
  `freeze_phase2_local_proj: bool`.
- Produces: Phase-2 log text stating whether `local_proj` is frozen.

- [ ] **Step 1: Add failing parser tests**

Extend `MSAOptionTest` in
`tests/test_msa_vae_full_sequence_launchers.py`:

```python
def test_phase2_local_projection_freeze_is_explicit_and_phase_scoped(self):
    with mock.patch.object(
        sys,
        "argv",
        [
            "train_msa_vae.py",
            "--phase",
            "2",
            "--freeze-phase2-local-proj",
        ],
    ):
        enabled = option_msa_vae.get_args_parser()
    self.assertTrue(enabled.freeze_phase2_local_proj)

    with mock.patch.object(sys, "argv", ["train_msa_vae.py"]):
        default = option_msa_vae.get_args_parser()
    self.assertFalse(default.freeze_phase2_local_proj)

    for phase in ("0", "1"):
        with self.subTest(phase=phase):
            with mock.patch.object(
                sys,
                "argv",
                [
                    "train_msa_vae.py",
                    "--phase",
                    phase,
                    "--freeze-phase2-local-proj",
                ],
            ):
                with self.assertRaises(SystemExit):
                    option_msa_vae.get_args_parser()
```

- [ ] **Step 2: Add a failing training-entrypoint wiring test**

Add to `DeterministicValidationSourceContractTest` in
`tests/test_msa_vae_training_entrypoint.py`:

```python
def test_entrypoint_applies_phase_trainability_before_optimizer(self):
    source = (ROOT / "train_msa_vae.py").read_text(encoding="utf-8")
    configure_at = source.index(
        "module_trainability = configure_msa_vae_trainability("
    )
    optimizer_at = source.index(
        "param_groups = build_phase2_optimizer_param_groups("
    )

    self.assertLess(configure_at, optimizer_at)
    self.assertIn(
        "freeze_phase2_local_proj=args.freeze_phase2_local_proj",
        source,
    )
    self.assertNotIn("def set_cnn_frozen(", source)
```

- [ ] **Step 3: Run Task 2 tests and verify RED**

Run:

```bash
conda run -n mgpt python -m unittest -v \
  tests.test_msa_vae_full_sequence_launchers.MSAOptionTest \
  tests.test_msa_vae_training_entrypoint.DeterministicValidationSourceContractTest.test_entrypoint_applies_phase_trainability_before_optimizer
```

Expected: the parser rejects the unknown flag, and the source-contract test
cannot find the helper calls.

- [ ] **Step 4: Implement the phase-scoped CLI flag**

Add under the phased-training arguments in
`options/option_msa_vae.py`:

```python
parser.add_argument(
    '--freeze-phase2-local-proj',
    '--freeze_phase2_local_proj',
    dest='freeze_phase2_local_proj',
    action='store_true',
    default=False,
    help=(
        'freeze the Phase-1 local projection during Phase 2 while '
        'preserving gradients to CNN mu'
    ),
)
```

Before returning parsed arguments, add:

```python
if args.freeze_phase2_local_proj and args.phase != 2:
    parser.error(
        '--freeze-phase2-local-proj is valid only with --phase 2'
    )
```

- [ ] **Step 5: Wire trainability and optimizer helpers into training**

Import from `utils.msa_vae_training`:

```python
build_phase2_optimizer_param_groups,
configure_msa_vae_trainability,
```

Replace the local `cnn_modules`, `top_modules`, `set_cnn_frozen`, and
phase branch with:

```python
module_trainability = configure_msa_vae_trainability(
    net,
    phase=args.phase,
    freeze_phase2_local_proj=args.freeze_phase2_local_proj,
)
if args.phase == 1:
    logger.info(
        'Phase 1: CNN encoder/decoder/decode_proj frozen; '
        'semantic modules trainable'
    )
elif args.phase == 2:
    logger.info(
        'Phase 2: CNN and semantic modules trainable; '
        f'local_proj frozen={args.freeze_phase2_local_proj}; '
        f'CNN LR scale={args.cnn_lr_scale}'
    )
else:
    logger.info(
        'Phase 0: legacy mode, all model parameters trainable'
    )
logger.info(
    'Module trainability: '
    + json.dumps(module_trainability, sort_keys=True)
)
```

Set `phase_desc` to:

```python
if args.phase == 1:
    phase_desc = 'Phase 1, CNN frozen'
elif args.phase == 2 and args.freeze_phase2_local_proj:
    phase_desc = 'Phase 2, local_proj frozen'
elif args.phase == 2:
    phase_desc = 'Phase 2, all trainable'
else:
    phase_desc = 'Phase 0, legacy'
```

Replace manual Phase-2 group construction with:

```python
param_groups = build_phase2_optimizer_param_groups(
    net,
    lr=args.lr,
    cnn_lr_scale=args.cnn_lr_scale,
)
optimizer = optim.AdamW(
    param_groups,
    betas=(0.9, 0.99),
    weight_decay=args.weight_decay,
)
logger.info(
    f'Optimizer: top LR={args.lr}, '
    f'CNN LR={args.lr * args.cnn_lr_scale}'
)
```

Do not alter `models/msa_vae.py`; its existing
`clip_local_feat = self.local_proj(mu)` call is the required gradient path.

- [ ] **Step 6: Run Task 2 tests and verify GREEN**

Run:

```bash
conda run -n mgpt python -m unittest -v \
  tests.test_msa_vae_full_sequence_launchers.MSAOptionTest \
  tests.test_msa_vae_training_entrypoint
```

Expected: all selected tests pass.

- [ ] **Step 7: Commit Task 2**

```bash
git add options/option_msa_vae.py train_msa_vae.py \
  tests/test_msa_vae_full_sequence_launchers.py \
  tests/test_msa_vae_training_entrypoint.py
git commit -m "feat: freeze local projection during phase 2"
```

---

### Task 3: Default-off shell interface and documentation

**Files:**
- Modify: `TRAIN_msa_vae_phase2.sh`
- Modify: `tests/test_msa_vae_full_sequence_launchers.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: the `--freeze-phase2-local-proj` CLI flag from Task 2.
- Produces: `FREEZE_PHASE2_LOCAL_PROJ=0|1`, default `0`.
- Produces: an exact reviewer-experiment invocation documented in README.

- [ ] **Step 1: Add failing launcher forwarding and validation tests**

Extend `MSAFullSequenceLauncherTest`:

```python
def test_phase2_local_projection_freeze_is_default_off_and_opt_in(self):
    default_arguments = self._run_launcher(
        "TRAIN_msa_vae_phase2.sh",
        {},
    )
    self.assertNotIn(
        "--freeze-phase2-local-proj",
        default_arguments,
    )

    enabled_arguments = self._run_launcher(
        "TRAIN_msa_vae_phase2.sh",
        {"FREEZE_PHASE2_LOCAL_PROJ": "1"},
    )
    self.assertIn(
        "--freeze-phase2-local-proj",
        enabled_arguments,
    )

    self._assert_launcher_rejected(
        "TRAIN_msa_vae_phase2.sh",
        {"FREEZE_PHASE2_LOCAL_PROJ": "yes"},
        "FREEZE_PHASE2_LOCAL_PROJ",
    )
```

- [ ] **Step 2: Run the launcher test and verify RED**

Run:

```bash
conda run -n mgpt python -m unittest -v \
  tests.test_msa_vae_full_sequence_launchers.MSAFullSequenceLauncherTest.test_phase2_local_projection_freeze_is_default_off_and_opt_in
```

Expected: enabled invocation lacks `--freeze-phase2-local-proj`, and malformed
input is not rejected.

- [ ] **Step 3: Implement the shell environment contract**

Near the other Phase-2 overrides in `TRAIN_msa_vae_phase2.sh`, add:

```bash
FREEZE_PHASE2_LOCAL_PROJ=${FREEZE_PHASE2_LOCAL_PROJ:-0}
case "$FREEZE_PHASE2_LOCAL_PROJ" in
  0)
    PHASE2_LOCAL_PROJ_ARGS=()
    ;;
  1)
    PHASE2_LOCAL_PROJ_ARGS=(--freeze-phase2-local-proj)
    ;;
  *)
    echo "ERROR: FREEZE_PHASE2_LOCAL_PROJ must be 0 or 1, got '${FREEZE_PHASE2_LOCAL_PROJ}'" >&2
    exit 2
    ;;
esac
```

Print:

```bash
echo "Freeze local proj: $FREEZE_PHASE2_LOCAL_PROJ"
```

Forward the array after `--phase 2`:

```bash
--phase 2 \
"${PHASE2_LOCAL_PROJ_ARGS[@]}" \
```

Update the launcher header to describe the optional fixed-projection policy,
without changing its default.

- [ ] **Step 4: Document the reviewer-experiment invocation**

In the README Phase-2 section, add:

````markdown
For the fixed-local-coordinate reviewer experiment, explicitly freeze the
Phase-1 local projection during Phase 2:

```bash
FREEZE_PHASE2_LOCAL_PROJ=1 \
PHASE1_DIR="$PHASE1_DIR" \
LOCAL_ALIGN_WEIGHT=0.05 \
bash TRAIN_msa_vae_phase2.sh <NUM_GPUS> t2m_272
```

The projection remains in the autograd graph: its parameters are excluded
from the optimizer, while local-alignment gradients still reach `mu` and the
CNN encoder. This option does not directly give the CNN decoder a local-loss
gradient.
````

- [ ] **Step 5: Run Task 3 tests and shell syntax check**

Run:

```bash
conda run -n mgpt python -m unittest -v \
  tests.test_msa_vae_full_sequence_launchers
bash -n TRAIN_msa_vae_phase2.sh
```

Expected: all launcher tests pass and Bash exits zero.

- [ ] **Step 6: Commit Task 3**

```bash
git add TRAIN_msa_vae_phase2.sh \
  tests/test_msa_vae_full_sequence_launchers.py README.md
git commit -m "docs: expose frozen local projection phase 2"
```

---

### Task 4: Cross-workflow regression and final verification

**Files:**
- Verify: `options/option_msa_vae.py`
- Verify: `utils/msa_vae_training.py`
- Verify: `train_msa_vae.py`
- Verify: `TRAIN_msa_vae_phase2.sh`
- Verify: `README.md`
- Verify: `tests/test_msa_vae_training_entrypoint.py`
- Verify: `tests/test_msa_vae_full_sequence_launchers.py`

**Interfaces:**
- Consumes: all deliverables from Tasks 1--3.
- Produces: fresh evidence that the opt-in protocol works and legacy/BABEL
  paths remain compatible.

- [ ] **Step 1: Run the focused and neighboring regression suite**

Run:

```bash
conda run -n mgpt python -m unittest -v \
  tests.test_msa_vae_training_entrypoint \
  tests.test_msa_vae_full_sequence_launchers \
  tests.test_msa_vae_training \
  tests.test_msa_vae_alignment \
  tests.test_msa_vae_alignment_pilot \
  tests.test_msa_vae_artifact_contract \
  tests.test_msa_vae_babel_launchers \
  tests.test_eval_msa_vae_babel
```

Expected: every test passes. If a programmatic `SimpleNamespace` fixture
constructs checkpoint metadata, add the explicit
`freeze_phase2_local_proj=False` field to that fixture rather than weakening
the production metadata contract.

- [ ] **Step 2: Compile every changed Python file**

Run:

```bash
conda run -n mgpt python -m py_compile \
  options/option_msa_vae.py \
  utils/msa_vae_training.py \
  train_msa_vae.py \
  tests/test_msa_vae_training_entrypoint.py \
  tests/test_msa_vae_full_sequence_launchers.py
```

Expected: exit zero with no output.

- [ ] **Step 3: Validate shell syntax and whitespace**

Run:

```bash
bash -n TRAIN_msa_vae_phase2.sh
git diff --check
```

Expected: both commands exit zero with no output.

- [ ] **Step 4: Inspect exact scope and commit history**

Run:

```bash
git status --short
git diff HEAD~3 -- \
  options/option_msa_vae.py \
  utils/msa_vae_training.py \
  train_msa_vae.py \
  TRAIN_msa_vae_phase2.sh \
  README.md \
  tests/test_msa_vae_training_entrypoint.py \
  tests/test_msa_vae_full_sequence_launchers.py
git log -4 --oneline
```

Expected: only the approved source, launcher, README, tests, design, and plan
commits relate to this feature; unrelated untracked files remain untouched.

- [ ] **Step 5: Report the verified behavior**

The handoff must state:

- the exact CLI flag and environment variable;
- default behavior remains unchanged;
- enabled Phase 2 freezes loaded Phase-1 `local_proj`;
- the backward test observed a finite, nonzero CNN encoder gradient and no
  local-projection gradient;
- frozen projection parameters are absent from the optimizer;
- no full training or GPU job was started;
- the exact regression count and verification commands.
