# TRACE v2 research proposal
## Predicting relationship-conditioned affective dynamics with TRILLsson, Mamba-2, and JEPA

Status: proposed experiments, not demonstrated results. Distillation is one experiment family, not a required component of the primary model. Prepared for Sathvik Napa, 9 October 2026.

## 1. Scientific goal and continuity with TRACE v1

TRACE v2 should test whether learning to predict a conversation partner's future paralinguistic representation improves detection of disrupted affective coordination, especially under distribution shift and limited labels. The scientific contribution is the predictive dyadic learning formulation and its evaluation; combining three named architectures alone is insufficient.

Keep the accepted paper's questions as the foundation:
- RQ1: Can disrupted affective coordination be reliably detected from speech under diverse relational and contextual conditions?
- RQ2: Do relationship type and conversational context provide complementary, architecture-sensitive signals for this detection?

Extend the experimental scope with two subordinate questions:
- Predictive learning: Does cross-speaker latent prediction improve RQ1/RQ2 outcomes beyond supervised learning, same-speaker prediction, and generic masked prediction?
- Efficiency: How much detection quality and human-rating agreement can a smaller causal student retain through task-specific distillation?

Working hypotheses, all falsifiable:
H1. TRILLsson complements emotion-fine-tuned Whisper, rather than necessarily replacing it.
H2. Mamba-2 reduces temporal-model memory or latency at long sequence lengths while maintaining detection quality. End-to-end gains may be limited by acoustic extraction.
H3. Cross-speaker JEPA yields larger benefits for low-label and held-out intervention conditions than for the original in-distribution test set.
H4. Relationship conditioning changes which response trajectories are informative; context may help only particular architectures or subgroups.
H5. Task-specific distillation improves the quality-efficiency tradeoff over the identical student trained without distillation.

### What v1 actually establishes

The supplied SLT paper reports DyadEE with 8,434 variants derived from 2,125 original dyads, 5,131 training and 3,303 test examples, and four data types totaling 1,069.8 hours. It uses fixed-duration windows per channel, at most 256 windows per speaker, emotion-fine-tuned Whisper-large-v3 embeddings, and a six-block bidirectional LLaMA-style temporal model. Context and relationship use 384-dimensional all-MiniLM-L6-v2 representations, projected to 128 dimensions each. TRACE-LF uses late fusion; TRACE-CE uses an auxiliary token querying audio through cross-attention.

The best Table III configuration is TRACE-CE Speech+Relationship: accuracy 93.47%, ROC-AUC 0.999, macro F1 0.934. Speech+Context+Relationship is 83.98% for CE and 83.34% for LF. Therefore v2 must not assume that using both metadata signals is optimal. Verify the original evaluation code and prediction files, including probability-based AUC, label mapping, and threshold selection, before treating these numbers as reproduced baselines.

The paper describes mostly speaker-disjoint splitting but reports one identity crossing splits in 12 rows. Preserve the original split for historical comparability, and add a strictly identity-disjoint audited split for primary v2 comparisons. Binary synthetic disruption labels are not continuous measures of real emotional entrainment. Human findings partly reflect speech delivery artifacts, so artifact-resistant evaluation is central.

## 2. Data and preprocessing contract

### Manifest

One row per conversation variant:
`sample_id, source_conversation_id, speaker_a_id, speaker_b_id, audio_a_path, audio_b_path, relationship, context_text, context_category, variant_type, label, synthesis_system, conversion_system, processing_version, split`.

Use anonymized speaker IDs, checksums, original channel offsets, sample rates, durations, and transformation provenance in associated metadata. Mark missing labels and missing social metadata explicitly. Store a separate window index with `speaker, start_s, end_s, valid_audio_samples, speech_fraction, valid_embedding, feature_cache_key`.

### Splitting before feature extraction or pretraining

Group all original, VC, resynthesized, and resynthesized+VC variants by source conversation. For the strict split, build connected components of speaker identities and conversations; assign entire components to train/validation/test. Also audit voice-reference identities, content hashes, transcript near-duplicates, and prompt distributions. Fit normalization only on training data. Prevent test speakers, conversations, and variants from entering JEPA pretraining or teacher distillation. Larger connected components may change the achievable split proportions; report actual counts rather than forcing the v1 ratio.

Maintain:
1. `legacy_v1`: original split, exact reproduction.
2. `strict_v2`: locked component-disjoint split, validation selected without test inspection.
3. `intervention_ood`: held-out synthesis/conversion system or transformation, applied only to held-out source groups.
4. `human_natural`: natural dyads with graded human annotations; held-out identities if available.

If only one synthesis system is available, temporal shuffling is a stress test, not evidence of generalization to unseen TTS systems. If an external natural dataset lacks entrainment annotations, collect them; do not substitute emotion labels.

### Audio windows and alignment

First reproduce v1's actual window duration from its code/configuration; the paper denotes it by t without specifying a numerical value. Then pilot 2, 4, and 8 second windows. Default proposed v2 setting: 4 seconds, non-overlapping, subject to the pilot. Keep both channels on the original shared clock. Their same-index windows are simultaneous time bins, not necessarily alternating conversational turns. Explicitly retain silence and overlap indicators. Use VAD to select eligible prediction targets and analyze activity, rather than deleting silence and destroying the common clock.

Resynthesis may change channel duration and timing. Audit temporal correspondence and speech boundaries for each variant. If reliable temporal alignment cannot be restored, report this as a limitation and keep JEPA pretraining on original dyads first. Do not infer cross-speaker lag effects from misaligned resynthesized channels.

Mask true padding separately from real silence. Use speech presence, channel identity, elapsed time, and duration as token features. Support long sequences with deterministic crops during training and causal chunking during inference. Avoid aggressive denoising, pitch modification, or time stretching as assumed label-preserving augmentations: these can alter the target affective dynamics.

## 3. Acoustic feature extraction

Three controlled settings:
- W: v1 emotion-fine-tuned Whisper, using the same checkpoint and window pooling.
- T: one pinned TRILLsson variant, extracted on exactly the same windows.
- W+T: normalized per-encoder projections and fusion.

TRILLsson is a family of distilled paralinguistic encoders. Its prior distillation is different from the task-specific teacher-student experiment proposed here. Its representations are not guaranteed to isolate emotion from identity, content, recording conditions, or synthesis traces. Verify the selected model's sample rate, output keys, native chunk behavior, embedding dimension, license, and checksum. Do not hard-code a family-wide embedding dimension.

Use separate extraction environments if the selected TRILLsson export requires TensorFlow while temporal training uses PyTorch. Exchange immutable float32 feature shards plus manifests. Freeze acoustic encoders in the initial experiments to keep cost and comparisons manageable. The JEPA-learned representations are then the adapters and temporal encoder outputs, not newly pretrained raw-audio foundations.

For window k of speaker s:
`uW = LN(PW(w[s,k])); uT = LN(PT(r[s,k])); u = PF(concat(uW,uT))`.

Starting dimensions: each projection 128; fused token 256. Single-encoder controls also output 256 dimensions. Append or project speaker, speech-activity, and elapsed-time features. Standardize cached features with training statistics and apply layer normalization in the model. Record native `dW` and `dT` from actual tensors.

## 4. Temporal architecture

### Primary model: dual-stream causal Mamba-2

Maintain a shared-weight Mamba-2 backbone with separate recurrent states for A and B. Sharing weights encourages invariance to arbitrary channel assignment; states remain distinct. Starting teacher: six blocks, width 256. Configure expansion, head dimension, state size, and convolution width against the pinned implementation, then record parameter counts. These are initial engineering choices, not tuned results.

At time k, read both previous states before updating either channel. Each speaker receives its own current token plus a projected summary of the other speaker's state at k-1. Use a learned sigmoid gate and residual mixing. Then update A and B synchronously. Never let B's same-bin future state depend on the already-updated A state through coding order. Alternate implementation: parallel per-speaker stacks with gated cross-speaker exchange between layers, constrained to earlier bins; benchmark both implementation throughput and the simpler interleaved baseline.

This enables explicit bidirectional influence across speakers while remaining causal in time. Cross-speaker exchange must be O(Nd) at fixed width, rather than full N-by-N attention. Straight Python recurrence can lose Mamba's parallel training benefits: start with interleaved Mamba-2 for kernel validation, then implement/test scan-compatible shifted exchanges. A dual-stream model is a hypothesis, not an automatic speed improvement.

### Baseline: interleaved Mamba-2

Use `A1,B1,A2,B2,...`, with speaker and shared timestamp embeddings. For future prediction, the cut occurs after the complete time bin k; do not interpret an arbitrary A-before-B order within a bin as a physical response. Test both channel orders or use exchange-symmetric fusion.

### Offline and causal tracks

The historical TRACE Transformer is bidirectional. Compare it with an offline forward+reverse Mamba variant for offline detection, with comparable parameter budgets. Evaluate causal Transformer and causal Mamba separately for streaming/prefix detection. Do not claim that a causal student matches an offline teacher under equal information. Offline whole-conversation distillation is an optional privileged-information experiment and must be named as such.

### Social conditioning and classifier

Reuse frozen all-MiniLM-L6-v2 for prompt text and relationship strings initially. Test none, relationship only, context only, and both. Project to a combined conditioning vector. Primary v2 integration: residual FiLM-style modulation, `h' = (1+gamma(m))*h + beta(m)`, on temporal features and predictor inputs. Initialize near identity; include metadata dropout and explicit missing metadata embeddings. Compare late fusion as a simpler control. Keep v1 CE unchanged in reproduction experiments.

At inference, masked mean pool valid temporal states (a running sum/count for causal deployment), concatenate optional metadata, and apply a two-layer binary classifier. Return probability of synthetic disruption/coordination class under the trained task definition. A prefix output describes evidence accumulated so far, not instantaneous ground-truth entrainment. A local scoring head requires local human annotations before it can be interpreted as local entrainment.

## 5. Cross-speaker JEPA training

### Prediction unit and information boundary

At cutoff k, context comprises both speakers' windows up to k. Predict the opposite speaker's latent representation for a future speech-active block at k+delta, conditioning on target speaker, gap, duration, and optional relationship/context. Sample both A-to-B and B-to-A directions. Include the target speaker's own past in the main formulation; a source-only context is an ablation, since useful prediction normally needs both histories.

Pilot horizons: 1, 2, and 4 bins (4, 8, 16 seconds at a 4-second window). Report physical delay distributions and separate fixed-clock prediction from next-speech-active-block prediction. For event selection, VAD uses future audio only to create training targets; target content never enters the context encoder. If target timing is supplied as a query, deployment prediction is conditional on that specified timing, not a prediction of when the partner will speak.

No overlap between context windows and target audio. If a pretrained extractor has a temporal receptive field beyond the requested window, ensure the extraction crop itself contains only that window. Prevent full-conversation normalization, bidirectional hidden states, or hidden-state caches from leaking future audio into context.

### Two train-time encoders

Online path: frozen acoustic features -> trainable adapters -> causal temporal encoder -> predictor. Target path: the same feature types for the target block -> EMA adapters and EMA temporal encoder -> normalized target embedding. The target encoder is an exponential moving average copy initialized from the online encoder; it receives the target block with reset state and no later audio. Pool only valid speech-active target windows. This defines the task as predicting the future response block's representation, rather than predicting a target that already summarizes the observed prefix.

The predictor receives the online prefix summary and query `(target_speaker, horizon, block_duration, social_metadata)`. Use a two-layer MLP initially, width 512 to latent width 256. Pool prefix and target with masks. A richer block predictor is optional after this works.

`Lpred = mean[1 - cosine(predictor(history, query), stopgrad(target_latent))]`.
`theta_target <- tau*theta_target + (1-tau)*theta_online`, with a proposed tau ramp from 0.996 toward 0.9999. Frozen acoustic encoders do not need EMA copies. EMA applies only to architecture-compatible learned modules in both paths; the predictor has no EMA target counterpart.

Track collapse explicitly: per-dimension standard deviation, effective rank, inter-sample cosine similarity, and prediction advantage over a constant-mean target. EMA alone is not a proof against collapse. Add a fixed, predeclared variance/covariance regularizer on online target-view embeddings (without stop-gradient) if the pilot needs it; report the regularizer as an additional objective and ablate it. Compute statistics over independent source groups where possible, not many near-identical windows from one dyad.

### Training schedule

Stage A: pretrain on original training dyads only, optionally adding validated VC positives. Treat these as presumed coordinated data, not universally entrained ground truth. Train 25-50 epochs initially with AdamW, lr 1e-4, warmup and cosine decay, mixed precision after numerical checks, and effective batch 32-128 independent dyads. Tune training duration using validation prediction quality plus noncollapse diagnostics. Never use test data as unlabeled pretraining data.

Stage B: compare frozen-encoder linear probing and full temporal/adaptor fine-tuning for supervised EE/NE detection. Main fine-tuning loss is cross-entropy. Optional joint training keeps Lpred only on original/validated positive training examples: `L = LCE + lambda_pred*Lpred + lambda_reg*Lreg`. Sweep lambda_pred over 0, 0.1, and 1 in the pilot. Predicting resynthesized disruption targets can teach the model to model the disruption distribution instead of natural response dynamics; test all-variant JEPA as a separate control.

Stage C: calibrate probabilities and choose thresholds using validation only. Freeze the model and protocol before test evaluation.

Do not directly treat JEPA residual as an entrainment score: human responses can be unpredictable yet appropriate. Test residual-based detection against learned classification, and test whether residuals correlate with human coordination after accounting for acoustic naturalness.

## 6. Experiment matrix and priorities

Use the strict split for matched comparisons, same seeds/data budgets, and comparable tuning budgets. All new rows are proposed runs.

| ID | Experiment | What it isolates | Priority |
|---|---|---|---|
| E0 | Reproduce Emotion MLP, DyadFormer, TRACE-LF and TRACE-CE on legacy and strict splits | Historical reference and split effects | Required |
| E1 | Matched Transformer with W vs T vs W+T, CE only | Acoustic representation contribution | Required |
| E2 | Transformer vs interleaved Mamba-2 vs dual-stream Mamba-2, CE only, fixed feature setting | Temporal architecture; causal/offline results kept separate | Required |
| E3 | Selected Mamba: supervised only vs JEPA initialization + fine-tuning; fixed encoders and data | Benefit of predictive pretraining | Required |
| E4 | Same-speaker vs cross-speaker vs mixed-direction JEPA; generic masked-block prediction control | Whether interaction prediction matters | Required |
| E5 | No metadata / relationship / context / both; FiLM vs late fusion on finalists | RQ2 and integration sensitivity | Required |
| E6 | 1%, 5%, 10%, 25%, 100% training-source labels with fixed held-out sets | Label efficiency; repeated stratified source-group sampling | Required |
| E7 | Human-natural ratings, held-out intervention systems, artifact-balanced tests | Real coordination and robustness | Required |
| E8 | Task-specific teacher-student distillation and component ablations | Quality-efficiency tradeoff | Requested experiment |
| E9 | Optional larger unlabeled corpus, adapter tuning, multi-horizon or multimodal additions | Scaling beyond the minimum study | Stretch |

Run encoder screening before the full backbone matrix. Do not exhaustively cross every option. Use three seeds for main comparisons, five for finalists if resources permit. Pilot one seed on training/validation subsets only. Prefer a strong frozen-feature study and credible natural evaluation over many weak architectural additions.

Essential controls: audio-only; metadata-only; silence/length-only; single-speaker; speaker-swap; shuffle one speaker's temporal order; shift one channel; train and test by synthesis variant; prediction of own future only; randomly paired partner target; constant-mean latent prediction; target-horizon sweep; with/without collapse regularization. Matched partner-shuffle controls should preserve relationship/context and activity/length distribution as much as possible. Shuffles/lag shifts are sensitivity probes, not automatically valid negative entrainment labels.

## 7. E8: Distillation as a separate experiment

### Teacher and student

Freeze the best validation-selected causal W+T Mamba-JEPA detector as teacher. If W+T does not beat T, still run a prespecified W+T teacher comparison, but identify the best empirical teacher separately. Default student: the same frozen TRILLsson variant, a smaller causal Mamba-2 (two blocks, width 128), and a binary head. Distillation removes Whisper and shrinks the temporal model; TRILLsson extraction may still dominate cost. A smaller TRILLsson variant is a second, separately measured compression factor.

Teacher and student must see identical prefixes, windows, social metadata availability, labels, and timestamps. Teacher features may include Whisper; teacher prefix must contain no later audio. The completed teacher's frozen prediction targets are not the JEPA EMA encoder. Keep separate classes and checkpoints to prevent accidental confusion.

### Losses

`LKD = T^2 * KL(softmax(zTeacher/T) || softmax(zStudent/T))` over two-class logits.
`Lstate = mean[1 - cosine(Q(hStudent), stopgrad(hTeacher))]` at aligned valid prefix endpoints.
`Lfuture = mean[1 - cosine(Qp(predStudent), stopgrad(predTeacher))]` for aligned future queries.
`Lstudent = alpha*LCE + beta*LKD + gamma*Lstate + delta*Lfuture`.

Use small learned projections Q/Qp when widths differ. Proposed pilot: temperature 2 or 4; coefficients selected on validation with equal budgets. Teacher targets have stop-gradient; projection and student receive gradients. If deploying only classification, student predictor is training-only and removed at inference. Future prediction distillation is optional and cannot be presented as necessary.

Distillation component table:
- S0: T + small causal Mamba, CE only, random temporal initialization.
- S1: identical S0, own JEPA pretraining, no task-specific distillation.
- S2: identical initialization to S0 + logit distillation.
- S3: identical initialization to S0 + logit and state distillation.
- S4: identical initialization to S0 + logits, states, future prediction distillation.
- S5: S1 initialization + full distillation, to measure combined benefit.

Freeze the student architecture and training budget across S0-S5. Use all training variants for supervised/logit distillation and natural/validated positive training blocks for future prediction distillation, consistent with JEPA pretraining. Save teacher outputs on training examples only or compute online with no gradients.

Predeclare a useful operating target, for example no more than 2 percentage points macro F1 loss versus the causal teacher and measurable end-to-end latency/memory reduction. This is an acceptance criterion, not a forecast. Report Pareto curves even if no student meets it. There is no need to make distillation the paper's title-level claim unless E8 is especially compelling.

## 8. Evaluation, statistics, and interpretation

### Primary and secondary endpoints

Primary synthetic benchmark metric: macro F1 on strict_v2. Secondary: accuracy, balanced accuracy, AUROC, AUPRC with positive label stated, per-class recall, Brier score and reliability plots. Report scores for all four variants separately and confusion matrices. Use probabilities, not hard labels, for AUC. Explain unexpectedly high ranking performance alongside lower accuracy via validation calibration/threshold checks.

Report effect sizes as paired differences with 95% clustered bootstrap intervals. Resample speaker connected components (or source dyads if components reduce to dyads), keeping all variants together; use at least 2,000 replicates. Separate variation across training seeds from evaluation sampling uncertainty. Apply a declared multiplicity procedure such as Holm correction to the small set of confirmatory comparisons; mark remaining analyses exploratory. Label-efficiency fractions are based on source groups, never independently sampled variants.

### Human validation

Plan roughly 150-250 held-out source dyads, stratified by relationship/context and real/synthetic status where feasible; pilot annotation cost first. Obtain at least three ratings per item. Separately ask for affective coordination/appropriate emotional response, speech naturalness, and audio quality. Include examples of regulation and complementarity so raters do not equate entrainment with emotion matching. Blind raters to systems/labels; randomize ordering and audit attention. Report inter-rater reliability with confidence intervals and aggregate-rating uncertainty.

Evaluate Spearman association of model outputs with coordination ratings, plus association with naturalness. Analyze coordination conditional on naturalness as an exploratory check, not causal adjustment proof. Prefer a natural-only rating subset so correlation cannot be driven entirely by original-vs-synthetic status. No continuous prediction claim without independent graded annotations.

### Generalization and shortcut analysis

Hold out contexts or relationship groups only when there are enough distinct source groups; siblings and parent-child are sparse in v1. Report support counts and uncertainty; do not treat augmented copies as new people. For rare or unseen relationships, explicit unknown-metadata tests are more defensible than tiny leaderboards. Compare synthesis-artifact probes against coordination models and evaluate matched synthesis conditions. Test unchanged metadata across transformations and adversarially shuffled metadata to detect brittle reliance.

### Efficiency and streaming

Benchmark the same GPU/CPU, software versions, batch sizes, sequence lengths, and warmup protocol. Measure p50/p95 latency, peak memory, parameters, checkpoint size, throughput, and real-time factor (`processing_seconds/audio_seconds`). Report acoustic extraction, temporal inference, transfer/I/O, and full pipeline separately. Benchmark feature-cached results and raw-audio results; only the latter support end-to-end efficiency claims.

Use 32, 64, 128, 256, 512 bins where data supports them. If feature sequences are repeated to fill a synthetic benchmark, call it a scaling benchmark and do not use it for accuracy. Report first usable prediction latency including the audio-window buffering delay. Demonstrate chunked-vs-full causal output agreement, no state leakage between conversations, and prefix performance at 25%, 50%, 75%, and 100% of a conversation. Online inference of a speech response is not speech generation or agent adaptation.

## 9. Proposed repository structure

This is an implementation blueprint; the files below are proposed, not an already implemented runnable repository.

```text
trace-v2/
  README.md                       # task definition, setup, experiment IDs
  pyproject.toml                  # PyTorch training package
  environments/
    extract-trillsson.lock         # isolated extraction environment
    train-cu.lock                 # pinned CUDA/Torch/Mamba stack
  configs/
    data/{legacy_v1,strict_v2,intervention_ood,human_natural}.yaml
    features/{whisper,trillsson,fused}.yaml
    model/{trace_v1,transformer,interleaved_mamba,dual_mamba,student}.yaml
    objective/{ce,jepa_same,jepa_cross,jepa_mixed,distill}.yaml
    experiment/{e0_reproduce,e1_features,e2_backbones,e3_jepa,
                e4_prediction,e5_metadata,e6_low_label,e7_robustness,
                e8_s0,e8_s1,e8_s2,e8_s3,e8_s4,e8_s5}.yaml
  src/trace_v2/
    data/
      schema.py                   # typed sample/window/feature records
      split_components.py         # identity graph and grouped assignment
      audit.py                    # split/provenance/hash validation
      audio.py                    # rates, offsets, mono channels, VAD
      windowing.py                # shared clock, masks, eligible targets
      dataset.py                  # cached features and paired variants
      collate.py                  # lengths, padding, activity/query masks
    features/
      whisper.py                  # frozen v1 checkpoint adapter
      trillsson.py                # model-specific interface and dimensions
      extraction.py               # shard creation and normalization stats
      cache.py                    # hashes and extraction provenance
    models/
      adapters.py                 # W/T/fused token projections
      transformer.py              # matched causal/offline controls
      trace_v1.py                 # original LF/CE wrappers
      mamba_interleaved.py         # speaker/time aware single stream
      mamba_dual.py                # separate states, synchronous exchange
      conditioning.py             # none, late fusion, FiLM
      predictor.py                # target speaker/horizon queries
      heads.py                    # masked pooling and detection head
      student.py                  # independently specified small model
    objectives/
      jepa.py                     # online/target loss and target-only view
      ema.py                      # EMA update/state serialization
      collapse.py                 # variance/rank diagnostics, regularizer
      distillation.py             # logits/state/future losses
    training/
      pretrain.py                 # Stage A
      finetune.py                 # Stage B, probe/full tuning
      distill.py                  # E8 only, frozen completed teacher
      checkpoint.py               # all RNG/optimizer/EMA states
    evaluation/
      metrics.py                  # discrimination and calibration
      bootstrap.py                # clustered paired intervals
      human.py                    # ratings and reliability
      robustness.py               # interventions and artifact controls
      streaming.py                # prefix metrics and state reset
      benchmark.py                # full pipeline and cached-feature timing
    inference/stream.py           # stateful causal API
  scripts/
    build_manifest.py
    audit_splits.py
    extract_features.py
    run_experiment.py
    export_teacher_targets.py
    evaluate_checkpoint.py
    benchmark_pipeline.py
    build_paper_tables.py
  slurm/
    extract_array.sbatch
    train_single_gpu.sbatch
    evaluate.sbatch
  tests/
    test_split_integrity.py
    test_window_alignment.py
    test_future_leakage.py
    test_padding_invariance.py
    test_ema_stopgrad.py
    test_synchronous_exchange.py
    test_streaming_equivalence.py
    test_distillation_alignment.py
  docs/
    architecture.svg
    data_card.md
    annotation_protocol.md
    experiment_registry.md
  data/                           # ignored; manifests and licensed data refs
  artifacts/                      # ignored; features/checkpoints/teacher caches
  results/                        # version compact manifests/tables, not raw audio
  notebooks/                      # visualization only; no hidden training logic
```

Use Hydra for composition, PyTorch for training, the official state-spaces/mamba package for Mamba-2, TensorFlow only when required by the chosen TRILLsson artifact, PyArrow/Parquet manifests, NumPy/Zarr or tensor shards for features, scikit-learn for metrics, SciPy for statistics, and SLURM for cluster jobs. Use either local JSONL tracking or one configured experiment tracker; avoid making a cloud tracker mandatory. Pin versions after a CUDA compatibility smoke test, not from unverified assumptions.

Each run writes a resolved config, git SHA, dataset/split hashes, encoder hashes, hardware/software information, seed, learned normalization statistics, validation-selected checkpoint, probability predictions, and summary metrics. Feature caches key on source checksum + channel + crop boundaries + preprocessing + model checksum + pooling. Do not cache tensors with trainable adapters baked in unless regenerating every update.

### Key contracts

A batch contains feature tensors `[B,2,N,dW]` and/or `[B,2,N,dT]`, valid masks `[B,2,N]`, activity/timestamp features, social metadata, source IDs, and optional target queries. A context encoder returns prefix summaries and valid per-bin states; a target encoder returns target-block latents with independent state. A detector returns two logits; probability mapping is documented. An inference state holds separate A/B Mamba states plus pooling accumulators and is reset on each new dyad.

Tests must prove: no split identity/variant leakage; future audio perturbation cannot change an earlier context embedding; padding cannot change valid predictions; targets have no backpropagated gradients; online weights do receive gradients; exchange does not depend on channel update order; full causal and chunked evaluation agree; teacher/student queries and logits are aligned. Include an explicit test that random channel swaps with appropriately swapped queries preserve expected symmetric outputs. Exact equality may not hold for interleaved channel-order baselines; quantify that limitation.

## 10. Implementation sequence and decision gates

| Phase | Work | Completion gate |
|---|---|---|
| Week 1 | Manifest, strict split audit, v1 wrappers, checkpoint verification | Reproduced scores or documented discrepancies; zero unaccounted leakage |
| Week 2 | W/T/fused extraction, matched Transformer and interleaved Mamba pilot | Stable shapes/cache hashes; empirical encoder choice |
| Week 3 | Dual-stream model, causal/offline comparisons, streaming tests | Correct information boundary and verified recurrent state behavior |
| Week 4 | JEPA same/cross-speaker pilot, collapse diagnostics | Better than constant predictor; no collapse; no future leakage |
| Week 5 | Supervised/probe, low-label and metadata experiments | Validation-selected finalists and matched ablations |
| Week 6 | E8 students and latency benchmarks | Distillation component results and full-pipeline Pareto curves |
| Weeks 7-8 | Human ratings, intervention OOD, statistics, paper tables | Locked evaluation, uncertainty estimates, supported claims |

This is a sequencing estimate, not a guaranteed eight-week turnaround. Begin human annotation logistics early because that work can dominate elapsed time. Feature extraction may require separate scheduling. Estimate resource needs from a 1-5% training pilot; record feature seconds/hour, training seconds/epoch, peak memory, and extraction/checkpoint storage. Extrapolate from measured throughput and number of planned runs. Do not budget all 1,069.8 hours as independent natural pretraining data: the original natural audio is 230.1 hours and many variants share content and speakers.

Stop/go decisions:
- If T alone is competitive and W+T adds no stable value, prefer T for the main model and retain fusion as an ablation.
- If dual-stream loses throughput without improving quality, select interleaved Mamba and report the failed hypothesis.
- If JEPA residuals primarily track identity or synthesis, revise targets/controls before increasing scale.
- If JEPA has no reproducible detection/generalization advantage, do not claim the predictive objective is useful merely because its training loss decreases.
- If natural-rated coordination does not improve, limit claims to controlled disruption detection.
- If student benefits appear only with cached features, limit efficiency claims accordingly.

## 11. Expected paper outputs and claim boundaries

Figures: architecture and train/inference boundaries; quality vs label fraction; OOD degradation; human coordination vs naturalness association; teacher/student Pareto curves; temporal-model memory/latency scaling. Tables: v1 reproduction; encoder/backbone decomposition; prediction objective and metadata ablations; human/generalization results; E8 distillation components. Include split audit and extraction/model cards in supplementary material.

Strong outcome: predictive dyadic training improves held-out coordination judgments and artifact-resistant detection, while a smaller causal student retains much of that benefit. Valid narrower outcome: Mamba/TRILLsson offer a practical quality-efficiency tradeoff on controlled disruption detection without proving real-world entrainment measurement. Either conclusion must follow the experiments.

## 12. Primary sources and implementation starting points

- Supplied accepted paper: TRACE: Temporal Relationship-Aware Conversational Entrainment Detection in Dyadic Speech, 2026332483.pdf. Dataset and baseline numbers above are taken from this attachment.
- TRILLsson paper: https://arxiv.org/abs/2203.00236
- Google explanation and model links: https://research.google/blog/trillsson-small-universal-speech-representations-for-paralinguistic-tasks/
- Mamba-2 / structured state-space duality: https://proceedings.mlr.press/v235/dao24a.html
- Official Mamba code: https://github.com/state-spaces/mamba
- I-JEPA: https://arxiv.org/abs/2301.08243
- Original TRACE code link reported by paper: https://github.com/SathvikNapa/NaturalnessPrediction
- Original TRACE dataset link reported by paper: https://huggingface.co/datasets/SathvikNapa/TRACE

TRILLsson supplies pretrained paralinguistic features; Mamba-2 supplies a temporal architecture; I-JEPA motivates latent-target prediction. None of these sources establishes the proposed combination's effectiveness for emotional entrainment. The cross-speaker design, schedule, dimensions, target selection, and experimental gates above are research proposals.
