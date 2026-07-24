# NeurIPS 2026 Submission 21588 - Consolidated Original Reviews

**Paper:** Align to Retrieve: Multi-Scale Semantic Latent Alignment for Retrieval-Augmented Text-to-Motion Generation
**Submission ID:** 21588
**Venue:** NeurIPS 2026
**Decision display:** No Recommendation

> Transcription note: This document consolidates the original English text shown in the three official review screenshots and the Area Chair meta-review screenshot. Only layout, line wrapping, and Markdown formatting have been normalized.

## Overall Scores

| Reviewer | Rating | Confidence |
|---|---:|---:|
| Reviewer 8Eee | 3 | 5 |
| Reviewer SjtL | 2 | 4 |
| Reviewer ojS1 | 3 | 4 |
| **Average** | **2.67** | **4.33** |

---

## Official Review by Reviewer ojS1

**Date:** 25 Jun 2026, 00:05 (modified: 23 Jul 2026, 23:38)

### Summary

This paper proposes MSA-T2M, a text-to-motion framework that combines a multi-scale semantic alignment VAE with retrieval-augmented autoregressive diffusion. MSA-VAE aligns local motion latents with action annotations and a global representation with sequence captions, while RAG-Diffusion-AR retrieves motion priors from the learned latent space to improve realism. The reported results show competitive performance on HumanML3D-272, including the best FID among the compared baseline methods.

### Contribution Type

General: Most submissions will fall into this type.

### Strengths And Weaknesses

**Strengths:** The paper addresses an important tension between semantic controllability and motion realism. The proposed pipeline is coherent: the learned global representation is reused as a retrieval key, and the retrieved priors are integrated into generation. The quantitative results are competitive, and the component ablation attempts to separate the effects of local alignment, global alignment, and retrieval augmentation.

**Weaknesses:**

1. The novelty relative to prior work is insufficiently established. MotionBind appears highly relevant but is not cited in the discussion. In particular, MSA-VAE is conceptually similar to MotionBind’s MuTMoT representation learning, while the retrieval mechanism and synergistic classifier-free guidance appear closely related to ReMoDiffuse and MotionBind’s REALM. This weakens the paper’s central novelty claim. The paper should provide a precise technical comparison identifying which components are inherited, which differ, and what measurable benefit is attributable to the proposed changes.
2. The local alignment formulation is not quite clear. BABEL provides action labels for temporal segments rather than frame-level labels. Assigning these segment annotations to individual latent steps and designating them as frame-level supervision may introduce incorrect alignment, particularly near action boundaries. Averaging the T5 embeddings of all labels within a temporal window is also not justified. The authors should clarify the exact segment-to-latent mapping, justify the embedding average, and evaluate alternatives such as aligning a pooled segment representation with its action label or using genuinely frame-level textual supervision.
3. The rationale and construction of the semantic alignment module are unclear. The paper does not adequately explain why a separate Semantic Encoder/Decoder is necessary, rather than directly aligning the representation encoder’s features. The global alignment target is also insufficiently motivated: it is unclear why training must use fixed-length windows of (T=64), why (α = T/L) is an appropriate interpolation weight, or precisely how (l_pooled) is constructed. These choices directly affect the claimed local/global semantic organization of the latent space and therefore require clearer derivation and targeted ablations, including comparison with full-length training.
4. The empirical evidence does not consistently support the alignment claims. ReMoDiffuse obtains substantially better R-Precision despite not using the proposed explicit local/global alignment, which complicates the claim that MSA-VAE produces more semantically precise generations. In the ablation, removing local alignment improves all reported retrieval metrics relative to the full model, while the local-only configuration is not reported (left blank).
5. Retrieval quality is not evaluated directly. Because a central contribution is the construction of a semantically discriminative retrieval index, generation metrics alone are insufficient to validate this claim. Direct text-to-motion and motion-to-text retrieval evaluations against appropriate baselines such as TMR and MoPa are needed.

**Quality:** 3: good
**Clarity:** 2: not good
**Significance:** 2: not good
**Originality:** 2: not good

### Questions

1. What are the key methodological differences between the proposed approach and MotionBind, which learns both global and fine-grained motion representations and uses the global representation for retrieval-augmented diffusion? Moreover, how do the retrieval and guidance mechanisms differ from those used in ReMoDiffuse?
2. Why is the Semantic Encoder/Decoder required, and what happens when semantic losses are applied directly to the representation features?
3. Why are the BABEL segment-level annotations assigned to individual latent time steps? Wouldn't frame-level annotations, such as those available in Motion-X, or a segment-level alignment, provide more precise supervision? Please clarify the rationale for the chosen annotation-to-latent alignment strategy.
4. What is the derivation or empirical justification for (α = T/L)? In addition, please define precisely how (l_pooled) is computed, including the pooling operation and the features over which it is applied.
5. Why does removing the local-alignment objective improve R-Precision? This result appears counterintuitive and warrants further analysis. Additionally, why is the result for the local-only configuration not reported?
6. How well does the learned ([CLS]) representation perform on standard standalone text-to-motion and motion-to-text retrieval benchmarks? A direct comparison with retrieval-focused methods such as TMR and MoPa would help establish the quality of the learned global representation.
7. What accounts for the discrepancy between the reported FID values and those typically obtained using the standard HumanML3D evaluation protocol associated with Guo et al.?

### Limitations

yes

**Rating:** 3: Borderline reject: Technically solid paper where reasons to reject, e.g., limited evaluation, outweigh reasons to accept, e.g., good evaluation. Please use sparingly.

**Confidence:** 4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

**Ethical Concerns:** NO or VERY MINOR ethics concerns only

**Paper Formatting Concerns:**

No concerns.

**Code Of Conduct Acknowledgement:** Yes

**Responsible Reviewing Acknowledgement:** Yes

---

## Official Review by Reviewer SjtL

**Date:** 22 Jun 2026, 15:01 (modified: 23 Jul 2026, 23:38)

### Summary

This paper proposes MSA-T2M, a retrieval-augmented text-to-motion generation framework designed to generate motions that are both semantically aligned with text prompts and physically realistic. The method first learns a multi-scale semantically organized latent space through MSA-VAE, using local frame-level BABEL action labels and global sequence-level HumanML3D captions. This latent space is then used as a retrieval bank: RAG-Diffusion-AR retrieves semantically similar real motion priors and injects them into an autoregressive diffusion generator to improve motion realism. Experiments on HumanML3D-272 show improved FID over prior methods, while maintaining competitive text-motion alignment metrics.

### Contribution Type

General: Most submissions will fall into this type.

### Strengths And Weaknesses

**Strengths:**

- The paper presents a clear and intuitive coupling between latent semantic alignment and retrieval-augmented generation.
- The local-global alignment design is well motivated for compositional motion prompts.
- The method achieves the best reported FID among compared methods on HumanML3D-272.

**Weaknesses:**

1. The paper identifies the alignment-realism trade-off as a central problem and claims to resolve it, yet the empirical evidence supporting this claim is inadequate. While R-Precision and MM-Dist are reasonable proxies for text-motion alignment, FID measures distributional similarity in a learned latent space and does not directly reflect whether the generated motions are physically plausible. A more principled evaluation of physical naturalness, such as foot skating metrics, joint acceleration analysis, or biomechanical plausibility scores, is necessary to substantiate the claim that the proposed method genuinely improves realism.
2. The assumption that an all-zero pose latent represents an end state is not well-justified. Human motions do not necessarily terminate near a zero pose; actions such as "lie down," "sit," or "cartwheel" end in postures that can be far from a neutral standing configuration. The paper should provide a principled justification for this design choice, along with empirical evidence that it generalizes across diverse motion types. Additionally, it is unclear whether ground-truth motion lengths are used during evaluation. If they are, this must be explicitly stated and justified, as it would constitute privileged information not available in real deployment scenarios.
3. The experimental evaluation is conducted on a narrow set of benchmarks. To substantiate the generality of the proposed approach, evaluation on additional datasets such as KIT-ML [1] and Motion-X [2] is necessary. Furthermore, the comparisons are made against models that may no longer represent the state of the art. Recent methods including BAMM [3], MoGenTS [4], and SALAD [5] should be included in both quantitative and qualitative comparisons. There is a real possibility that the limitations the paper attributes to prior work have already been addressed by these more recent models, which would significantly weaken the motivation of the proposed approach.
4. The paper does not isolate or quantify the contribution of the MSA-VAE component. Given that the VAE plays a central role in the architecture, it is essential to determine what portion of the overall performance gain originates from this module. Beyond performance attribution, the paper should provide analysis verifying that the design intentions of the VAE are realized in practice. For instance, it should be demonstrated that the CLS token actually encodes the intended semantic information, and that each stage of the multi-scale architecture contributes meaningfully rather than redundantly.
5. Diffusion-based methods are well-known for their slow inference, and this is a practical concern that the paper does not address. A comparison of inference time and overall computational cost against methods such as MoMask [6], SALAD, and Light-T2M [7] is necessary. Without this, it is difficult to assess whether the proposed method offers a practical advantage or simply trades speed for marginal quality gains.
6. The paper attributes performance improvements to the quality of the aligned latent bank used in retrieval-augmented generation. However, the current experiments do not disentangle whether the gains come from the retrieval content itself or simply from the addition of an extra conditioning token. A controlled experiment, such as replacing retrieved latents with random or unaligned latents while keeping the conditioning pipeline intact, is needed to confirm that the retrieval quality is the actual source of improvement.
7. The local-only ablation result is left empty in Table 2, yet multi-scale alignment is presented as one of the central contributions of this work. Without this result, it is impossible to understand the individual role of local alignment or to make a fair comparison against the global-only variant. This experiment should be completed and reported.
8. The ablation results in Table 2 reveal an inconsistency that undermines the claim that RAG consistently improves motion quality. The global-only variant without RAG achieves an FID of 11.236, whereas adding RAG to the global alignment variant (i.e., the configuration without local alignment) yields an FID of 11.358, which is actually worse. This directly contradicts the stated claim. The authors should discuss under what conditions RAG provides a benefit, and acknowledge that its effect appears to be dependent on specific architectural combinations rather than being universally beneficial.
9. There are conflicting statements about the value of K used in experiments. In one part of the ablation discussion, it is stated that K=3 is used for both ablations and SOTA comparisons. However, the same section later reads "we fix K=3 for SOTA comparison and ablations," which conflicts with Figure 4, where K=5 is shown to yield the best FID. This inconsistency needs to be resolved with a clear and uniform statement of which K value is used for each set of experiments, and why.
10. There are discrepancies between the numbers reported in Table 1 and those cited in the body of the paper. Table 1 reports FID of 10.826, MM-Dist of 15.820, and R@1 of 0.659, while the corresponding text references values of 10.832, 15.917, and 0.638 respectively. These inconsistencies must be corrected before publication, as they undermine confidence in the reported results.

#### References cited by the reviewer

[1] Plappert, Matthias, Christian Mandery, and Tamim Asfour. "The kit motion-language dataset." Big data 4.4 (2016): 236-252.
[2] Lin, Jing, et al. "Motion-x: A large-scale 3d expressive whole-body human motion dataset." Advances in Neural Information Processing Systems 36 (2023): 25268-25280.
[3] Pinyoanuntapong, Ekkasit, et al. "Bamm: bidirectional autoregressive motion model." European Conference on Computer Vision. Cham: Springer Nature Switzerland, 2024.
[4] Yuan, Weihao, et al. "Mogents: Motion generation based on spatial-temporal joint modeling." Advances in Neural Information Processing Systems 37 (2024): 130739-130763.
[5] Hong, Seokhyeon, et al. "Salad: Skeleton-aware latent diffusion for text-driven motion generation and editing." Proceedings of the Computer Vision and Pattern Recognition Conference. 2025.
[6] Guo, Chuan, et al. "Momask: Generative masked modeling of 3d human motions." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024.
[7] Zeng, Ling-An, et al. "Light-t2m: A lightweight and fast model for text-to-motion generation." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 39. No. 9. 2025.

**Quality:** 2: not good
**Clarity:** 2: not good
**Significance:** 2: not good
**Originality:** 2: not good

### Questions

My questions are in the weaknesses part.

### Limitations

Yes.

**Rating:** 2: Reject: For instance, a paper with technical flaws, weak evaluation, inadequate reproducibility and incompletely addressed ethical considerations.

**Confidence:** 4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

**Ethical Concerns:** NO or VERY MINOR ethics concerns only

**Paper Formatting Concerns:**

None.

**Code Of Conduct Acknowledgement:** Yes

**Responsible Reviewing Acknowledgement:** Yes

---

## Official Review by Reviewer 8Eee

**Date:** 01 Jun 2026, 20:26 (modified: 23 Jul 2026, 23:38)

### Summary

MSA-T2M proposes a framework coupling multi-scale semantic alignment with retrieval-augmented generation (RAG) for text-to-motion synthesis. The key insight is an alignment-realism trade-off: forcing motion latents toward text distributions improves R-Precision but distorts the motion manifold, hurting FID. The solution has two stages—MSA-VAE performs dual-scale alignment (local frame-level via BABEL, global sequence-level via caption interpolation) to structure the latent space as a semantic index; RAG-Diffusion-AR then retrieves semantically similar priors from this bank to anchor generation back to the natural motion manifold. On HumanML3D-272, MSA-T2M achieves SOTA FID (10.826, 8.1% over MotionStreamer) with competitive R-Precision (0.659).

### Contribution Type

General: Most submissions will fall into this type.

### Strengths And Weaknesses

**Strengths**

1. **The alignment-realism trade-off is a genuinely useful framing.** I hadn't seen this tension articulated so clearly before—it's one of those observations that feels obvious in retrospect. Figures 1 and 2 communicate it beautifully.
2. **Clean ablation.** Table 2 tells a crisp story: local alignment structures the space, global alignment boosts retrieval quality, and RAG pulls FID back down after alignment pushes it up. Each piece earns its place.
3. **Multi-scale alignment is well thought out.** Joint local (BABEL-supervised) and global (caption-supervised) alignment is more principled than the single-scale approaches I've seen in prior work, and the dynamic interpolation trick for cropped windows is a nice practical touch.
4. **Clear writing.** The three-stage curriculum makes sense, the loss functions are explicit, and I rarely had to flip back to earlier sections to understand what was happening.

**Weaknesses**

1. **Table 1 has only one continuous-latent baseline.** MoLingo (CVPR 2026) does latent-space semantic alignment and reports on the same MS-272 protocol—its absence is hard to justify. MARDM, MotionGPT3, COME, and SALAD are also continuous-latent works cited but not compared. With only MotionStreamer as the continuous-latent reference point, I can't confidently assess how much of the FID gain is real progress vs. protocol differences.
2. **No MSA-VAE reconstruction metrics.** The paper's central claim is that alignment distorts the motion manifold, but I can't see how much. MPJPE, foot skating, and acceleration error for the reconstruction-only baseline vs. the fully aligned model would tell me whether the distortion is serious enough to need the RAG fix.
3. **The RAG story is about FID, not retrieval quality.** Adding RAG actually drops R@1 from 0.669 to 0.659 (Table 2) while improving FID. That's consistent with the trade-off narrative, but it means the paper's contribution is the FID improvement, not the retrieval mechanism per se. Meanwhile, R@1 substantially trails ReMoDiffuse (0.659 vs. 0.718), and the diversity-based explanation for that gap is asserted rather than shown.
4. **BABEL dependency is a real constraint.** Local alignment needs frame-level action labels, which most motion datasets don't have. No experiment shows what happens without BABEL supervision, and this limits how broadly the method can be adopted.
5. **Single dataset, single runs.** HumanML3D-272 only, no KIT-ML, and point estimates without error bars for a three-stage pipeline that likely has non-trivial variance.

**Quality:** 3: good
**Clarity:** 3: good
**Significance:** 3: good
**Originality:** 2: not good

### Questions

1. MoLingo uses the same MS-272 protocol. Can the authors compare under that configuration, or at minimum explain what prevents it? A side-by-side with this and other continuous-latent methods (MARDM, MotionGPT3) would substantially strengthen the empirical claims.
2. What's the reconstruction quality of MSA-VAE before and after alignment? If alignment barely affects MPJPE, the trade-off claim is weaker than the paper suggests.
3. How does the method behave without BABEL? Even a single experiment on KIT-ML or a BABEL-free HumanML3D subset would tell me whether local alignment supervision is essential or just helpful.
4. I'd like to see evidence for the claim that MSA-T2M preserves diversity better than ReMoDiffuse. A diversity comparison or retrieval recall analysis would make the argument concrete.
5. What's the inference-time cost of top-K retrieval vs. retrieval-free generation, and how does it scale with bank size?

### Limitations

The authors are honest about BABEL dependency and offline-only generation. I'd suggest adding a few points that could help readers and future work:

(a) results on KIT-ML would give a more complete picture of generalization;
(b) the compute resources (GPU count, wall-clock time) aren't reported but matter for reproducibility;
(c) it would be interesting to discuss whether the alignment-realism trade-off framework might apply to other cross-modal generation tasks beyond motion.

**Rating:** 3: Borderline reject: Technically solid paper where reasons to reject, e.g., limited evaluation, outweigh reasons to accept, e.g., good evaluation. Please use sparingly.

**Confidence:** 5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

**Ethical Concerns:** NO or VERY MINOR ethics concerns only

**Paper Formatting Concerns:**

No major issues.

**Code Of Conduct Acknowledgement:** Yes

**Responsible Reviewing Acknowledgement:** Yes

---

## Meta Review by Area Chair Zgwo

**Date:** 22 Jul 2026, 17:41 (modified: 24 Jul 2026, 02:10)

### Metareview

**Summary of claims and findings** The paper proposes MSA-T2M, which combines multi-scale semantic alignment with retrieval-augmented autoregressive diffusion for text-to-motion generation. MSA-VAE aligns local motion latents with BABEL action annotations and a global CLS representation with sequence-level captions; the resulting CLS space is then used as a retrieval bank for RAG-Diffusion-AR. The paper’s main claim is that semantic alignment improves text-motion correspondence but distorts the motion manifold, while retrieval of real motion priors restores realism. On HumanML3D-272, the method reports the best FID among the included baselines and competitive text-motion alignment metrics.

**Strengths** The reviewers recognize several strengths. The alignment–realism trade-off is clearly motivated, and the coupling between semantic latent organization and retrieval-augmented generation is coherent. The local/global alignment design is intuitive for compositional motion descriptions, the three-stage training procedure is reasonably presented, and the reported FID improvement over MotionStreamer is promising.

**Weaknesses and missing evidence** However, the current submission is heading toward rejection due to the following central concerns: 1. The novelty relative to closely related work is not sufficiently established. The reviewers request a precise comparison with MotionBind and MoLingo, and clearer positioning relative to ReMoDiffuse and MotionStreamer, including which components are inherited, which are technically different, and what measurable gains arise from the proposed changes. 2. The central alignment–realism claim lacks direct validation. The reviewers request direct reconstruction evidence, such as MPJPE, foot skating, acceleration, or related measures, comparing reconstruction-only and aligned variants. 3. The quality and contribution of the aligned retrieval space are not directly demonstrated. The CLS representation is not independently evaluated for text-to-motion or motion-to-text retrieval. In addition, random or unaligned retrieval controls are missing, so it remains unclear whether the generation gains come from semantically relevant retrieval or simply from adding another conditioning token. 4. The ablation evidence is incomplete and partly inconsistent with the stated conclusions. The local-only result is missing; removing local alignment improves several R-Precision metrics; and adding RAG worsens FID in the global-only configuration. These results require explanation and currently do not fully support the claimed complementary contributions of local alignment, global alignment, and RAG. 5. Several methodological and reporting details remain unclear or inconsistent. These include the mapping from BABEL segment annotations to latent steps, the motivation for the Semantic Encoder/Decoder and the interpolated global target, the zero-pose termination criterion, conflicting K settings, and discrepancies between Table 1 and the main text. Additional concerns include limited baselines, single-dataset evaluation, and missing inference-cost analysis

**Issues that could change the acceptance assessment** The author response could significantly affect the decision if it provides: a convincing technical comparison with the closest prior work; direct evidence for the claimed alignment–realism trade-off; standalone retrieval evaluation and controlled retrieval ablations; a satisfactory explanation or completion of the key ablations; and a clear resolution of the methodological and numerical inconsistencies.
