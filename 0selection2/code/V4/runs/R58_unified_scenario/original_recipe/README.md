# Original R58 Recipe: Preserved, Over Budget

This is the unchanged evidence for the first source-frozen R58 recipe, before
the timing-motivated deployment revision. All128 TRAIN-only preflights passed.
The label-only TRAIN-size-weighted point projection was181.637GPUh; its
257.806GPUh adjacent-endpoint reference is not a bound. These are projections,
not consumed hours.

The real CPU-only `labels` pipeline returned3, `skipped_over_budget`, before
deployment lock or any full column generation. No scenario was released and
no baseline was trained under this recipe.

Implementation and original evidence are also preserved at Git commit
`0dd2862c8` (earlier full qualification commit `2ba1cd5d2`). Source semantics
are classical B; original RF inference used8 augmentations and diffusion used
50 denoising steps with3 T2T rewrites. The revised timing candidate is NOT an
equivalent algorithm-quality acceleration or a matched selector improvement.

The parent directory will contain current qualifications and the final budget
decision. Files here are historical evidence and must not qualify changed code.
