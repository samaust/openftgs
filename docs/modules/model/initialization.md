# Initialization steps 7–9 in `openftgs.model`

How `FreeTimeGaussians.from_init_points` turns init points into a model: steps 7–9 of §1.3 "Initialization" and the paragraph after them on points supplied through the public interface. Steps 1–6 belong to `openftgs.init` (M12).

## Order of work

1. The init points are validated as a `ftgs-init` file would be (`io.validate_init_points`, Q-22).
2. The budget: F frames of the sequence and B points per frame (D-24). With a single timestamp F is 1; otherwise it is rounded half up from the frame interval (Q-21). If F·B exceeds the cap N_max passed in, initialization stops with an error (Q-16).
3. If there are more points than F·B, a seeded uniform subset is kept, in its original order; with no more points than that, the generator isn't touched (Q-21).
4. Time centers, durations and velocities convert from raw to normalized time (E14; Q-27, Q-19). Missing velocities become zero, the "w/o 4D initialization" ablation (D-23). Missing durations become one frame interval (D-21).
5. Missing frame ids become the nearest init frame slot, ties to the lower slot (D-47, Q-21).
6. Scales: within each frame id, among the points kept in step 3, each point takes the mean distance to its nearest neighbours, floored and stored as a log, isotropic (D-25). A frame with fewer points than neighbours uses the ones it has; a lone point gets the floor (Q-21). Distances are computed by brute force in chunks, in float64.
7. The fixed initial values: opacity from `InitConfig.init_opacity`, identity quaternions, the DC band from the colors (`sh.rgb_to_sh0`), and zero higher bands at the requested SH degree (D-5, Q-16).

The per-init-frame subset of k_stride·B points belongs to `roma_init` (M12); the total cap here also covers points from other sources.
