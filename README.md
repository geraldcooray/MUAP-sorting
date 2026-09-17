# spike_sort — Fully Automated Intramuscular MUAP Sorting for Paediatric EMG

Code accompanying the manuscript *"Fully Automated Intramuscular MUAP Sorting
for Paediatric EMG"* (G. K. Cooray). This folder contains exactly the code
used to produce the pipeline, simulations, figures and tables reported in the
paper — it excludes all data (see **Data availability** below).

## Layout

```
spike_sort/     the core pipeline package (detection, features, clustering,
                post-processing, metrics, plotting)
*.py            top-level scripts that build the MUAP library, run the
                simulation study, the merge experiment and the experimental
                (clinical) analysis, and generate every figure in the paper
```

All top-level scripts are flat, sibling modules that `import` one another by
name (e.g. `import call_spike_sort_emg`), matching how they were developed and
run — keep them in one directory, and run them from that directory (or add it
to `PYTHONPATH`), rather than moving individual files elsewhere.

## The `spike_sort` package

| Module | Role |
|---|---|
| `detection.py` | band-pass/high-pass filtering, envelope-based spike detection, waveform re-extraction |
| `features.py` | PCA / wavelet-coefficient / t-SNE shape features, amplitude/duration/polarity features, DCT high-pass |
| `clustering.py` | $k$-means, Gaussian mixture, Student's-$t$ mixture clustering; silhouette sweep; firing-regularity reassignment |
| `isi_cleanup.py` | refractory-period cleanup |
| `latency_linkage.py` | latency-linkage merge of split MUAP types |
| `superposition.py` | matching-pursuit superposition detection |
| `l2_merge.py` | gap-statistic sequential merge used in the over-segmentation experiment |
| `muap_selection.py` | MUAP panel-review / selection helpers used when curating the template library |
| `metrics.py` | standard MUAP parameters (duration, amplitude, phases, turns, area) and discharge statistics |
| `plotting.py` | shared plotting helpers (galleries, feature-space panels, colour maps) |
| `io.py` | signal loading and sample-rate lookup |

`call_spike_sort_emg.py` is the thin orchestrator described in the Methods
("Overview of the sorting pipeline") that wires these stages together with a
single set of default parameters (`BASE_PARAMS`).

## Script &rarr; paper mapping

| Script | Produces |
|---|---|
| `joint_run_spike_sort.py`, `dct_muap_compare.py`, `build_muap_library_curated.py` | the 31-template curated MUAP library (§2.8, Fig. 1) |
| `number_library_gallery.py` | Figure 1 (numbered template gallery) |
| `simulate_emg_traces.py` | trace-synthesis machinery (gamma renewal discharge, pink noise, §2.8 "Trace synthesis") |
| `simulate_and_sort_curated.py`, `spikesort_simulated_traces.py` | generates the six $K \in \{3,5,10,15,20,30\}$ simulated traces and runs/evaluates all nine feature/clustering combinations (Table "Parameters used to synthesise...", Table "Simulation sorting results") |
| `make_manuscript_gallery_figs.py` | Figures "gallery_features", "gallery_tsne_clustering", "gallery_tsne_alltypes" |
| `plot_tsne_types.py`, `plot_tsne_voronoi.py` | clustering + fixed t-SNE Voronoi tessellation used by the merge experiment |
| `plot_tsne_sequential_merge.py`, `plot_tsne_voronoi_merge.py`, `muap_split_refinement.py` | the sequential gap-statistic merge (§"Merging over-segmented types") |
| `merge_effect_table.py` | Table "Effect of the fixed-Voronoi sequential gap-merge..." |
| `make_merge_figure.py`, `make_voronoi_merge_figure.py` | Figures "merge_n10", "voronoi_merge_n10" |
| `experimental_pipeline.py` | the fixed pipeline applied to the six clinical subjects (Table "Experimental sort per subject") |
| `experimental_kde_bycat.py` | Figure "exp_kde_bycat_types" (amplitude/duration by clinical category) |
| `experimental_power_vs_rate.py` | Figure "exp_power_vs_rate_kde" (aggregate power vs. firing rate, by category) |

Scripts from earlier exploratory analysis that are **not** part of the
pipeline reported in the paper (e.g. alternative library-building variants,
diagnostic/presentation scripts) are not included here.

## Dependencies

Python 3.11+ with `numpy`, `scipy`, `scikit-learn`, `PyWavelets`, `matplotlib`
and `pandas`. All stochastic components (PCA, $k$-means/mixture-model
initialisation, t-SNE, trace synthesis, waveform display sampling) are seeded
for reproducibility, as described in Methods ("Implementation").

## Data availability

**No data is included in this repository.** The MUAP template library is
built from real concentric-needle EMG recordings of six paediatric patients;
those recordings, the resulting template library, the simulated traces, and
the experimental (clinical) sort outputs are not distributed here because
they derive from real patient data. Reproducing the library-building and
experimental-analysis steps requires access to the corresponding raw EMG
recordings, which is arranged separately (contact the corresponding author).
The simulation study (`simulate_and_sort_curated.py` onward) can be run
against any equivalently-formatted MUAP template library.

## Citation

If you use this code, please cite the paper (see the manuscript for full
citation details).
