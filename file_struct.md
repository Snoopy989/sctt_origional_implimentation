sctt_origional_work/
│
├── README.md
├── CONVERGENCE_ANALYSIS.md
├── requirements.txt
├── .env
├── .gitignore
│
├── data/                           # All data files
│   ├── raw/                        # Original data
│   │   ├── all_sctt_jrt.csv
│   │   └── sctt_item-generalization_jrt.csv
│   ├── processed/                  # Generated splits
│   │   ├── training_items_sctt.csv
│   │   ├── validation_items_sctt.csv
│   │   ├── test_items_sctt.csv
│   │   └── heldoutprompt_items_sctt.csv
│   └── debug/                      # Duplicate analysis, etc.
│       ├── sctt_jrt_cleaned.csv
│       ├── d_dupes.csv
│       └── d_dupes_avg.csv
│
├── models/                         # Model checkpoints & adapters
│   └── Llama-2-7b_sctt_regression/
│       ├── adapter_config.json
│       └── adapter_model.safetensors
│
├── scripts/                        # All executable scripts
│   ├── training/
│   │   ├── run_sctt_llama2chat_lora.py
│   │   ├── run_sctt_llama2_lora.py
│   │   ├── run_sctt_llama2_13b_chat_lora.py
│   │   └── run_sctt_best_fits.py
│   ├── data/
│   │   ├── generate_datasets.py
│   │   └── dataprocessing.py
│   ├── evaluation/
│   │   ├── inference_sctt.py
│   │   ├── calculate_metrics.py
│   │   └── compute_epoch_wise_results.py
│   ├── debug/
│   │   ├── test_model_load.py
│   │   ├── debug_checkpoint_loading.py
│   │   └── find_best_epoch.py
│   └── setup/
│       ├── cuda_install.py
│       └── cuda_troubleshoot.py
│
├── src/                            # Reusable code modules
│   ├── __init__.py
│   ├── misc.py                     # Shared utilities
│   ├── lora_misc.py                # LoRA utilities
│   └── lora_misc_llama2_13b.py
│
├── phase1/                         # Phase 1 work (archived)
│   └── (move src_phase1 contents here)
│
├── phase2/                         # Phase 2: Semantic Entropy
│   ├── README_phase2.md
│   ├── requirements_phase2.txt
│   ├── phase2_semantic_entropy.py
│   ├── quick_test.py               # Convergence analysis
│   ├── compare_paradigms.py
│   └── phase2_visualizations.py
│
├── bin/                            # PowerShell/bash runners
│   ├── run_phase2_cuda.ps1
│   ├── run_convergence_analysis.ps1
│   ├── run_with_cuda.sh
│   └── upload_to_huggingface.py
│
├── results/                        # All outputs
│   ├── training/                   # Training outputs
│   │   ├── sctt_results_*/
│   │   └── tmp_trainer/
│   ├── phase2/                     # Phase 2 results
│   │   ├── results_regression_*.csv
│   │   ├── metrics_regression_*.json
│   │   ├── convergence_analysis_*.png
│   │   └── sequences_*.json
│   └── visualizations/
│
├── docs/                           # Documentation
│   ├── results.md
│   └── CONVERGENCE_ANALYSIS.md
│
└── notebooks/                      # Jupyter notebooks (if any)