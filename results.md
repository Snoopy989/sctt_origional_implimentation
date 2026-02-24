======================================================================
2026-02-09 12:42:49,349 - INFO - TRAINING COMPLETE - Saving final model and generating predictions
2026-02-09 12:42:49,349 - INFO - ======================================================================
2026-02-09 12:42:49,349 - INFO - Merging LoRA adapters and saving full model...
Writing model shards: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [01:02<00:00, 62.99s/it]
2026-02-09 12:43:52,514 - INFO - ✓ Full merged model saved to: ./final_model_sctt_results_LORA_10_epochs_Llama-2-13b-chat-hf
2026-02-09 12:43:52,515 - INFO - 
Running validation predictions...
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 171/171 [02:11<00:00,  1.30it/s]
2026-02-09 12:46:04,645 - INFO - ✓ Validation predictions saved (1710 samples)
2026-02-09 12:46:04,645 - INFO -   Metrics: {'test_loss': 0.010308998636901379, 'test_mse': 0.010308998636901379, 'test_rmse': 0.10153324156999588, 'test_corr': 0.7197179685601902, 'test_runtime': 132.1245, 'test_samples_per_second': 12.942, 'test_steps_per_second': 1.294}
2026-02-09 12:46:04,645 - INFO - 
Running test predictions...
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 342/342 [04:24<00:00,  1.29it/s]
2026-02-09 12:50:29,609 - INFO - ✓ Test predictions saved (3420 samples)
2026-02-09 12:50:29,609 - INFO -   Metrics: {'test_loss': 0.009861323051154613, 'test_mse': 0.009861323051154613, 'test_rmse': 0.0993041917681694, 'test_corr': 0.736372709670435, 'test_runtime': 264.9561, 'test_samples_per_second': 12.908, 'test_steps_per_second': 1.291}
2026-02-09 12:50:29,609 - INFO - 
Running heldout predictions...
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 107/107 [01:21<00:00,  1.31it/s]
2026-02-09 12:51:51,963 - INFO - ✓ Heldout predictions saved (1062 samples)
2026-02-09 12:51:51,963 - INFO -   Metrics: {'test_loss': 0.02332054264843464, 'test_mse': 0.02332054078578949, 'test_rmse': 0.15271064639091492, 'test_corr': 0.4299292305224519, 'test_runtime': 82.3492, 'test_samples_per_second': 12.896, 'test_steps_per_second': 1.299}
2026-02-09 12:51:51,963 - INFO - 
======================================================================
2026-02-09 12:51:51,963 - INFO - ALL TASKS COMPLETE!
2026-02-09 12:51:51,963 - INFO - ======================================================================
2026-02-09 12:51:51,963 - INFO - Final model: ./final_model_sctt_results_LORA_10_epochs_Llama-2-13b-chat-hf
2026-02-09 12:51:51,963 - INFO - Validation predictions: validation_output_sctt_results_LORA_10_epochs_Llama-2-13b-chat-hf.csv
2026-02-09 12:51:51,963 - INFO - Test predictions: test_output_sctt_results_LORA_10_epochs_Llama-2-13b-chat-hf.csv
2026-02-09 12:51:51,963 - INFO - Heldout predictions: heldoutprompt_output_sctt_results_LORA_10_epochs_Llama-2-13b-chat-hf.csv
ubuntu@129-213-31-47:~/sctt_origional_implimentation$ python compute_epoch_wise_results.py



###############
7b
##############
==========================================================================================
EPOCH PERFORMANCE (VALIDATION - Pearson r)
==========================================================================================
     split  epoch  pearson_r       p_value  n_samples                                                                                     source_file                    source_dir
validation     10   0.722830 1.916043e-276       1710              results\training\validation_output_inference_LORA_10_epochs_Llama-2-7b-chat-hf.csv              results\training
validation     10   0.552590 2.557075e-137       1710 results\training_cluster_good\validation_output_inference_LORA_10_epochs_Llama-2-7b-chat-hf.csv results\training_cluster_good
validation      1   0.248261  1.963790e-25       1710               results\training\validation_output_inference_LORA_1_epochs_Llama-2-7b-chat-hf.csv              results\training
==========================================================================================

BEST (validation): epoch=10
  Pearson r: 0.7228
  p-value:   0.000000
  Samples:   1710
Saved: epoch_performance_validation.csv

==========================================================================================
EPOCH PERFORMANCE (TEST - Pearson r)
==========================================================================================
split  epoch  pearson_r       p_value  n_samples                                                                               source_file                    source_dir
 test     10   0.742607  0.000000e+00       3420              results\training\test_output_inference_LORA_10_epochs_Llama-2-7b-chat-hf.csv              results\training
 test     10   0.573267 4.450750e-298       3420 results\training_cluster_good\test_output_inference_LORA_10_epochs_Llama-2-7b-chat-hf.csv results\training_cluster_good
==========================================================================================

BEST (test): epoch=10
  Pearson r: 0.7426
  p-value:   0.000000
  Samples:   3420
Saved: epoch_performance_test.csv

==========================================================================================
EPOCH PERFORMANCE (HELDOUT - Pearson r)
==========================================================================================
  split  epoch  pearson_r      p_value  n_samples                                                                                  source_file                    source_dir
heldout     10   0.492094 7.463784e-66       1062              results\training\heldout_output_inference_LORA_10_epochs_Llama-2-7b-chat-hf.csv              results\training
heldout     10   0.280166 1.322447e-20       1062 results\training_cluster_good\heldout_output_inference_LORA_10_epochs_Llama-2-7b-chat-hf.csv results\training_cluster_good
==========================================================================================

BEST (heldout): epoch=10
  Pearson r: 0.4921
  p-value:   0.000000
  Samples:   1062
Saved: epoch_performance_heldout.csv

##########################################################################################
RECOMMENDED EPOCH (based on VALIDATION Pearson r)
##########################################################################################
epoch:      10
pearson_r:  0.7228
source:     results\training\validation_output_inference_LORA_10_epochs_Llama-2-7b-chat-hf.csv
##########################################################################################
(.venv) PS C:\Users\Phillip\Documents\05 - KSU (MSAI)\Fall_2025\sctt_origional_work>