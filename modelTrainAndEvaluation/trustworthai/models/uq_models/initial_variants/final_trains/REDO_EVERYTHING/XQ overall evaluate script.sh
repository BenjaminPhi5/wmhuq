run_experiment -b sbatch_run_bigmem.sh -e XQ_eval_experiment_ensemble.txt -m 50
run_experiment -b sbatch_run_bigmem.sh -e XQ_eval_experiment_evid_region.txt -m 50
# run_experiment -b sbatch_run_bigmem.sh -e XQ_eval_experiment_evid_sensoy.txt -m 50
run_experiment -b sbatch_run_bigmem.sh -e XQ_eval_experiment_mc_dropout.txt -m 50
run_experiment -b sbatch_run_bigmem_land04.sh -e XQ_eval_experiment_punet_muonly.txt -m 50
run_experiment -b sbatch_run_bigmem_land04.sh -e XQ_eval_experiment_punet.txt -m 50
run_experiment -b sbatch_run_bigmem.sh -e XQ_eval_experiment_ssn_ensemble.txt -m 50
run_experiment -b sbatch_run_bigmem.sh -e XQ_eval_experiment_ssn_independent_muonly.txt -m 50
# run_experiment -b sbatch_run_bigmem.sh -e XQ_eval_experiment_ssn_independent.txt -m 50
# run_experiment -b sbatch_run_bigmem.sh -e XQ_eval_experiment_ssn_mixture.txt -m 50
run_experiment -b sbatch_run_bigmem.sh -e XQ_eval_experiment_ssn.txt -m 50
