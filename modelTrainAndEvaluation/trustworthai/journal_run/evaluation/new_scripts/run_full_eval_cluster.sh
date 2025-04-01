#!/bin/bash

## ssn
python run_model_evaluations.py --repo_dir=/home/s2208943/ipdis/WMH_UQ_assessment --ckpt_dir=/home/s2208943/ipdis/results/journal_models/cross_validated_models --eval_split='test' --script_loc=trustworthai/journal_run/evaluation/new_scripts/stochastic_model_full_eval.py --dataset=ed --overwrite=true --uncertainty_type=ssn --result_dir=trustworthai/journal_run/evaluation/results/cross_validated_results/in_domain_results --model_name=ssn_ens0_cv0

python run_model_evaluations.py --repo_dir=/home/s2208943/ipdis/WMH_UQ_assessment --ckpt_dir=/home/s2208943/ipdis/results/journal_models/cross_validated_models --eval_split='test' --script_loc=trustworthai/journal_run/evaluation/new_scripts/stochastic_model_full_eval.py --dataset=ed --overwrite=true --uncertainty_type=ssn --result_dir=trustworthai/journal_run/evaluation/results/cross_validated_results/in_domain_results --model_name=ssn_ens0_cv1

python run_model_evaluations.py --repo_dir=/home/s2208943/ipdis/WMH_UQ_assessment --ckpt_dir=/home/s2208943/ipdis/results/journal_models/cross_validated_models --eval_split='test' --script_loc=trustworthai/journal_run/evaluation/new_scripts/stochastic_model_full_eval.py --dataset=ed --overwrite=true --uncertainty_type=ssn --result_dir=trustworthai/journal_run/evaluation/results/cross_validated_results/in_domain_results --model_name=ssn_ens0_cv2

python run_model_evaluations.py --repo_dir=/home/s2208943/ipdis/WMH_UQ_assessment --ckpt_dir=/home/s2208943/ipdis/results/journal_models/cross_validated_models --eval_split='test' --script_loc=trustworthai/journal_run/evaluation/new_scripts/stochastic_model_full_eval.py --dataset=ed --overwrite=true --uncertainty_type=ssn --result_dir=trustworthai/journal_run/evaluation/results/cross_validated_results/in_domain_results --model_name=ssn_ens0_cv3

python run_model_evaluations.py --repo_dir=/home/s2208943/ipdis/WMH_UQ_assessment --ckpt_dir=/home/s2208943/ipdis/results/journal_models/cross_validated_models --eval_split='test' --script_loc=trustworthai/journal_run/evaluation/new_scripts/stochastic_model_full_eval.py --dataset=ed --overwrite=true --uncertainty_type=ssn --result_dir=trustworthai/journal_run/evaluation/results/cross_validated_results/in_domain_results --model_name=ssn_ens0_cv4

python run_model_evaluations.py --repo_dir=/home/s2208943/ipdis/WMH_UQ_assessment --ckpt_dir=/home/s2208943/ipdis/results/journal_models/cross_validated_models --eval_split='test' --script_loc=trustworthai/journal_run/evaluation/new_scripts/stochastic_model_full_eval.py --dataset=ed --overwrite=true --uncertainty_type=ssn --result_dir=trustworthai/journal_run/evaluation/results/cross_validated_results/in_domain_results --model_name=ssn_ens0_cv5


## ssn ens
python run_model_evaluations.py --repo_dir=/home/s2208943/ipdis/WMH_UQ_assessment --ckpt_dir=/home/s2208943/ipdis/results/journal_models/cross_validated_models --eval_split='test' --script_loc=trustworthai/journal_run/evaluation/new_scripts/stochastic_model_full_eval.py --dataset=ed --overwrite=true --uncertainty_type=ssn_ens --result_dir=trustworthai/journal_run/evaluation/results/cross_validated_results/in_domain_results --model_name=ssn_ens0_cv0

python run_model_evaluations.py --repo_dir=/home/s2208943/ipdis/WMH_UQ_assessment --ckpt_dir=/home/s2208943/ipdis/results/journal_models/cross_validated_models --eval_split='test' --script_loc=trustworthai/journal_run/evaluation/new_scripts/stochastic_model_full_eval.py --dataset=ed --overwrite=true --uncertainty_type=ssn_ens --result_dir=trustworthai/journal_run/evaluation/results/cross_validated_results/in_domain_results --model_name=ssn_ens0_cv1

python run_model_evaluations.py --repo_dir=/home/s2208943/ipdis/WMH_UQ_assessment --ckpt_dir=/home/s2208943/ipdis/results/journal_models/cross_validated_models --eval_split='test' --script_loc=trustworthai/journal_run/evaluation/new_scripts/stochastic_model_full_eval.py --dataset=ed --overwrite=true --uncertainty_type=ssn_ens --result_dir=trustworthai/journal_run/evaluation/results/cross_validated_results/in_domain_results --model_name=ssn_ens0_cv2

python run_model_evaluations.py --repo_dir=/home/s2208943/ipdis/WMH_UQ_assessment --ckpt_dir=/home/s2208943/ipdis/results/journal_models/cross_validated_models --eval_split='test' --script_loc=trustworthai/journal_run/evaluation/new_scripts/stochastic_model_full_eval.py --dataset=ed --overwrite=true --uncertainty_type=ssn_ens --result_dir=trustworthai/journal_run/evaluation/results/cross_validated_results/in_domain_results --model_name=ssn_ens0_cv3

python run_model_evaluations.py --repo_dir=/home/s2208943/ipdis/WMH_UQ_assessment --ckpt_dir=/home/s2208943/ipdis/results/journal_models/cross_validated_models --eval_split='test' --script_loc=trustworthai/journal_run/evaluation/new_scripts/stochastic_model_full_eval.py --dataset=ed --overwrite=true --uncertainty_type=ssn_ens --result_dir=trustworthai/journal_run/evaluation/results/cross_validated_results/in_domain_results --model_name=ssn_ens0_cv4

python run_model_evaluations.py --repo_dir=/home/s2208943/ipdis/WMH_UQ_assessment --ckpt_dir=/home/s2208943/ipdis/results/journal_models/cross_validated_models --eval_split='test' --script_loc=trustworthai/journal_run/evaluation/new_scripts/stochastic_model_full_eval.py --dataset=ed --overwrite=true --uncertainty_type=ssn_ens --result_dir=trustworthai/journal_run/evaluation/results/cross_validated_results/in_domain_results --model_name=ssn_ens0_cv5

## ind
python run_model_evaluations.py --repo_dir=/home/s2208943/ipdis/WMH_UQ_assessment --ckpt_dir=/home/s2208943/ipdis/results/journal_models/cross_validated_models --eval_split='test' --script_loc=trustworthai/journal_run/evaluation/new_scripts/stochastic_model_full_eval.py --dataset=ed --overwrite=true --uncertainty_type=ssn --result_dir=trustworthai/journal_run/evaluation/results/cross_validated_results/in_domain_results --model_name=ssn_ens0_cv0