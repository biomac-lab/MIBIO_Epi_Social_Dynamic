## random 10% of individuals always defect.
# python3 run/run_sims_game_test.py --network_type scale_free --network_name scale_free_1000 --num_nodes 1000 --type_sim global --game_strategy defect --fixed_fraction 0.1
# python3 run/run_sims_game_test.py --network_type scale_free --network_name scale_free_1000 --num_nodes 1000 --type_sim local --game_strategy defect --fixed_fraction 0.1

# ## random 25% of individuals always defect.
# python3 run/run_sims_game_test.py --network_type scale_free --network_name scale_free_1000 --num_nodes 1000 --type_sim global --game_strategy defect --fixed_fraction 0.25
# python3 run/run_sims_game_test.py --network_type scale_free --network_name scale_free_1000 --num_nodes 1000 --type_sim local --game_strategy defect --fixed_fraction 0.25

## random 50% of individuals always defect.
python3 run/run_sims_game_test.py --network_type scale_free --network_name scale_free_1000 --num_nodes 1000 --type_sim global --game_strategy defect --fixed_fraction 0.5
python3 run/run_sims_game_test.py --network_type scale_free --network_name scale_free_1000 --num_nodes 1000 --type_sim local --game_strategy defect --fixed_fraction 0.5

## random 10% of individuals always cooperate.
python3 run/run_sims_game_test.py --network_type scale_free --network_name scale_free_1000 --num_nodes 1000 --type_sim global --game_strategy cooperate --fixed_fraction 0.1
python3 run/run_sims_game_test.py --network_type scale_free --network_name scale_free_1000 --num_nodes 1000 --type_sim local --game_strategy cooperate --fixed_fraction 0.1

# ## random 25% of individuals always cooperate.
# python3 run/run_sims_game_test.py --network_type scale_free --network_name scale_free_1000 --num_nodes 1000 --type_sim global --game_strategy cooperate --fixed_fraction 0.25
# python3 run/run_sims_game_test.py --network_type scale_free --network_name scale_free_1000 --num_nodes 1000 --type_sim local --game_strategy cooperate --fixed_fraction 0.25

# ## random 50% of individuals always cooperate.
# python3 run/run_sims_game_test.py --network_type scale_free --network_name scale_free_1000 --num_nodes 1000 --type_sim global --game_strategy cooperate --fixed_fraction 0.25
# python3 run/run_sims_game_test.py --network_type scale_free --network_name scale_free_1000 --num_nodes 1000 --type_sim local --game_strategy cooperate --fixed_fraction 0.25
