# linear
# python3 plots/plot_heatmaps_tests.py --network_type scale_free --network_name scale_free_1000 --num_nodes 1000 --type_sim global --type_hm R0 --beta_function linear
python3 plots/plot_heatmaps_tests.py --network_type scale_free --network_name scale_free_1000 --num_nodes 1000 --type_sim local --type_hm R0 --beta_function linear

# concave
python3 plots/plot_heatmaps_tests.py --network_type scale_free --network_name scale_free_1000 --num_nodes 1000 --type_sim global --type_hm R0 --beta_function concave
python3 plots/plot_heatmaps_tests.py --network_type scale_free --network_name scale_free_1000 --num_nodes 1000 --type_sim local --type_hm R0 --beta_function concave

# convex
python3 plots/plot_heatmaps_tests.py --network_type scale_free --network_name scale_free_1000 --num_nodes 1000 --type_sim global --type_hm R0 --beta_function convex
python3 plots/plot_heatmaps_tests.py --network_type scale_free --network_name scale_free_1000 --num_nodes 1000 --type_sim local --type_hm R0 --beta_function convex

# exponential
python3 plots/plot_heatmaps_tests.py --network_type scale_free --network_name scale_free_1000 --num_nodes 1000 --type_sim global --type_hm R0 --beta_function exponential
python3 plots/plot_heatmaps_tests.py --network_type scale_free --network_name scale_free_1000 --num_nodes 1000 --type_sim local --type_hm R0 --beta_function exponential

# s-shape
python3 plots/plot_heatmaps_tests.py --network_type scale_free --network_name scale_free_1000 --num_nodes 1000 --type_sim global --type_hm R0 --beta_function s-shape
python3 plots/plot_heatmaps_tests.py --network_type scale_free --network_name scale_free_1000 --num_nodes 1000 --type_sim local --type_hm R0 --beta_function s-shape