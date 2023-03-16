from mistify.integrations import fuzzy_optim, adaptive_max, fuzzy_optim2
import sys

if __name__ == '__main__':
    if len(sys.argv) == 1: exit()
    if sys.argv[1] == "theta": 
        fuzzy_optim.check_if_optimizes_theta()
    if sys.argv[1] == "adamax":
        
        adaptive_max.check_if_adamax_is_close_to_max()
        adaptive_max.check_if_adamax_on_is_close_to_max()
        adaptive_max.check_if_adamin_is_close_to_min()
        adaptive_max.check_if_adamin_on_is_close_to_min()
    if sys.argv[1] == "union_on":
        fuzzy_optim2.check_if_unionon_optimizes_x()
    if sys.argv[1] == "intersect_on":
        fuzzy_optim2.check_if_intersecton_optimizes_x()
    if sys.argv[1] == "maxmin2":
        fuzzy_optim2.check_if_maxmin2_optimizes_w()
    if sys.argv[1] == "minmax2":
        fuzzy_optim2.check_if_minmax2_optimizes_w()
    if sys.argv[1] == "minmax2_x":
        fuzzy_optim2.check_if_minmax2_optimizes_x()
    if sys.argv[1] == "maxmin3_w":
        fuzzy_optim2.check_if_maxmin3_optimizes_w()
    if sys.argv[1] == "maxmin3_w2":
        fuzzy_optim2.check_if_maxmin3_optimizes_w_with_two_dims()
    if sys.argv[1] == "maxmin3_x":
        fuzzy_optim2.check_if_maxmin3_optimizes_x()
    if sys.argv[1] == "minmax3_w":
        fuzzy_optim2.check_if_minmax3_optimizes_w()
    if sys.argv[1] == "minmax3_x":
        fuzzy_optim2.check_if_minmax3_optimizes_x()
    # if sys.argv[1] == "maxmin_x":
    #    fuzzy_optim2.check_if_maxmin_optimizes_theta_with_two_layers()
