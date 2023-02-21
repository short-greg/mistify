from mistify.integrations import fuzzy_optim, adaptive_max
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
