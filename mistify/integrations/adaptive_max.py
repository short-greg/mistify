from .. import fuzzy
from .. import utils
from .. import core
import torch

torch.manual_seed(1)

def check_if_adamax_is_close_to_max():

    print("Adamax")
    t1 = torch.rand(4, 1)
    t2 = torch.rand(4, 4)
    print(torch.max(t1, t2))
    print(core.adamax(t1, t2))


def check_if_adamin_is_close_to_min():
    print("Adamin")

    t1 = torch.rand(4, 1)
    t2 = torch.rand(4, 4)
    print(torch.min(t1, t2))
    print(core.adamin(t1, t2))

def check_if_adamax_on_is_close_to_max():

    print("Adamax On")
    t = torch.rand(4, 4)
    print(torch.max(t, dim=1)[0])
    print(core.adamax_on(t, dim=1))


def check_if_adamin_on_is_close_to_min():
    print("Adamin On")

    t = torch.rand(4, 4)
    print(torch.min(t, dim=1)[0])
    print(core.adamin_on(t, dim=1))

