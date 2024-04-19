# TODO: Use this for responsiblity

# Use hooks in 

# def resp_loss(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    
#     # assume that each of the components has some degree of
#     # responsibility
#     t = t.clamp(min=0.0) + torch.rand_like(t) * 1e-6
#     r = t / t.sum(dim=-1, keepdim=True)
#     Nk = r.sum(dim=0, keepdim=True)
#     target_loc = (r * x[:,:,None]).sum(dim=0, keepdim=True) / Nk

#     target_scale = (r * (x[:,:,None] - target_loc) ** 2).sum(dim=0, keepdim=True) / Nk

#     cur_scale = torch.nn.functional.softplus(self._scale)
    
#     scale_loss = self._fit_loss(cur_scale, target_scale.detach())
#     loc_loss = self._fit_loss(self._loc, target_loc.detach())
#     return scale_loss + loc_loss
