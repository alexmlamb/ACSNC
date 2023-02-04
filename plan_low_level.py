
import torch


def plan(state_init, state_goal, img_encoder, state2img, latent_forward, latent2state): 


    action = torch.zeros((1,2))
    return action


if __name__ == "__main__":

    s0 = torch.Tensor([[0.4, 0.4]])
    sg = torch.Tensor([[0.6,0.6]])

    img_encoder = lambda img: torch.zeros((1,256))
    state2img = lambda state: torch.zeros((1,100*100))
    latent_forward = lambda state,action: state*0.0
    latent2state = lambda latent: torch.zeros((1,2))

    plan(s0, sg, img_encoder, state2img, latent_forward, latent2state)

