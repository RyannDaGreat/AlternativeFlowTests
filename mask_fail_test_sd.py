import torch
import rp.git.CommonSource.stable_diffusion as sd

if not 's' in dir():
    s=sd.StableDiffusion('mps')

baboon=load_image('https://raw.githubusercontent.com/mikolalysenko/baboon-image/master/baboon.png'              ,use_cache=True)
bichon=load_image('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTtOiJ0HLzKYs_-O7JYh8vlEy6YUKuXkhjT4Q&s',use_cache=True)
baboon=cv_resize_image(baboon,(512,512))
bichon=cv_resize_image(bichon,(512,512))
bichon=as_rgb_image(as_float_image(bichon))
baboon=as_rgb_image(as_float_image(baboon))

white = baboon - baboon + 1.0
black = baboon - baboon + 0.0
circle = as_rgb_image(as_float_image(flat_circle_kernel(512)))
circle = as_rgb_image(as_float_image(get_checkerboard_image(512,512,tile_size=128,)))
circle=rotate_image(circle,45)
circle=crop_image(circle,512,512,'center')
circle=as_rgb_image(circle)
circle=as_float_image(circle)


def encode(x):
    return s.encode_img(as_torch_image(x, device="mps", dtype=torch.float32))

with torch.no_grad():
    display_image(bichon)
    
    e_baboon = encode(baboon)
    e_bichon = encode(bichon)
    #e_white  = encode(white)
    e_black  = encode(black)
    #e_circle = encode(circle)
    #e_alpha  = iblend(e_circle,e_black,e_white)
    #e_blend = e_alpha * e_baboon + (1-e_alpha) * e_bichon
    
    e_noncirc_baboon = encode((1-circle)*baboon)
    e_circ_bichon    = encode((  circle)*bichon)
    e_blend = (e_baboon-e_noncirc_baboon) + (e_bichon-e_circ_bichon) +e_black

    o_blend = s.decode_latent(e_blend)

    display_image(o_blend)
