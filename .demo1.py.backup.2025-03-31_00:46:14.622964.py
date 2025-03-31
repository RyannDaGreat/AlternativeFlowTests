import torch
import rp

vpath_ori, vpath_gen = [
    "dinkeroutput_copy23.mp4",
    "dinkeroutput_copy9.mp4",
]

vori, vgen = load_videos(
    vpath_ori,
    vpath_gen,
    use_cache=True,
)

vori,vgen=(vgen,vori)

#vori, vgen = resize_lists(vori, vgen, length=9)

if "flow_ori" not in vars():
    flow_ori = calculate_flows(vori, show_progress=True)

cum_flow_ori = accumulate_flows(flow_ori, reduce=False)
cum_flow_ori = torch.tensor(cum_flow_ori)


def scatter_add_mean(image,dx,dy):
    validate_tensor_shapes(
        image="torch: C H W",
        dx   ="         H W",
        dy   ="         H W",
        C=3,
    )
    
    ARGB = torch_scatter_add_image(
        image, dx, dy,
        relative=True,
        prepend_ones=True,
         interp="bilinear",
        #interp="floor",
    )
    RGB = ARGB[1: ]
    A   = ARGB[ :1]
    
    
    RGB = RGB / A
    
    ##DO INPAINTING
    #RGB=as_numpy_image(RGB)
    #RGB=cv_inpaint_image(RGB,mask=as_numpy_array(A[0]==0))
    
    return RGB

scatter_frames = (scatter_add_mean(as_torch_image(vori[0]), dx, dy) for dx, dy in cum_flow_ori)

display_video(
    scatter_frames,
    loop=True,
)
