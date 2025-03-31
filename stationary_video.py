import torch
import rp

vpath_ori, vpath_gen = [
    "dinkeroutput_copy23.mp4",
    "dinkeroutput_copy9.mp4",
]

vori, vgen = rp.load_videos(
    vpath_ori,
    vpath_gen,
    use_cache=True,
)

assert vori.shape==vgen.shape
VT,VH,VW,VC=vori.shape

if "flow_ori_rev" not in vars(): flow_ori_rev = rp.calculate_flows(vori[::-1], show_progress=True)[::-1] #IF THIS WORKS, ADD A REVERSE OPTION
flow_ori_rev=as_numpy_array(flow_ori_rev)

#####

if 'cum_flow_ori_rev' not in vars():
    cum_flow_ori_rev = [rp.accumulate_flows(flow_ori_rev[:i+1]) for i in eta(range(VT))]
    cum_flow_ori_rev = torch.tensor(cum_flow_ori_rev)

def scatter_add_mean(image,dx,dy):
    rp.validate_tensor_shapes(
        image="torch: C H W",
        dx   ="         H W",
        dy   ="         H W",
        C=3,
    )
    
    ARGB = rp.torch_scatter_add_image(
        image, dx, dy,
        relative=True,
        prepend_ones=True,
         interp="bilinear",
        #interp="floor",
    )
    RGB = ARGB[1: ]
    A   = ARGB[ :1]
    
    
    RGB = RGB / A
    
    return RGB

stationary_video = [
    as_numpy_image(scatter_add_mean(as_torch_image(frame), *rev_flow))
    for frame, rev_flow in eta(zip(vori, cum_flow_ori_rev),length=VT)
]


preview_video = labeled_videos(
    *list_transpose(
        [
            [vori, "Input Video"],
            [stationary_video, "Stationary Video"],
            [[vori[0]], "First Frame"],
        ]
    ),
    font="R:Futura",
    size=30,
    show_progress=True,
)
preview_video = horizontally_concatenated_videos(preview_video)
display_video(preview_video, loop=True)


