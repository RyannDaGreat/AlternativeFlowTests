import torch
import rp

vpath_ori, vpath_gen = [
    "dinkeroutput_copy23.mp4",
    "dinkeroutput_copy9.mp4",
]

vpath_ori, vpath_gen = [
    "/Users/ryan/Downloads/dinkeroutput_copy22.mp4",
    "/Users/ryan/Downloads/dinkeroutput_copy26.mp4",
]

vori, vgen = rp.load_videos(
    vpath_ori,
    vpath_gen,
    use_cache=True,
)

assert vori.shape==vgen.shape
VT,VH,VW,VC=vori.shape

#I flippa and I floppa
vori,vgen=(vgen,vori)

#vori, vgen = resize_lists(vori, vgen, length=9)

if "flow_ori" not in vars(): flow_ori = rp.calculate_flows(vori, show_progress=True)
if "flow_gen" not in vars(): flow_gen = rp.calculate_flows(vgen, show_progress=True)

cum_flow_ori = rp.accumulate_flows(flow_ori, reduce=False)
cum_flow_gen = rp.accumulate_flows(flow_gen, reduce=False)
cum_flow_ori = torch.tensor(cum_flow_ori)
cum_flow_gen = torch.tensor(cum_flow_gen)

cum_flow_delta = cum_flow_gen-cum_flow_ori

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
    
    ##DO INPAINTING
    #RGB=as_numpy_image(RGB)
    #RGB=cv_inpaint_image(RGB,mask=as_numpy_array(A[0]==0))
    
    return RGB

scatter_frames = (
    scatter_add_mean(rp.as_torch_image(frame), dx, dy)
    for frame, (dx, dy) in zip(vori, cum_flow_delta)
)

scatter_frames = (rp.as_numpy_image(x) for x in scatter_frames)

preview_frames = (
    rp.vertically_concatenated_images(ori, scat, gen)
    for ori, scat, gen in zip(vori, scatter_frames, vgen)
)


rp.display_video(
    rp.eta(preview_frames, length=VT),
    loop=True,
)
