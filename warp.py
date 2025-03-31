import torch
import rp

[vpath_ori, vpath_gen] = [
    #'/Users/burgert/Downloads/download (11).mp4',
    #'/Users/burgert/Downloads/12891282_1440_720_30fps.mp4',
    "dinkeroutput_copy23.mp4",
    "dinkeroutput_copy9.mp4",
]

[vori, vgen] = rp.load_videos(
    vpath_ori,
    vpath_gen,
    use_cache=True,
)

[vori, vgen] = resize_videos_to_fit(
    resize_lists_to_fit([vori, vgen], 100),
    height=480,
    width=720,
)

vori=as_numpy_array(vori)
vgen=as_numpy_array(vgen)

assert vori.shape==vgen.shape
VT,VH,VW,VC=vori.shape

if "flow_ori_rev" not in vars(): flow_ori_rev = rp.calculate_flows(vori[::-1], show_progress=True)[::-1]
if "flow_gen"     not in vars(): flow_gen     = rp.calculate_flows(vgen      , show_progress=True)
flow_ori_rev=as_numpy_array(flow_ori_rev)
flow_gen    =as_numpy_array(flow_gen    )

#####

if 'cum_flow_ori_rev' not in vars():
    cum_flow_ori_rev = [rp.accumulate_flows(flow_ori_rev[:i+1]) for i in eta(range(VT))]
    cum_flow_ori_rev = torch.tensor(cum_flow_ori_rev)

if 'cum_flow_gen' not in vars():
    cum_flow_gen = rp.accumulate_flows(flow_gen, reduce=False)

def scatter_add_mean(image,dx,dy):
    image=as_torch_image(image)
    dx=torch.tensor(dx)
    dy=torch.tensor(dy)
    
    rp.validate_tensor_shapes(
        image="torch: C H W",
        dx   ="torch:   H W",
        dy   ="torch:   H W",
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
    as_numpy_image(scatter_add_mean(frame, *rev_flow))
    for frame, rev_flow in eta(zip(vori, cum_flow_ori_rev),length=VT)
]
stationary_video=as_numpy_array(stationary_video)

warp_video = [
    as_numpy_image(scatter_add_mean(frame, *flow))
    for frame, flow in eta(zip(stationary_video, cum_flow_gen),length=VT)
]

#display_video(horizontally_concatenated_videos(vgen,warp_video),loop=True)

stationary_video=np.nan_to_num(stationary_video) #Remove the poison AFTER calculating warp video hehe, easy hack to keep unknown regions unknown...


preview_video = labeled_videos(
    *list_transpose(
        [
            [vori, "Input Video"],
            [stationary_video, "Stationary Video"],
            [vori[:1], "First Frame"],
            [vori[:1] / 255 / 2 + stationary_video / 2, "Overlaid"],
            [vgen, "Target Video"],
            [warp_video, "Warped Video"],
        ]
    ),
    font="R:Futura",
    size=30,
    show_progress=True,
)

preview_video = [as_rgb_images(as_byte_images(x)) for x in eta(preview_video,'Converting dtypes')]

preview_video = tiled_videos(
    preview_video,
    length=4,
)

display_video(preview_video, loop=True)


