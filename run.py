import replicate

output = replicate.run(
    "zurk/hunyuanvideo:95866b6bb3b06ce2dd705a0354d848bffd3861da495773bf7d3a32587e786dc3",
    input={
        "width": 1280,
        "height": 720,
        "prompt": "A cat walks on the grass, realistic style.",
        "flow_shift": 7,
        "flow_reverse": True,
        "video_length": 129,
        "guidance_scale": 6,
        "num_inference_steps": 50,
        "embedded_guidance_scale": 6
    }
)
print(output)
