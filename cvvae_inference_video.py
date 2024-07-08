import torch
import os

from decord import VideoReader, cpu, gpu
from einops import rearrange
from torchvision.io import write_video
from torchvision import transforms
from fire import Fire

from models.modeling_vae import CVVAEModel


def main(vae_path, video_path, save_path, height=576, width=1024, frame_sample_rate=1, device='cuda', gpu_id=-1):
    vae3d = CVVAEModel.from_pretrained(vae_path, subfolder="vae3d", torch_dtype=torch.float16)
    vae3d.requires_grad_(False).to(device)

    transform = transforms.Compose([
        transforms.Resize(size=(height, width))
    ])

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    video_reader_context = gpu(gpu_id) if gpu_id >= 0 else cpu(0)
    video_reader = VideoReader(video_path, ctx=video_reader_context)

    frame_indices = list(range(0, len(video_reader), frame_sample_rate))
    video = video_reader.get_batch(frame_indices).asnumpy()
    video = torch.tensor(video, device=device)
    video = rearrange(video, 't h w c -> t c h w')
    video = transform(video)
    video = rearrange(video, 't c h w -> 1 c t h w').half()
    frame_end = 1 + (len(video_reader) - 1) // 4 * 4
    video = video / 127.5 - 1.0
    video = video[:, :, :frame_end, :, :]

    print(f'Shape of input video: {video.shape}')
    latent = vae3d.encode(video).latent_dist.sample()

    print(f'Shape of video latent: {latent.shape}')
    results = vae3d.decode(latent).sample

    results = rearrange(results, '1 c t h w -> t h w c')
    results = ((torch.clamp(results, -1.0, 1.0) + 1.0) * 127.5).to('cpu', dtype=torch.uint8)

    write_video(save_path, results, fps=video_reader.get_avg_fps(), options={'crf': '10'})


if __name__ == '__main__':
    Fire(main)
