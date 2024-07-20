import torch
import os
import time

from decord import VideoReader, cpu, gpu
from einops import rearrange
from torchvision.io import write_video
from torchvision import transforms
from fire import Fire
from prettytable import PrettyTable

from cv_vae.models.modeling_vae import CVVAEModel


def log_time_benchmark(ts_1, ts_2, ts_3, ts_4, ts_5):
    table = PrettyTable()
    table.field_names = ["Step", "Time (s)", "Time (ms)"]
    table.add_row(["Read video", round(ts_2 - ts_1, 2), round((ts_2 - ts_1) * 1000, 2)])
    table.add_row(["Encode video", round(ts_3 - ts_2, 2), round((ts_3 - ts_2) * 1000, 2)])
    table.add_row(["Read & Encode video", round(ts_3 - ts_1, 2), round((ts_3 - ts_1) * 1000, 2)])
    table.add_row(["Decode video", round(ts_4 - ts_3, 2), round((ts_4 - ts_3) * 1000, 2)])
    table.add_row(["Write video", round(ts_5 - ts_4, 2), round((ts_5 - ts_4) * 1000, 2)])
    print(table)


def load_video(video_path, frame_sample_rate, device, gpu_id, transform):
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
    return video_reader, video


def main(vae_path, video_path, save_path, height=576, width=1024,
         frame_sample_rate=1, device='cuda', gpu_id=-1, timeit=False):
    vae3d = CVVAEModel.from_pretrained(vae_path, subfolder="vae3d", torch_dtype=torch.float16)
    vae3d.requires_grad_(False).to(device)

    transform = transforms.Compose([
        transforms.Resize(size=(height, width))
    ])

    ts_1 = time.time()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    video_reader, video = load_video(video_path, frame_sample_rate, device, gpu_id, transform)

    ts_2 = time.time()

    print(f'Shape of input video: {video.shape}')
    latent = vae3d.encode(video).latent_dist.sample()

    ts_3 = time.time()

    print(f'Shape of video latent: {latent.shape}')
    results = vae3d.decode(latent).sample

    ts_4 = time.time()

    results = rearrange(results, '1 c t h w -> t h w c')
    results = ((torch.clamp(results, -1.0, 1.0) + 1.0) * 127.5).to('cpu', dtype=torch.uint8)

    write_video(save_path, results, fps=video_reader.get_avg_fps(), options={'crf': '10'})

    ts_5 = time.time()

    if timeit is True:
        log_time_benchmark(ts_1, ts_2, ts_3, ts_4, ts_5)


if __name__ == '__main__':
    Fire(main)
