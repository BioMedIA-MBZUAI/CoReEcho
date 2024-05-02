import albumentations as A
import cv2
import numpy as np

def get_augmentation(
    n_frames=36,
):
    augmentation = A.Compose([
        A.PadIfNeeded(124, 124, border_mode=cv2.BORDER_CONSTANT, value=0, always_apply=True),
        A.RandomCrop(112, 112, always_apply=True),
    ])
    
    augmentation = VideoAlbumentations(n_frames, augmentation)
    
    return augmentation

class VideoAlbumentations:
    def __init__(
        self,
        n_frames,
        transform,
    ):
        self.n_frames = n_frames
        
        self.transform = transform
        self.transform.add_targets({
            **{f'image{i}': 'image' for i in range(self.n_frames)},
        })
    
    def __call__(
        self,
        video,
    ):
        if len(video) != self.n_frames:
            raise ValueError(f'Get {len(video)} frames but we expect {self.n_frames} frames ...')
        
        inputs = {f'image{i}': video[i] for i in range(self.n_frames)}
        inputs['image'] = video[0]
        
        outs = self.transform(**inputs)
        
        out_video = [outs[f'image{i}'][None, :] for i in range(self.n_frames)]
        out_video = np.concatenate(out_video, axis=0)
        
        return out_video