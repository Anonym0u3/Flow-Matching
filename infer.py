import torch
from rectified_flow import RectifiedFlow
import cv2
import os
from help_function import instantiate_from_config
from omegaconf import OmegaConf

def infer(
        checkpoint_path,
        model_config,
        steps=50,
        num_imgs=5,
        y=None,
        cfg_scale=7.0,
        save_path='./results',
        device='cuda'):
    
    os.makedirs(save_path, exist_ok=True)
    if y is not None:
        assert len(y.shape) == 1 or len(y.shape) == 2, 'y must be 1D or 2D tensor'
        assert y.shape[0] == num_imgs or y.shape[0] == 1, 'y.shape[0] must be equal to num_imgs or 1'
        if y.shape[0] == 1:
            y = y.repeat(num_imgs, 1).reshape(num_imgs)
        y = y.to(device)

    model = instantiate_from_config(OmegaConf.load(model_config))
    model.to(device)
    model.eval()

    rf = RectifiedFlow()

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])

    with torch.no_grad():
        # 无条件或有条件生成图片
        for i in range(num_imgs):
            print(f'Generating {i}th image...')
            # Euler法间隔
            dt = 1.0 / steps

            # 初始的x_t就是x_0，标准高斯噪声
            x_t = torch.randn(1, 1, 28, 28).to(device)

            # 提取第i个图像的标签条件y_i
            if y is not None:
                y_i = y[i].unsqueeze(0)

            for j in range(steps):
                if j % 10 == 0:
                    print(f'Generating {i}th image, step {j}...')
                t = j * dt
                t = torch.tensor([t]).to(device)

                if y is not None:
                    # classifier-free guidance需要同时预测有条件和无条件的输出
                    # 利用CFG的公式：x = x_uncond + cfg_scale * (x_cond - x_uncond)
                    # 为什么用score推导的公式放到预测向量场v的情形可以直接用？ SDE ODE
                    v_pred_uncond = model(x=x_t, t=t)
                    v_pred_cond = model(x=x_t, t=t, c=y_i)
                    v_pred = v_pred_uncond + cfg_scale * (v_pred_cond -
                                                          v_pred_uncond)
                else:
                    v_pred = model(x=x_t, t=t)

                # 使用Euler法计算下一个时间的x_t
                x_t = rf.euler(x_t, v_pred, dt)

            # 最后一步的x_t就是生成的图片
            # 先去掉batch维度
            x_t = x_t[0]
            # 归一化到0到1
            # x_t = (x_t / 2 + 0.5).clamp(0, 1)
            x_t = x_t.clamp(0, 1)
            img = x_t.detach().cpu().numpy()
            img = img[0] * 255
            img = img.astype('uint8')
            cv2.imwrite(os.path.join(save_path, f'{i}.png'), img)

if __name__ == '__main__':
    # 每个条件生成10张图像
    # label一个数字出现十次
    y = []
    for i in range(10):
        y.extend([i] * 10)

    checkpoint_path = './checkpoints/c_v1/miniunet_49.pth'
    #checkpoint_path = './checkpoints/v1.1/miniunet_49.pth'
    model_config = './configs/model.yaml'
    infer(
        checkpoint_path,
        model_config,
        steps=100,
        num_imgs=100,
        y=torch.tensor(y),
        cfg_scale=5.0,
        save_path='./results_c',
        device='cuda')