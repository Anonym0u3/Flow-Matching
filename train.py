import torch
from help_function import instantiate_from_config, load_state_dict
import wandb
from omegaconf import OmegaConf
import torch
from help_function import instantiate_from_config, load_state_dict
from rectified_flow import RectifiedFlow
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import StepLR
import os


def train(config):
    #wandb.init(project="Flow", name="train_unconditional_1.1")
    wandb.init(project="Flow", name="train_conditional_1")
    config = OmegaConf.load(config)
    device = config.device
    dataset = MNIST("/hy-tmp/flowmatching/MNIST", download=True, transform=ToTensor())
    dataloader = DataLoader(dataset, batch_size=config.data.batch_size, shuffle=True)
    model = instantiate_from_config(OmegaConf.load(config.model.config))
    model.to(device)

     # 优化器加载 Rectified Flow的论文里面有的用的就是AdamW
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)
    # 学习率调整
    scheduler = StepLR(optimizer, step_size=config.lr_adjust_epoch, gamma=0.1)

    # RF加载
    rf = RectifiedFlow()

    # 记录训练时候每一轮的loss
    #loss_list = []

    # 一些文件夹提前创建
    os.makedirs(config.save_path, exist_ok=True)

    # 训练循环
    for epoch in range(config.epochs):
        for batch, data in enumerate(dataloader):
            x_1, c = data  # x_1原始图像，c是标签，用于CFG
            # 均匀采样[0, 1]的时间t randn 标准正态分布
            t = torch.rand(x_1.size(0))

            # 生成flow（实际上是一个点）
            x_t, x_0 = rf.create_flow(x_1, t)

            # 4090 大概占用显存3G
            x_t = x_t.to(device)
            x_0 = x_0.to(device)
            x_1 = x_1.to(device)
            t = t.to(device)

            optimizer.zero_grad()

            # 这里我们要做一个数据的复制和拼接，复制原始x_1，把一半的y替换成-1表示无条件生成，这里也可以直接有条件、无条件累计两次计算两次loss的梯度
            # 一定的概率，把有条件生成换为无条件的 50%的概率 [x_t, x_t] [t, t]
                         
            if config.use_cfg:
                x_t = torch.cat([x_t, x_t.clone()], dim=0)
                t = torch.cat([t, t.clone()], dim=0)
                c = torch.cat([c, -torch.ones_like(c)], dim=0)
                x_1 = torch.cat([x_1, x_1.clone()], dim=0)
                x_0 = torch.cat([x_0, x_0.clone()], dim=0)
                c = c.to(device)
            else: 
                c = None

            v_pred = model(x=x_t, t=t, c=c)

            loss = rf.mse_loss(v_pred, x_1, x_0)
            wandb.log({"loss": loss})

            loss.backward()
            optimizer.step()

            if batch % config.batch_print_interval == 0:
                print(f'[Epoch {epoch}] [batch {batch}] loss: {loss.item()}')

            #loss_list.append(loss.item())

        scheduler.step()

        if epoch % config.checkpoint_save_interval == 0 or epoch == config.epochs - 1:
            # 第一轮也保存一下，快速测试用，大家可以删除
            # 保存模型
            print(f'Saving model {epoch} to {config.save_path}...')
            save_dict = dict(model=model.state_dict(),
                             optimizer=optimizer.state_dict(),
                             epoch=epoch,
                             #loss_list=loss_list)
                            )
            torch.save(save_dict,
                       os.path.join(config.save_path, f'miniunet_{epoch}.pth'))


if __name__ == '__main__':
    train(config='./configs/train_config.yaml')



