import torch

def inverse_warp(img, depth, depth_pseudo, pose1, pose2, K, bg_mask=None):

    '''
    img: origin image of closest view
    depth: rendered depth of closest view
    depth_pseudo: rendered depth of pseudo view
    pose1: camera pose of closest view
    pose2: camera pose of pseudo view
    K: camera intrinsic matrix
    '''

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    _, H, W = img.shape
    y, x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
    x = x.float().to(img.device)
    y = y.float().to(img.device)
    z = depth_pseudo[0]
    
    x = (x - cx) / fx
    y = (y - cy) / fy
    coordinates = torch.stack([x, y, torch.ones_like(z)], dim=0)
    coordinates = coordinates * z
    
    coordinates = coordinates.view(3, -1)
    coordinates = torch.cat([coordinates, torch.ones_like(z).view(1, -1)], dim=0)
    pose = torch.matmul(pose1, torch.inverse(pose2))
    coordinates = torch.matmul(pose, coordinates)
    
    coordinates = coordinates[:3, :]
    coordinates = coordinates.view(3, H, W)
    x = fx * coordinates[0, :] / coordinates[2, :] + cx
    y = fy * coordinates[1, :] / coordinates[2, :] + cy

    grid = torch.stack([2.0*x/W - 1.0, 2.0*y/H - 1.0], dim=-1).unsqueeze(0).to('cuda')

    warped_img = torch.nn.functional.grid_sample(img.unsqueeze(0), grid, mode='nearest', padding_mode='zeros').squeeze(0).to('cuda') 
    warped_depth = torch.nn.functional.grid_sample(depth.unsqueeze(0), grid, mode='nearest', padding_mode='zeros').squeeze(0).to('cuda') 
    warped_bg_mask = None
    if not (bg_mask is None):
        warped_bg_mask = torch.nn.functional.grid_sample(bg_mask.unsqueeze(0).float(), grid, mode='nearest', padding_mode='zeros').squeeze(0).to('cuda') 
        warped_bg_mask = (warped_bg_mask > 0.5)
    mask_warp = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    mask_warp = mask_warp.to('cuda')
    
    warped_depth_clone = warped_depth.clone()
    warped_depth_max = warped_depth_clone.max()
    warped_depth_zero = (warped_depth_clone > 0)
    warped_depth[~warped_depth_zero] = 1e4
    warped_depth_min = warped_depth.min()
    norm_warped_depth = (warped_depth[0].detach().clone() - warped_depth_min) / (warped_depth_max - warped_depth_min)
    
    warped_depth[~warped_depth_zero] = 0
    norm_warped_depth[~warped_depth_zero[0]] = 0
    
    norm_depth_pseudo = (depth_pseudo[0].detach() - depth_pseudo[0].min()) / (depth_pseudo[0].max() - depth_pseudo[0].min())
    mask_depth = (torch.abs(norm_warped_depth - norm_depth_pseudo) < 0.3)
    mask_depth_strict = (torch.abs(norm_warped_depth - norm_depth_pseudo) < 0.1)
    
    mask_depth = mask_depth.to('cuda')
    mask_depth_strict = mask_depth_strict.to('cuda')
    mask = mask_warp & mask_depth
    mask = mask.to('cuda')
    
    warped_masked_img = warped_img * mask
    
    mask_inv = ~mask
    
    return {"warped_img": warped_img, "warped_depth": warped_depth, "mask_warp": mask_warp, "mask_depth": mask_depth, "mask": mask, "warped_masked_img": warped_masked_img, "mask_inv": mask_inv, "mask_depth_strict": mask_depth_strict, "warped_bg_mask": warped_bg_mask}
