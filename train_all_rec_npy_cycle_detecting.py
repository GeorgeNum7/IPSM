import os
import csv
import json
import yaml
import time

if __name__ == '__main__':
    
    overlay = True
    straight = True
    continue_flag = False
    detecting_dir = '/media/localdisk1/wqs/code/sparse3D/IPSM_release/exp/exp-dtu_undistorted-mask-0000/fewshot/dtu_undistorted/resolution-4/3_views/iter-10000'
    det_record_eval_dir = os.path.join(detecting_dir, 'record_eval.csv')
    
    ##################################################################################################################################################
    # load param
    ##################################################################################################################################################
    
    exp_setting = './configs/sds7.yaml'
    
    with open(exp_setting, 'r', encoding='utf-8') as f:
        result = yaml.load(f.read(), Loader=yaml.FullLoader)

    cycle_num = result['cycle_num']
    dataset = result['dataset']
    n_views = result['n_views']
    exp_idx_start = result['exp_idx_start']
    cuda_idx = result['cuda_idx']
    use_sd = result['use_sd']
    use_wsds = result['use_wsds']
    use_wsds_2 = result['use_wsds_2']
    use_sds = result['use_sds']
    use_sdsi = result['use_sdsi']
    use_sds_ori = result['use_sds_ori']
    use_lora = result['use_lora']
    use_lora_2 = result['use_lora_2']
    use_pixel = result['use_pixel']
    stop_iterations = result['stop_iterations']
    sh_interval = result['sh_interval']
    reset_param = result['reset_param']
    densify_grad_threshold = result['densify_grad_threshold']
    depth_weight = result['depth_weight']
    depth_pseudo_weight = result['depth_pseudo_weight']
    opacity_reset_interval = result['opacity_reset_interval']
    lambda_dssim = result['lambda_dssim']
    sample_pseudo_interval = result['sample_pseudo_interval']
    pixel_pseudo_weight = result['pixel_pseudo_weight']
    sd_pseudo_weight = result['sd_pseudo_weight']
    warp_sds_pseudo_weight = result['warp_sds_pseudo_weight']
    warp_sds_pseudo_weight_2 = result['warp_sds_pseudo_weight_2']
    sds_pseudo_weight = result['sds_pseudo_weight']
    resolution = result['resolution']
    warp_sds_guidance_scale = result['warp_sds_guidance_scale']
    resume = result['resume']
    resume_root_dir = result['resume_root_dir']

    iterations = result['iterations']
    sd_guidance_scale = result['sd_guidance_scale']
    num_inference_steps = result['num_inference_steps']
    lora_start_iter = result['lora_start_iter']
    sd_guidance_start_iter = result['sd_guidance_start_iter']
    warp_sds_guidance_start_iter = result['warp_sds_guidance_start_iter']
    pixel_guidance_start_iter = result['pixel_guidance_start_iter']
    start_sample_pseudo = result['start_sample_pseudo']
    
    ##################################################################################################################################################
    # detecting previous exp
    ##################################################################################################################################################
    
    if not straight:
        while True:
            print('detecting:', detecting_dir)
            if os.path.exists(det_record_eval_dir): break
            time.sleep(3600)
        time.sleep(3600)

    ##################################################################################################################################################
    # exp start
    ##################################################################################################################################################

    for exp_idx in range(exp_idx_start, exp_idx_start + cycle_num):
        
        ##################################################################################################################################################
        # train start
        ##################################################################################################################################################
    
        dataset_dir = f'/media/localdisk1/wqs/data/{dataset}'
        exp_name = f'exp-{dataset}-recnpy-{str(exp_idx).zfill(4)}'
        
        output_dir = f'./exp/{exp_name}/fewshot/{dataset}/resolution-{resolution}/{str(n_views)}_views/iter-{str(iterations)}'
        start_scene = 'flower'
        scene_list = sorted(os.listdir(dataset_dir)) # NOTE
        # scene_list = ['fern'] # test
        output_dataset_dir = output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        tot_record_dir = os.path.join(output_dir, 'record.csv')
        if(not continue_flag):
            with open(tot_record_dir, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['scene', 'iters', 'psnr', 'ssim', 'lpips'])
                f.close()
        
        for scene_name in scene_list:
            if(scene_name != start_scene and continue_flag == True): continue
            if(scene_name == start_scene): continue_flag = False
            print(scene_name)
            scene_dir = os.path.join(dataset_dir, scene_name)
            scene_out_dir = os.path.join(output_dataset_dir, scene_name)
            if(os.path.exists(scene_out_dir)):
                pycode = f'rm -rf {scene_out_dir}'
            os.makedirs(scene_out_dir, exist_ok=True)
            resume_dir = os.path.join(resume_root_dir, scene_name, 'chkpnt2000.pth')
            
            pycode = f'CUDA_VISIBLE_DEVICES={str(cuda_idx)} python train_record_npy.py -s {str(scene_dir)} -m {str(scene_out_dir)} --eval --n_views {str(n_views)} --iterations {str(iterations)} --depth_weight {str(depth_weight)} --depth_pseudo_weight {str(depth_pseudo_weight)} --opacity_reset_interval {str(opacity_reset_interval)} --start_sample_pseudo {str(start_sample_pseudo)} --sample_pseudo_interval {str(sample_pseudo_interval)} --images images_{str(resolution)} --lambda_dssim {str(lambda_dssim)} --my_debug --sh_interval {str(sh_interval)} --reset_param {str(reset_param)} --densify_grad_threshold {str(densify_grad_threshold)}'
            
            if resume:
                pycode = pycode + f' --start_checkpoint {resume_dir}'
            
            if stop_iterations != -1:
                pycode = pycode + f' --stop_iterations {str(stop_iterations)}'
            
            if use_sd:
                if use_lora:
                    pycode = pycode + f' --add_sd_guidance --guidance_scale {sd_guidance_scale} --num_inference_steps {num_inference_steps} --sd_pseudo_weight {sd_pseudo_weight} --use_lora --lora_start_iter {lora_start_iter} --sd_guidance_start_iter {sd_guidance_start_iter}'
                elif use_lora_2:
                    pass
                else:
                    pycode = pycode + f' --add_sd_guidance --guidance_scale {sd_guidance_scale} --num_inference_steps {num_inference_steps} --sd_pseudo_weight {sd_pseudo_weight} --sd_guidance_start_iter {sd_guidance_start_iter}'
            
            if use_wsds or use_wsds_2 or use_sds or use_sds_ori:
                if use_lora:
                    pycode = pycode + f' --add_warp_sds_guidance --guidance_scale {warp_sds_guidance_scale} --warp_sds_pseudo_weight {warp_sds_pseudo_weight} --use_lora --lora_start_iter {lora_start_iter} --warp_sds_guidance_start_iter {warp_sds_guidance_start_iter}'
                if use_lora_2:
                    pass
                if use_wsds:
                    pycode = pycode + f' --add_warp_sds_guidance --guidance_scale {warp_sds_guidance_scale} --warp_sds_pseudo_weight {warp_sds_pseudo_weight} --warp_sds_guidance_start_iter {warp_sds_guidance_start_iter}'
                if use_wsds_2:
                    pycode = pycode + f' --add_warp_sds_guidance_2 --guidance_scale {warp_sds_guidance_scale} --warp_sds_pseudo_weight_2 {warp_sds_pseudo_weight_2} --warp_sds_guidance_start_iter {warp_sds_guidance_start_iter}'
                if use_sds:
                    pycode = pycode + f' --add_sds_guidance --guidance_scale {warp_sds_guidance_scale} --sds_pseudo_weight {sds_pseudo_weight} --warp_sds_guidance_start_iter {warp_sds_guidance_start_iter}'
                    if use_sdsi:
                        pycode = pycode + ' --use_sdsi'
                if use_sds_ori:
                    pycode = pycode + f' --add_sds_guidance_ori --guidance_scale {warp_sds_guidance_scale} --sds_pseudo_weight {sds_pseudo_weight} --warp_sds_guidance_start_iter {warp_sds_guidance_start_iter}'
                    
            if use_pixel:
                pycode = pycode + f' --add_pixel_guidance --pixel_guidance_start_iter {pixel_guidance_start_iter} --pixel_pseudo_weight {pixel_pseudo_weight}'
            
            print(pycode)

            os.system(pycode)
            
            scene_record_dir = os.path.join(scene_out_dir, 'record.csv')
            with open(scene_record_dir, newline='') as csvfile:
                spamreader = csv.reader(csvfile)
                last_line = None
                for row in spamreader:
                    if(row != '\n'):
                        last_line = row
                        
            assert last_line is not None            
            
            with open(tot_record_dir, 'a+', newline='') as f:
                writer = csv.writer(f)
                row_data = []
                row_data.append(scene_name)
                row_data.extend(last_line)
                writer.writerow(row_data)
                f.close()
            
        continue_flag = False
        
        
        ##################################################################################################################################################
        # eval start
        ##################################################################################################################################################
        
        tot_record_dir = os.path.join(output_dir, 'record_eval.csv')
        
        if(not continue_flag):
            with open(tot_record_dir, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['scene', 'iters', 'psnr', 'ssim', 'lpips'])
                f.close()       
        
        for scene_name in scene_list:
            if(scene_name != start_scene and continue_flag == True): continue
            if(scene_name == start_scene): continue_flag = False
            if(scene_name == 'record.csv'): continue
            if(scene_name == 'record_eval.csv'): continue
            scene_dir = os.path.join(dataset_dir, scene_name)
            scene_out_dir = os.path.join(output_dataset_dir, scene_name)
            pycode = f'CUDA_VISIBLE_DEVICES={str(cuda_idx)} python render.py --eval --source_path {scene_dir} --model_path {scene_out_dir} --iteration {str(iterations)} --images images_{str(resolution)} --n_views {str(n_views)}'
            print(pycode)
            state = os.system(pycode)
            pycode = f'CUDA_VISIBLE_DEVICES={str(cuda_idx)} python metrics.py --source_path {scene_dir}  --model_path  {scene_out_dir} --iteration {str(iterations)}'
            print(pycode)
            state = os.system(pycode)
            
            json_rst_dir = os.path.join(scene_out_dir, 'results.json')

            with open(json_rst_dir, 'r') as f:
                content = json.load(f)   
            
            with open(tot_record_dir, 'a+', newline='') as f:
                writer = csv.writer(f)
                row_data = []
                row_data.append(scene_name)
                row_data.append(f'{str(iterations)}')
                row_data.append(content[f'ours_{str(iterations)}']['PSNR'])
                row_data.append(content[f'ours_{str(iterations)}']['SSIM'])
                row_data.append(content[f'ours_{str(iterations)}']['LPIPS'])
                writer.writerow(row_data)
                f.close()
            
