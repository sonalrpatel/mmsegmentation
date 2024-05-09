# Copyright (c) OpenMMLab. All rights reserved.
import configargparse
from mmengine.model import revert_sync_batchnorm
from mmseg.apis import inference_model, init_model, show_result_pyplot


def main():
    parser = configargparse.ArgParser()
    parser.add('--img', default='demo/demo1.png', help='Image file')
    
    # parser.add_argument('--config', default='configs/pspnet/pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py', help='Config file')
    # parser.add_argument('--checkpoint', default='pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth', help='Checkpoint file')
    
    parser.add_argument('--config', default='configs/bisenetv2/bisenetv2_fcn_4xb8-160k_cityscapes-1024x1024.py', help='Config file')
    parser.add_argument('--checkpoint', default='bisenetv2_fcn_4x8_1024x1024_160k_cityscapes_20210903_000032-e1a2eed6.pth', help='Checkpoint file')
    
    parser.add_argument('--out-file', default='demo/results/result11.jpg', help='Path to output file')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--opacity', type=float, default=0.5, help='Opacity of painted segmentation map. In (0, 1] range')
    parser.add_argument('--with-labels', action='store_true', default=False, help='Whether to display the class labels')
    parser.add_argument('--title', default='result', help='The image identifier')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)
    if args.device == 'cpu':
        model = revert_sync_batchnorm(model)
    # test a single image
    result = inference_model(model, args.img)
    # show the results
    show_result_pyplot(
        model,
        args.img,
        result,
        title=args.title,
        opacity=args.opacity,
        with_labels=args.with_labels,
        draw_gt=False,
        show=False if args.out_file is not None else True,
        out_file=args.out_file)


if __name__ == '__main__':
    main()
