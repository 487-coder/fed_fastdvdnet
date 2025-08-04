import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # Data specifc paremeters
    parser.add_argument('--train_dataset', default='./mp4',
                        help='train file root')
    parser.add_argument('--client_numbers', type= int, default=10,
                        help='client number')
    parser.add_argument('--test_dataset', default='./480p',
                        help='test file root')
    parser.add_argument('--patch_size',type= int, default=96,
                        help='Patch size')
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Training batch size")
    parser.add_argument("--client_noise_level",nargs=2, type=int,default=[5,55],
                        help="Noise training interval")
    parser.add_argument("--client_noise_type", default="Gaussian",
                        help="noise type")
    parser.add_argument('--upload_model', action="store_true", default=False,
                        help='allow clients to upload models to the server')
    parser.add_argument('--test_noise',type = float, default=25,
                        help='noise level used on validation set')
    parser.add_argument("--temp_psz", "--tp", type=int, default=5,
                        help="Temporal patch size")
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='number of epochs')
    parser.add_argument('--sampling_rate', type=float, default=1,
                        help='frac of local models to update')
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Initial learning rate")
    parser.add_argument('--local_ep', type=int, default=5,
                        help='iterations of local updating')
    parser.add_argument('--clip_grad', type=float, default=None,
                        help='gadient clipping')
    parser.add_argument('--save_dir',  default=None,
                        help='save model path')
    args = parser.parse_args()
    return args