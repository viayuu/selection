'''
This module defines common argument parsing functionality for the project.

'''

def obtain_all_settings(parser):
    # machine parameters
    parser.add_argument("--cuda", type=int, default=0, help="CUDA device number to use")
    parser.add_argument("--seed", type=int, default=3407, help="Random seed for reproducibility")

    # environment parameters for training
    parser.add_argument("--problem_size", type=int, default=100, help="problem size for training")
    parser.add_argument("--capacity", type=int, default=50, help="Vehicle capacity for training")

    # environment parameters for testing
    parser.add_argument("--problem_size_list", type=int, nargs='+', default=[100],
                        help="List of problem sizes to consider")
    parser.add_argument("--problem_type", type=str, default="acvrpb",
                        help="Type of problem to test, you can use prepared problem lists, tsplib, cvrplib_xxl, or customize problem sets")

    # model parameters
    parser.add_argument("--embedding_dim", type=int, default=128, help="Embedding dimension for the model")
    parser.add_argument("--encoder_layer_num", type=int, default=12, help="Number of encoder layers in the model")
    parser.add_argument("--ff_hidden_dim", type=int, default=512,
                        help="Hidden dimension for feed-forward layer in the model")
    parser.add_argument("--logit_clipping", type=float, default=50, help="Logit clipping value for the model")
    parser.add_argument("--eval_type", type=str, default="greedy", help="Evaluation type for the model",
                        choices=['sampling', 'greedy'])
    parser.add_argument("--no_demand_max1", action="store_true", help="Do not normalize demand to a maximum value of 1")

    # optimizer parameters
    parser.add_argument("--optimizer_type", type=str, default="AdamW", help="Optimizer type for the model",
                        choices=['AdamW', 'Adam'])
    parser.add_argument("--optimizer_lr", type=float, default=1e-4, help="Learning rate for the optimizer")
    parser.add_argument("--weight_decay", type=float, default=1e-6, help="Weight decay for the optimizer")
    parser.add_argument("--lr_decay_epoch", type=int, nargs='+', default=[451],
                        help="Epochs at which to decay the learning rate")

    # Training parameters for datasets
    parser.add_argument("--training_epochs", type=int, default=500, help="Total epochs for training")
    parser.add_argument("--batches_per_epoch", type=int, default=5, help="Steps per epoch")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training in stage 1")

    # Inference parameters for datasets
    parser.add_argument("--test_episodes", type=int, default=1000, help="Number of test episodes")
    parser.add_argument("--test_batch_size", type=int, default=1000, help="Batch size for testing")
    parser.add_argument("--disable_aug", action="store_true", help="Disable instance augmentation during testing")
    parser.add_argument("--aug_factor", type=int, default=8, help="Augmentation factor for testing")
    parser.add_argument("--aug_batch_size", type=int, default=1000, help="Batch size for augmented testing")