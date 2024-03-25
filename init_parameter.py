from argparse import ArgumentParser

def init_model():
    parser = ArgumentParser()
    # data
    parser.add_argument("--data_dir", default='data', type=str,
                        help="The input data dir. Should contain the .csv files (or other data files) for the task.")
    parser.add_argument("--dataset", default=None, type=str, required=True, 
                        help="The name of the dataset to train selected.")
    # model
    parser.add_argument("--gpu_id", type=str, default='0', help="Select the GPU id.")
    
    parser.add_argument('--seed', type=int, default=0, help="Random seed for initialization.")

    parser.add_argument("--model_name", default="/workspace/model/bert-base-uncased", type=str, help="The path or name of the pre-trained BERT model.")
    
    parser.add_argument("--freeze_bert_parameters", action="store_true", help="Freeze the last parameters of BERT.")
    
    parser.add_argument("--save_model", action="store_true", help="Save trained model.")

    parser.add_argument("--save_model_path", default="./model_mtp", type=str,
                        help="Path to save model checkpoints. Set to None if not save.")
    # hyperparameters
    parser.add_argument("--feat_dim", default=768, type=int, help="The feature dimension.")

    
    parser.add_argument("--warmup_proportion", default=0.1, type=float)

    parser.add_argument("--momentum_factor", default=0.99, type=float, help="The weighting factor of the momentum BERT.")
    
    parser.add_argument("--rtr_prob", default=0.25, type=float,
                        help="Probability for random token replacement")

    # training and testing
    parser.add_argument("--train_batch_size", default=64, type=int,
                        help="Batch size for training.")
    
    parser.add_argument("--pretrain_batch_size", default=32, type=int,
                        help="Batch size for pre-training")
    
    parser.add_argument("--eval_batch_size", default=64, type=int,
                        help="Batch size for evaluation.")

    parser.add_argument("--num_train_epochs", default=20, type=float,
                        help="The training epochs.")
    
    parser.add_argument("--num_pretrain_epochs", default=100, type=float,
                        help="The pre-training epochs.")

    parser.add_argument("--lr_pre", default=5e-5, type=float,
                        help="The learning rate for pre-training.")
                    
    parser.add_argument("--temperature", default=0.07, type=float,
                        help="The temperature for dot product.")
    
    parser.add_argument("--lr", default=5e-5, type=float,
                        help="The learning rate for training.")  
    
    parser.add_argument("--wait_patient", default=20, type=int,
                        help="Patient steps for Early Stop in pretraining.") 
    
    parser.add_argument("--view_strategy", default="rtr", type=str,
                        help="Choose from rtr|shuffle|none")

    parser.add_argument("--update_per_epoch", default=1, type=int,
                        help="Update pseudo labels after certain amount of epochs")
    
    parser.add_argument("--topk", default=250, type=int,
                        help="Select topk nearest neighbors")
    
    parser.add_argument("--grad_clip", default=1.0, type=float,
                        help="Value for gradient clipping.")
    return parser
