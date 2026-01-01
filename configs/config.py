class TrainConfig:
    # reproducibility
    seed = 42
    use_cuda = True

    # data
    sample_ratio = 0.8
    train_batch_size = 32

    # model
    dim = 128
    input_dim = 128
    hidden_dim = 128
    output_dim = 3
    num_head = 4

    # training
    epoch_num = 100
    learning_rate = 1e-4
    weight_decay = 1e-5