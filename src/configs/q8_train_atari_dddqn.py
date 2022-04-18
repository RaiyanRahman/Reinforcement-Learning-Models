class config():
    # env config
    render_train     = False
    render_test      = False
    env_name         = "Pong-v0"
    overwrite_render = True
    record           = True
    high             = 255.

    # output config
    output_path  = "results/q8_train_atari_dddqn/"
    model_output = output_path + "model.weights"
    A_output = output_path + "A.weights"
    V_output = output_path + "V.weights"
    log_path     = output_path + "log.txt"
    plot_output  = output_path + "scores.png"
    record_path  = output_path + "monitor/"

    # model and training config
    load_path         = "results/q8_train_atari_dddqn/model.weights"
    A_load_path       = "results/q8_train_atari_dddqn/A.weights"
    V_load_path       = "results/q8_train_atari_dddqn/V.weights"
    num_episodes_test = 50
    grad_clip         = True
    clip_val          = 10
    saving_freq       = 250000
    log_freq          = 50
    eval_freq         = 250000
    record_freq       = 250000
    soft_epsilon      = 0.05

    # nature paper hyper params
    nsteps_train       = 3100000
    batch_size         = 32
    buffer_size        = 1000000
    target_update_freq = 10000
    gamma              = 0.99
    learning_freq      = 4
    state_history      = 4
    skip_frame         = 4
    lr_begin           = 0.00008
    lr_end             = 0.00005
    lr_nsteps          = 500000
    eps_begin          = 0.5
    eps_end            = 0.1
    eps_nsteps         = 1000000
    learning_start     = 100000
