## JoTR: A Joint Transformer and Reinforcement Learning Framework for Dialogue Policy Learning
**Authors**: Wai-Chung Kwan, Huimin Wang, Hongru Wang, Zezhong Wang, Xian Wu, Yefeng Zhang, Kam-Fai Wong.
Code of our PPTOD paper: [JoTR: A Joint Transformer and Reinforcement Learning Framework for Dialogue Policy Learning]()

## Setup the conda environment
```yaml
conda env create --name jotr --file environment.yml 
```

## Pretrain 
### MultiWoz
```yaml
n_samples=10000
python prefinetune.py \
--n_epochs 80 \
--learning_rate 3e-4 \
--warmup_step_ratio 0.1 --lr_decay \
--batch_size 32 \
--hidden_size 256 \
--n_heads 1 \
--n_encoder_layers 1 \
--n_decoder_layers 1 \
--context_layer 4 \
--context_head 1 \
--context_hidden_size 256 \
--max_resp_len 256 \
--n_samples $n_samples \
--patience 10 \
--character sys \
--add_encoder_type_embedding \
--add_decoder_pos_embedding \
--do_train \
--save_prefix "multiwoz_pretrained"
```
The explanation of some parameters:
```yaml
n_samples: How many turns used for pre-tuning. It's common to use 4000 or 40000.
--warmup_step_ratio: The learning rate is increasing linearly from 0 to the specified learning rate in n steps. n is determined by warmup_step_ratio * total_training_steps.
--lr_decay: The learning rate will decrease linearly to 0 towards the end of training.
--context_model_name: THe huggingface pretrained model name for the encoder. Only bert and alberta are supported now.
--hidden_size: The hidden size of the transformer model.
--n_heads: The number of heads used in the transformer model.
--n_encoder_layers: The number of layers in the encoder of the transformer.
--n_decoder_layers: The number of layers in the decoder of the transformer.
--patience: Training will be stopped if the slot_f1 has no improvement in previous n epochs.
--save_prefix: The prefix of the saving folder under saved/.
--model_name: The transformer pretraining model name. Most experiment uses `distilgpt2`.
--n_samples: Number of of dialogue sessions for pretraining.  Among the samples, 70% will be used for training and the remaining will be used for validation. Using 200 samples seem achieve fairly good performance.
--add_encoder_type_embedding: The type embedding used in the encoder to indicate the type of context (db, belief, user act, system act)
--add_decoder_pos_embedding: The type embedding used in the encoder to indicate the type of context (db, belief, user act, system act)
```

### SGD
```yaml
n_samples=10000
python sgd_prefinetune.py \
--n_epochs 80 \
--learning_rate 3e-4 \
--warmup_step_ratio 0.1 --lr_decay \
--batch_size 32 \
--context_layer 1 \
--context_head 1 \
--context_hidden_size 256 \
--hidden_size 256 \
--n_heads 1 \
--n_encoder_layers 1 \
--n_decoder_layers 1 \
--max_resp_len 256 \
--patience 10 \
--character sys \
--add_encoder_type_embedding \
--add_decoder_pos_embedding \
--do_train \
--n_samples $n_samples \
--from_scratch \
--save_prefix "sgd_pretrained" 
```

## Finetune with PPO
### MultiWoz
```yaml
prefinetuned_path="saved/multiwoz_pretrained" 
alr=5e-7
clr=5e-7
minibatch=16
collect=1024

python ppo.py \
--prefinetuned_path $prefinetuned_path \
--remove_dropout \
--actor_learning_rate $alr \
--critic_learning_rate $clr \
--grad_clip 1 \
--critic_hidden_sizes 256 256 \
--vf_coef 0.5 --et_coef 0 \
--minibatch_per_epoch $minibatch \
--epoch_per_update 10 \
--step_per_collect $collect \
--episode_per_test 1000 \
--update_per_test 128 \
--update_per_log 128 \
--gamma 0.95 \
--n_train_steps 50000 \
--save_prefix "ppo_multiwoz" \
--norm_adv --norm_value --recompute_adv \
--use_critic_transformer \
--rew_type aggressive \
--do_train
```

The explanation of some parameters:
```yaml
--vf_coef: The critic loss coefficient.
--minibatch_per_epoch: The number of miinibatches in one update epoch.
--epoch_uper_update: The number of update epoch in one update iteration.
--step_per_collect: The number of steps to collect in rollout before performing one update iteration.
--episode_per_test: The number of episodes for testing.
--update_per_test: The number of update steps to perform one testing. 
--update_per_log: The number of update steps to perform one logging.
--n_train_steps: The total number of steps/frames for the whole training.
--norm_adv: Normalize the advantage function.
--norm_value: Normalize the value approximation.
--recompute_adv: Recompute the advantage in every update.
--use_critic_transformer: Use a fresh transformer as critic.
--save_prefix: The folder name for saving.
--rew_type: The reward function to use. `aggressive` refers to using reward shaping, `normal` refers to without using reward shaping.
```

### SGD
```yaml
prefinetuned_path="saved/multiwoz_pretrained" 
alr=5e-7
clr=5e-7
minibatch=16
collect=1024

python ppo_sgd.py \
--prefinetuned_path $prefinetuned_path \
--remove_dropout \
--actor_learning_rate $alr \
--critic_learning_rate $clr \
--grad_clip 1 \
--critic_hidden_sizes 256 256 \
--vf_coef 0.5 --et_coef 0 \
--minibatch_per_epoch $minibatch \
--epoch_per_update 10 \
--step_per_collect $collect \
--episode_per_test 1000 \
--update_per_test 128 \
--update_per_log 128 \
--gamma 0.95 \
--n_train_steps 50000 \
--save_prefix "ppo_multiwoz" \
--norm_adv --norm_value --recompute_adv \
--use_critic_transformer \
--rew_type aggressive \
--do_train
```
