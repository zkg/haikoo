# haikoo
Notes and dataset from the paper "Haiku Generation A Transformer Based Approach With Lots Of Control"

In "Fine-tuning GPT-2 on Haikus.ipynb" you'll find everything your need to download a pretrained GPT-2 model and fine-tune it using Hugging Face.
"haiku_utils.ipynb" contains some eval functions and utilities to quickly conjure a bag of words on a given domain.

Head over to [PPLM](https://github.com/uber-research/PPLM) and setup their code. Once you are able to run their demo successfully, simply run:

	python run_pplm.py -B bag_of_words --cond_text "Starting text" --length 50 --gamma 1.5 --num_iterations 3 --num_samples 10 --stepsize 0.03 --window_length 5 --kl_scale 0.01 --gm_scale 0.99 --colorama --sample --pretrained_model "/home/username/haiku/gpt2-haiku/" --seed $(shuf -i 1-999 -n 1)

Make sure to replace _bag_of_words_ with whatever file you wish to link (just look into PPLM-master/paper_code/wordlists); _Starting text_ with whatever prompt you wish to initiate your poem with. And of course, make sure to point _pretrained_model_ to the relevant directory. Please refer to the paper for additional considerations, especially about _stepsize_.

Hopefully you should get some inspiration.
