from transformers import AutoModelWithLMHead, AutoTokenizer
from transformers.modeling_gpt2 import GPT2LMHeadModel
from transformers import GPT2Tokenizer

pretrained_model="/home/username/gpt2-haiku3/"
model = GPT2LMHeadModel.from_pretrained(
        pretrained_model,
        output_hidden_states=True)
tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model)

# Padding text helps XLNet with short prompts - proposed by Aman Rusia in https://github.com/rusiaaman/XLNet-gen#methodology
PADDING_TEXT = """In 1991, the remains of Russian Tsar Nicholas II and his family
(except for Alexei and Maria) are discovered.
The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the
remainder of the story. 1883 Western Siberia,
a young Grigori Rasputin is asked by his father and a group of men to perform magic.
Rasputin has a vision and denounces one of the men as a horse thief. Although his
father initially slaps him for making such an accusation, Rasputin watches as the
man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous,
with people, even a bishop, begging for his blessing. <eod> </s> <eos>"""

prompt = "Another "
inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")

prompt_length = len(tokenizer.decode(inputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True))
for x in range(5):
	outputs = model.generate(inputs, max_length=250, do_sample=True, top_p=0.95, top_k=60)
	generated = prompt + tokenizer.decode(outputs[0])[prompt_length:]

	print(generated)
