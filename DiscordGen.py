from aitextgen.TokenDataset import TokenDataset
from aitextgen.tokenizers import train_tokenizer
from aitextgen.utils import GPT2ConfigCPU
from aitextgen import aitextgen

file_name = "general2.txt"

train_tokenizer(file_name)
tokenizer_file = "aitextgen.tokenizer.json"
config = GPT2ConfigCPU()
ai = aitextgen(tokenizer_file=tokenizer_file, config=config)
data = TokenDataset(file_name, tokenizer_file=tokenizer_file, block_size=64)
ai.train(data, batch_size=8, num_steps=50000, generate_every=5000, save_every=5000)
ai.generate(10, prompt="Joe:")
ai2 = aitextgen(model_folder="trained_model",
                tokenizer_file="aitextgen.tokenizer.json")
ai2.generate(10, prompt="Joe:")
