import os
import torch
from transformers import AutoTokenizer, pipeline, logging, TextStreamer, LlamaTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import time
import colorama
import argparse

colorama.init()

class LlamaChat:
    def __init__(self, model_path, model_basename, use_streamer=True) -> None:
        self.model_path = model_path
        self.tokenizer = LlamaTokenizer.from_pretrained(model_path)
        start_time = time.time()
        self.model = AutoGPTQForCausalLM.from_quantized(model_path,
                model_basename=model_basename,
                use_safetensors=True,
                trust_remote_code=True,
                device="cuda:0",
                use_triton=True,
                quantize_config=None)
        
        self.use_streamer = use_streamer
        
        self.B_INST, self.E_INST = "[INST]", "[/INST]"
        self.B_SYS, self.E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
        self.DEFAULT_SYSTEM_PROMPT = """\
Your name is Llama. You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.
You have wide knowledge in coding, math, science, and other topics.
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

        self.dialogs = [
            [

            ]
        ]
        os.system("clear")
        print(f"Model loaded in {time.time() - start_time} seconds")

    def get_response(self, prompt, max_length=2048, temperature=0.75):
        self.dialogs[0].append({
            "role": "user",
            "content": prompt,
        })
        
        if self.use_streamer:
            streamer = TextStreamer(self.tokenizer, True)
        else:
            streamer = None

        prompt = torch.tensor(self.make_template()).cuda()
        output = self.model.generate(inputs=prompt, temperature=temperature, max_length=max_length, do_sample=True, streamer=streamer)
        response = self.tokenizer.decode(output[0]).split(self.E_INST)[-1].strip()
        self.dialogs[0].append({
            "role": "assistant",
            "content": response,
        })
        print(colorama.Fore.CYAN + "Tokens: " + str(prompt.shape[1]) + colorama.Fore.RESET)

    
    def make_template(self):
        prompt_tokens = []
        for dialog in self.dialogs:
            if dialog[0]["role"] != "system":
                dialog = [
                    {
                        "role": "system",
                        "content": self.DEFAULT_SYSTEM_PROMPT,
                    }
                ] + dialog
            dialog = [
                {
                    "role": dialog[1]["role"],
                    "content": self.B_SYS
                    + dialog[0]["content"]
                    + self.E_SYS
                    + dialog[1]["content"],
                }
            ] + dialog[2:]
            assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
                [msg["role"] == "assistant" for msg in dialog[1::2]]
            ), (
                "model only supports 'system', 'user' and 'assistant' roles, "
                "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
            )
            dialog_tokens = sum(
                [
                    self.tokenizer.encode(
                        f"{self.B_INST} {(prompt['content']).strip()} {self.E_INST} {(answer['content']).strip()} ",
                        # bos=True,
                        # eos=True,
                    )
                    for prompt, answer in zip(
                        dialog[::2],
                        dialog[1::2],
                    )
                ],
                [],
            )
            assert (
                dialog[-1]["role"] == "user"
            ), f"Last message must be from user, got {dialog[-1]['role']}"
            dialog_tokens += self.tokenizer.encode(
                f"{self.B_INST} {(dialog[-1]['content']).strip()} {self.E_INST}",
                # bos=True,
                # eos=False,
            )
            prompt_tokens.append(dialog_tokens)

            return prompt_tokens

if __name__ == "__main__":
    model_path = "TheBloke/Llama-2-7b-Chat-GPTQ"
    model_basename = "gptq_model-4bit-128g"
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_new_tokens", type=int, default=4000)
    args = parser.parse_args()
    llama = LlamaChat(model_path, model_basename)
    while True:
        prompt = input(colorama.Fore.GREEN + "\n>> " + colorama.Fore.RESET)
        llama.get_response(prompt, args.max_new_tokens)
