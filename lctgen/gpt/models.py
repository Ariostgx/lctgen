import abc
import pathlib
import os
import openai

folder = os.path.dirname(__file__)
print(folder)

# org_path = os.path.join(folder, 'api.org')
# api_path = os.path.join(folder, 'api.key')

# openai.organization = open(org_path).read().strip()
# openai.api_key = open(api_path).read().strip()

from lctgen.core.basic import BasicLLM
from lctgen.core.registry import registry

@registry.register_llm(name='codex')
class CodexModel(BasicLLM):
  def __init__(self, config):
    super().__init__(config)
    self.codex_cfg = config.LLM.CODEX
    prompt_path = os.path.join(folder, 'prompts', self.codex_cfg.PROMPT_FILE)
    self.base_prompt = open(prompt_path).read().strip()
    
    sys_prompt_file = self.codex_cfg.SYS_PROMPT_FILE
    if sys_prompt_file:
      sys_prompt_path = os.path.join(folder, 'prompts', sys_prompt_file)
      self.sys_prompt = open(sys_prompt_path).read().strip()
    else:
      self.sys_prompt = "Only answer with a function starting def execute_command."
  
  def prepare_prompt(self, query, base_prompt):
    extended_prompt = base_prompt.replace("INSERT_QUERY_HERE", query)
    return extended_prompt

  def llm_query(self, extended_prompt):
    if self.codex_cfg.MODEL == 'debug':
      resp = self.sys_prompt
    elif self.codex_cfg.MODEL in ("gpt-4", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"):
      responses = openai.ChatCompletion.create(
              model=self.codex_cfg.MODEL,
              messages=[
                  {"role": "system", "content": self.sys_prompt},
                  {"role": "user", "content": extended_prompt}
              ],
              temperature=self.codex_cfg.TEMPERATURE,
              max_tokens=self.codex_cfg.MAX_TOKENS,
              top_p = 1.,
              frequency_penalty=0,
              presence_penalty=0,
              )
      resp = responses['choices'][0]['message']['content']
    else:
      response = openai.Completion.create(
          model="code-davinci-002",
          temperature=self.codex_cfg.temperature,
          prompt=extended_prompt,
          max_tokens=self.codex_cfg.max_tokens,
          top_p=1,
          frequency_penalty=0,
          presence_penalty=0,
          best_of=self.codex_cfg.best_of,
      )
      resp = response['choices'][0]['text']
    
    return resp

  def post_process(self, response):
    return response
  
@registry.register_llm(name='null')
class NullCodex(CodexModel):
  def __init__(self, config):
    super().__init__(config)
  
  def llm_query(self, extended_prompt):
    # for debugging
    return extended_prompt
  